import numpy as np

from daskms import xds_from_ms, xds_from_table, xds_to_table
import dask
import dask.array as da
import dask.delayed as dd
from dask.graph_manipulation import clone

import xarray 

import rime.tools as RT


def compute_chunking(msname, tchunk, fchunk, group_by, compute=True):

    # Create an xarray data set containing indexing columns. This is
    # necessary to determine initial chunking over row and chan. TODO: Test
    # multi-SPW/field cases. Implement a memory budget.

    indexing_xds_list = xds_from_ms(
        msname,
        columns=("TIME", "INTERVAL"),
        index_cols=("TIME",),
        group_cols=group_by,
        chunks={"row": -1}
    )

    utime_chunking_per_xds, row_chunking_per_xds = row_chunking(
        indexing_xds_list,
       	tchunk,
        compute=False
    )

    spw_xds_list = xds_from_table(
        msname + "::SPECTRAL_WINDOW",
        group_cols=["__row__"],
        columns=["CHAN_FREQ", "CHAN_WIDTH"],
        chunks={"row": 1, "chan": -1}
    )

    chan_chunking_per_spw = chan_chunking(
        spw_xds_list,
        fchunk,
        compute=False
    )

    chan_chunking_per_xds = [chan_chunking_per_spw[xds.DATA_DESC_ID]
                             for xds in indexing_xds_list]

    zipper = zip(row_chunking_per_xds, chan_chunking_per_xds)
    chunking_per_data_xds = [{"row": r, "chan": c} for r, c in zipper]

    chunking_per_spw_xds = \
        [{"__row__": 1, "chan": c} for c in chan_chunking_per_spw.values()]

    if compute:
        return da.compute(utime_chunking_per_xds,
                          chunking_per_data_xds,
                          chunking_per_spw_xds)
    else:
        utime_chunking_per_xds, chunking_per_data_xds, chunking_per_spw_xds


def chan_chunking(spw_xds_list,
                  freq_chunk,
                  compute=True):
    """Compute frequency chunks for the input data.
    Given a list of indexing xds's, and a list of spw xds's, determines how to
    chunk the data in frequency given the chunking parameters.
    Args:
        indexing_xds_list: List of xarray.dataset objects contatining indexing
            information.
        spw_xds_list: List of xarray.dataset objects containing SPW
            information.
        freq_chunk: Int or float specifying chunking.
        compute: Boolean indicating whether or not to compute the result.
    Returns:
        A list giving the chunking in freqency for each SPW xds.
    """
    chan_chunking_per_spw = {}

    for ddid, xds in enumerate(spw_xds_list):

        # If the chunking interval is a float after preprocessing, we are
        # dealing with a bandwidth rather than a number of channels.

        if isinstance(freq_chunk, float):

            def interval_chunking(chan_widths, freq_chunk):

                chunks = ()
                bin_width = 0
                bin_nchan = 0
                for width in chan_widths:
                    bin_width += width
                    bin_nchan += 1
                    if bin_width > freq_chunk:
                        chunks += (bin_nchan,)
                        bin_width = 0
                        bin_nchan = 0
                if bin_width:
                    chunks += (bin_nchan,)

                return np.array(chunks, dtype=np.int32)

            chunking = da.map_blocks(interval_chunking,
                                     xds.CHAN_WIDTH.data[0],
                                     freq_chunk,
                                     chunks=((np.nan,),),
                                     dtype=np.int32)

        else:

            def integer_chunking(chan_widths, freq_chunk):

                n_chan = chan_widths.size
                freq_chunk = freq_chunk or n_chan  # Catch zero case.
                chunks = (freq_chunk,) * (n_chan // freq_chunk)
                remainder = n_chan - sum(chunks)
                chunks += (remainder,) if remainder else ()

                return np.array(chunks, dtype=np.int32)

            chunking = da.map_blocks(integer_chunking,
                                     xds.CHAN_WIDTH.data[0],
                                     freq_chunk,
                                     chunks=((np.nan,),),
                                     dtype=np.int32)

        # We use delayed to convert to tuples and satisfy daskms/dask.
        chan_chunking_per_spw[ddid] = dd(tuple)(chunking)

    if compute:
        return da.compute(chan_chunking_per_spw)[0]
    else:
        return chan_chunking_per_spw


def row_chunking(indexing_xds_list,
                 time_chunk,
                 compute=True):
    """Compute time and frequency chunks for the input data.
    Given a list of indexing xds's, and a list of spw xds's, determines how to
    chunk the data given the chunking parameters.
    Args:
        indexing_xds_list: List of xarray.dataset objects contatining indexing
            information.
        time_chunk: Int or float specifying chunking.
        compute: Boolean indicating whether or not to compute the result.
    Returns:
        A tuple of utime_chunking_per_xds and row_chunking_per_xds which
        describe the chunking of the data.
    """
    # row_chunks is a list of dictionaries containing row chunks per data set.

    row_chunking_per_xds = []
    utime_chunking_per_xds = []

    for xds in indexing_xds_list:

        # If the chunking interval is a float after preprocessing, we are
        # dealing with a duration rather than a number of intervals. TODO:
        # Need to take resulting chunks and reprocess them based on chunk-on
        # columns and jumps.

        # TODO: BDA will assume no chunking, and in general we can skip this
        # bit if the row axis is unchunked.

        if isinstance(time_chunk, float):

            def interval_chunking(time_col, interval_col, time_chunk):

                utimes, uinds, ucounts = \
                    np.unique(time_col, return_counts=True, return_index=True)
                cumulative_interval = np.cumsum(interval_col[uinds])
                cumulative_interval -= cumulative_interval[0]
                chunk_map = \
                    (cumulative_interval // time_chunk).astype(np.int32)

                _, utime_chunks = np.unique(chunk_map, return_counts=True)

                chunk_starts = np.zeros(utime_chunks.size, dtype=np.int32)
                chunk_starts[1:] = np.cumsum(utime_chunks)[:-1]

                row_chunks = np.add.reduceat(ucounts, chunk_starts)

                return np.vstack((utime_chunks, row_chunks)).astype(np.int32)

            chunking = da.map_blocks(interval_chunking,
                                     xds.TIME.data,
                                     xds.INTERVAL.data,
                                     time_chunk,
                                     chunks=((2,), (np.nan,)),
                                     dtype=np.int32)

        else:

            def integer_chunking(time_col, time_chunk):

                utimes, ucounts = np.unique(time_col, return_counts=True)
                n_utime = utimes.size
                time_chunk = time_chunk or n_utime  # Catch time_chunk == 0.

                utime_chunks = [time_chunk] * (n_utime // time_chunk)
                last_chunk = n_utime % time_chunk

                utime_chunks += [last_chunk] if last_chunk else []
                utime_chunks = np.array(utime_chunks)

                chunk_starts = np.arange(0, n_utime, time_chunk)

                row_chunks = np.add.reduceat(ucounts, chunk_starts)

                return np.vstack((utime_chunks, row_chunks)).astype(np.int32)

            chunking = da.map_blocks(integer_chunking,
                                     xds.TIME.data,
                                     time_chunk,
                                     chunks=((2,), (np.nan,)),
                                     dtype=np.int32)

        # We use delayed to convert to tuples and satisfy daskms/dask.
        utime_per_chunk = dd(tuple)(chunking[0, :])
        row_chunks = dd(tuple)(chunking[1, :])

        utime_chunking_per_xds.append(utime_per_chunk)
        row_chunking_per_xds.append(row_chunks)

    if compute:
        return da.compute(utime_chunking_per_xds, row_chunking_per_xds)
    else:
        return utime_chunking_per_xds, row_chunking_per_xds


def read_xds_list(ms_opts, freq0):
    """Reads a measurement set and generates a list of xarray data sets.
    Args:
        model_columns: A list of strings containing additional model columns to
            be read.
        ms_opts: A MSInputs configuration object.
    Returns:
        data_xds_list: A list of appropriately chunked xarray datasets.
    """

    # Determine all the chunking in time, row and channel.
    msname = ms_opts["msname"]
    time_chunk = ms_opts["time_chunk"]
    freq_chunk = ms_opts["freq_chunk"]
    group_by = ['FIELD_ID', 'DATA_DESC_ID', 'SCAN_NUMBER']

    field_xds = xds_from_table(msname+"::FIELD")[0]
    phase_dir = np.squeeze(field_xds.PHASE_DIR.values)
    field_names = field_xds.NAME.values

    print(f"Field table indicates phase centre is at ({phase_dir[0]} {phase_dir[1]}).")

    RT.ra0, RT.dec0 = phase_dir 
    RT.freq0 = freq0

    chunking_info = compute_chunking(msname, time_chunk, freq_chunk, group_by, compute=True)

    utime_chunking_per_data_xds = chunking_info[0]
    chunking_per_data_xds = chunking_info[1]
    chunking_per_spw_xds = chunking_info[2]

    # Once we have determined the row chunks from the indexing columns, we set
    # up an xarray data set for the data. Note that we will reload certain
    # indexing columns so that they are consistent with the chunking strategy.

    columns = ("TIME", "INTERVAL", "ANTENNA1", "ANTENNA2", "FEED1", "FEED2",
                "FLAG", "FLAG_ROW", "UVW", "DATA", "MODEL_DATA")
    # columns += (ms_opts.data_column,)
    # columns += (ms_opts.weight_column,) if ms_opts.weight_column else ()
    # columns += (ms_opts.sigma_column,) if ms_opts.sigma_column else ()
    # columns += (*model_columns,)

    available_columns = list(xds_from_ms(msname)[0].keys())
    assert all(c in available_columns for c in columns), \
            f"One or more columns in: {columns} is not present in the data."

    model_columns = ["DATA", "MODEL_DATA"]
    schema = {cn: {'dims': ('chan', 'corr')} for cn in model_columns}

    # known_weight_cols = ("WEIGHT", "WEIGHT_SPECTRUM")
    # if ms_opts.weight_column not in known_weight_cols:
    #     schema[ms_opts.weight_column] = {'dims': ('chan', 'corr')}

    data_xds_list = xds_from_ms(
        msname,
        columns=columns,
        index_cols=("TIME",),
        group_cols=group_by,
        chunks=chunking_per_data_xds,
        table_schema=["MS", {**schema}])

    spw_xds_list = xds_from_table(
        msname + "::SPECTRAL_WINDOW",
        group_cols=["__row__"],
        columns=["CHAN_FREQ", "CHAN_WIDTH"],
        chunks=chunking_per_spw_xds
    )

    _data_xds_list = []


    for xds_ind, xds in enumerate(data_xds_list):
        # Add coordinates to the xarray datasets.
        _xds = xds.assign_coords({"chan": np.arange(xds.dims["chan"])})

        # Add the actual channel frequecies to the xds - this is in preparation
        # for solvers which require this information. Also adds the antenna
        # names which will be useful when reference antennas are required.

        chan_freqs = clone(spw_xds_list[xds.DATA_DESC_ID].CHAN_FREQ.data)
        chan_widths = clone(spw_xds_list[xds.DATA_DESC_ID].CHAN_WIDTH.data)

        _xds = _xds.assign({"CHAN_FREQ": (("chan",), chan_freqs[0]),
                            "CHAN_WIDTH": (("chan",), chan_widths[0])})

        # Add an attribute to the xds on which we will store the names of
        # fields which must be written to the MS. Also add the attribute which
        # stores the unique time chunking per xds. We have to convert the
        # chunking to python integers to avoid problems with serialization.

        utime_chunks = tuple(map(int, utime_chunking_per_data_xds[xds_ind]))
        field_id = getattr(xds, "FIELD_ID", None)
        field_name = "UNKNOWN" if field_id is None else field_names[field_id]

        _xds = _xds.assign_attrs({"UTIME_CHUNKS": utime_chunks,
                                    "FIELD_NAME": field_name})

        _data_xds_list.append(_xds)


    data_xds_list = _data_xds_list


    return data_xds_list, chunking_per_data_xds


def make_stokes_beamgain_xds_list(data_xds_list, stokes, beamgains, chunks, chunking_per_data_xds):
    """Create a list of xarray.Datasets containing the stokes and the beamgains."""

    # This may need to be more sophisticated. TODO: Can we guarantee that
    # these only ever have one element?
    stokes_xds_list = []
    beamgains_xds_list = []
    nsrc = beamgains.shape[0]
    nc = stokes.shape[2]

    stokes = da.from_array(stokes, chunks=(chunks["source"], chunks["time"], nc))
    beamgains = da.from_array(beamgains, chunks=(chunks["source"], chunks["time"]))

    # import pdb; pdb.set_trace()

    for xds, chunk_xds in zip(data_xds_list, chunking_per_data_xds):

        bgains_stokes = da.blockwise(_make_stokes_beamgain, "stp",
                                 xds.TIME.data, "t",
                                 clone(beamgains), "sa",
                                 clone(stokes), "sac",
                                 nc, None,
                                 align_arrays=False,
                                 concatenate=True,
                                 dtype=np.float64,
                                #  adjust_chunks={"t": xds.UTIME_CHUNKS},
                                 new_axes={'p': nc+1})

        bgain_xds = xarray.Dataset(
            {
                "BEAM_GAINS": (("source", "time"), bgains_stokes[:,:,0])
            },
            coords={
                "source":np.arange(nsrc),
                "time": np.arange(np.sum(chunk_xds['row'])),
            },
            # attrs={
            #     "UTIME_CHUNKS": xds.UTIME_CHUNKS
            # }
        )

        beamgains_xds_list.append(bgain_xds)

        stokes_xds = xarray.Dataset(
            {
                "STOKES": (("source", "time", "stoke"), bgains_stokes[:,:,1:])
            },
            coords={
                "source":np.arange(nsrc),
                "time": np.arange(np.sum(chunk_xds['row'])),
                "stoke":np.arange(nc), 
            },
            # attrs={
            #     "UTIME_CHUNKS": xds.UTIME_CHUNKS
            # }
        )

        stokes_xds_list.append(stokes_xds)

    return stokes_xds_list, beamgains_xds_list


def _make_stokes_beamgain(time_col, beamgains, stokes, nc):
    """Handles the construction of the parallactic angles using measures.
    Args:
        time_col: Array containing time values for each row.
        beamgains: just repeats the beamgains
        stokes: flux values
    Returns:
        angles: Array of parallactic angles per antenna per unique time.
    """
    n_time = time_col.size
    nsrc = beamgains.shape[0]
    nc = stokes.shape[2]
    n_p = nc+1
 
    # Init angles from receptor angles. TODO: This only works for orthogonal
    # receptors. The more general case needs them to be kept separate.
    bgains = np.zeros((nsrc, n_time, n_p), dtype=np.float64)
   
    unique_times, indices = np.unique(time_col, return_index=True)
    indices = np.append(indices, n_time)

    for t in range(len(unique_times)):
        bgains[:, indices[t]:indices[t+1], 0] = beamgains[:, t]
        bgains[:, indices[t]:indices[t+1], 1:] = stokes[:, t, :]

    return bgains