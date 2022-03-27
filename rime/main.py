import sys
import numpy as np
import time

from contextlib import ExitStack
from loguru import logger
from rime import configure_loguru

from daskms import  xds_to_table
import dask
import dask.array as da
import dask.delayed as dd
from dask.graph_manipulation import clone


from dask.distributed import Client, LocalCluster, performance_report
from dask.diagnostics import ProgressBar

from rime.dask_wrappers import rime
from rime.parser import create_parser
from rime.read_data import load_model
from rime.chunkify import read_xds_list, make_stokes_beamgain_xds_list




# again, this only works on startup!
from jax.config import config
config.update("jax_enable_x64", True)

def _main(exitstack):

    configure_loguru("rime")
    
    logger.info("Running: pulsar-rime " + " ".join(sys.argv[1:]))

    t0 = time.time()
    
    parser = create_parser()
    args = parser.parse_args()

    # import pdb; pdb.set_trace()
    kw = vars(args)
    logger.info('Input Options:')
    for key in kw.keys():
        logger.info('     %25s = %s' % (key, kw[key]))

    params, beamgains = load_model(args.model, args.beam)

    # scheduler = "threads"
    # address = None
    # workers = 1
    # threads = 10

    chunks = dict(source=1, time=args.tchunk)

    ms_opts = dict(msname=args.msname, time_chunk=args.tchunk, freq_chunk=args.fchunk)
    data_xds_list, chunking_per_data_xds  = read_xds_list(ms_opts, args.freq0)
    stokes_xds_list, beam_xds_list = make_stokes_beamgain_xds_list(data_xds_list, params["stokes"], beamgains, chunks, chunking_per_data_xds)

    
    radec = da.from_array(params["radec"], chunks=(chunks["source"], 2))
    shape_params = da.from_array(params["shape_params"], chunks=(chunks["source"], 3))
    na = params["alpha"].shape[1]
    alpha = da.from_array(params["alpha"], chunks=(chunks["source"], na))

    outcols = [args.datacol]
    
    xds_list = []

    for xds, sds, bds in zip(data_xds_list, stokes_xds_list, beam_xds_list):
        
        uvw = xds.UVW.data
        freq = xds.CHAN_FREQ.data 
        stokes = sds.STOKES.data
        beamgain = bds.BEAM_GAINS.data
        vis = rime(radec, uvw, freq, shape_params, stokes, alpha, beamgain)

        dims = xds.DATA.dims  # All visiblity columns share these dims.
        data_vars = {"DATA": (dims, vis)}

        post_solve_data_xds = xds.assign(data_vars)

        xds_list.append(post_solve_data_xds)
    
    xds_list = [xds.drop_vars(["CHAN_FREQ", "CHAN_WIDTH"],
	                              errors='ignore')
				for xds in xds_list]
    
    xds_list = xds_to_table(xds_list, args.msname, columns=outcols)

    with ProgressBar():
        dask.compute(xds_list)
    
    
    ep_min, ep_hr = np.modf((time.time() - t0)/3600.)
    logger.success("{}hr{:0.2f}mins taken for predict".format(int(ep_hr), ep_min*60))
    
@logger.catch
def main():
    with ExitStack() as stack:
        _main(stack)
