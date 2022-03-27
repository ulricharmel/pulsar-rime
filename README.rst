Computing visibilities for a variable source

==============
Installation
==============
Installation from source_,
working directory where source is checked out

.. code-block:: bash
  
    $ pip install .

Pre release from git 

.. code-block:: bash
  
    $ pip install -e https://github.com/ulricharmel/pulsar-rime.git

To run and see options

.. code-block:: bash

    $ pulsar-rime --help 

    $ pulsar-rime --msname test.ms --datacol DATA --model pulsar/model.npy --beam pulsar/beam.npy --tchunk 72 --fchunk 1 --freq0 1e9

    $ pulsar-beam --force --primary-beam "output/cassbeam/JVLA-L-centred-\$(corr)_\$(reim).fits" --beam-clip 0.013 --pa-from-ms 3C147-DCB-HILO/B147-LO-NOIFS-NOPOL-4M5S.MS --app-to-int --verbose input/3C147-PTVLACC-spi-one.lsm.html input/3C147-PTVLACC-spi-int.lsm.html --bprefix testbgpython beam_variation.py --force --primary-beam "output/cassbeam/JVLA-L-centred-\$(corr)_\$(reim).fits" --beam-clip 0.013 --pa-from-ms 3C147-DCB-HILO/B147-LO-NOIFS-NOPOL-4M5S.MS --app-to-int --verbose input/3C147-PTVLACC-spi-one.lsm.html input/3C147-PTVLACC-spi-int.lsm.html --bprefix testbg

=======
License
=======

This project is licensed under the GNU General Public License v3.0 - see license_ for details.