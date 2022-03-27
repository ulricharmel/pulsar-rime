import jax.numpy as jnp
import numpy as np
from loguru import logger
import json
import line_profiler
profile = line_profiler.LineProfiler()

def load_model(modelfile, beamfile):
    """load model save a npy file.
        Array with shape (nsources x flux x ra x dec x emaj, emin x pa)
    Args:
        modelfile
            numpy array file
        beamfile 
            numpy array file
    Returns:
        dictionary with the intial parameters    
    """

    # we assume the model can be a json file if we just want continue with the fit
    assert modelfile.endswith('.npy')
    assert beamfile.endswith('.npy')

    beamgains = np.load(beamfile)
    nt = beamgains.shape[1]

    model = np.load(modelfile)
    stokes = model[:,0:nt]
    radec = model[:,nt:nt+2]
    shape_params = model[:,nt+2:nt+5]
    alpha = model[:,nt+5:]

    params = {}
    params["stokes"] = np.zeros((stokes.shape[0], stokes.shape[1], 4))
    params["stokes"][:,:,0] = stokes
    params["radec"]  = radec
    params["shape_params"] = shape_params
    params["alpha"] = alpha
    # params["beam"] = beamgains

   

    return params, beamgains







