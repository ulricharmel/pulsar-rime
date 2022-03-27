import argparse

def create_parser():
    p = argparse.ArgumentParser()
    
    p.add_argument("--msname", "-ms", type=str, dest="msname", required=True, help="measurement set")
    
    p.add_argument("--datacol", "-dc", dest="datacol",type=str, help="datacol to store visibilities", default="DATA")
    
    p.add_argument("--model", "-model", type=str, help="model file, numpy format for now", required=True)

    p.add_argument("--beam", "-beam", type=str, help="beam model file", required=True)

    p.add_argument("--tchunk", type=int, help="time chunk size", default=64)
    
    p.add_argument("--fchunk", default=64, type=int, help="frequency chunk size")

    p.add_argument("--freq0", "-freq0", type=float, 
                        help="Refrence frequence for spectral index", required=True)

    return p
