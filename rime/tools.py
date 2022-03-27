import jax.numpy as jnp
from jax import jit, vmap
from jax import lax, ops
from jax.experimental import loops
from scipy.constants import c as lightspeed

# define some constants
deg2rad = jnp.pi / 180.0;
arcsec2rad = deg2rad / 3600.0;
uas2rad = 1e-6 * deg2rad / 3600.0;
rad2deg = 1./deg2rad
rad2arcsec = 1./arcsec2rad
deg2arcsec = deg2rad*rad2arcsec

minus_two_pi_over_c = 2*jnp.pi/lightspeed # remove -2
ra0 = 0 # overwite this in main
dec0 = 0

freq0 = 1e9 # reference frequency for source spectrum


@jit
def lm2radec(lm):
    #let say the ra and dec in radians
    source = lm.shape[0]
    radec = jnp.empty((source, 2), lm.dtype)
    
    sin_pc_dec = jnp.sin(dec0)
    cos_pc_dec = jnp.cos(dec0) 

    def body(s, radec):
        l,m = lm[s]
        n = jnp.sqrt(1.0 -l**2 -m**2)

        ra = ra0 + jnp.arctan2(l, (n*cos_pc_dec - m*sin_pc_dec))
        dec = jnp.arcsin(m*cos_pc_dec + n*sin_pc_dec)

        radec = ops.index_update(radec, (s, 0), ra)
        radec = ops.index_update(radec, (s, 1), dec)

        return radec
     
    return lax.fori_loop(0, source, body, radec)
        

@jit
def radec2lm(radec):
    source = radec.shape[0]
    lm = jnp.empty((source, 2), radec.dtype)

    def body(s, lm):
        ra, dec = radec[s]*deg2rad
        delta_ra = ra - ra0
        l = jnp.cos(dec)*jnp.sin(delta_ra)
        m = jnp.sin(dec)*jnp.cos(dec0) - jnp.cos(dec)*jnp.sin(dec0)*jnp.cos(delta_ra)
   
        lm = ops.index_update(lm, (s, 0), l)
        lm = ops.index_update(lm, (s, 1), m)

        return lm
  
    return lax.fori_loop(0, source, body, lm)

@jit
def source_spectrum(alpha, freqs):
    # for now we assume the refrencece frequency is the first frequency
    # freq0 is imported from tools.py and it value is updated from main
    frf = freqs/freq0
    logfr = jnp.log10(frf)
    
    spectrum = frf ** sum([a * jnp.power(logfr, n) for n, a in enumerate(alpha)])
    return spectrum[None, :]

