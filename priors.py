import numpy as np

## --------------------------------------------------------------------- ##
## Computes a (l,r) unif with hyper.params (l,r).                         #
##

def interval_unif(params, hyper_params):
  delta = params
  r = hyper_params[1]
  l = hyper_params[0]
  if delta < l or delta > r: 
        return -np.inf
  else: 
      return 0

def interval_unif_multi(params, hyper_params):
  r = hyper_params[1]
  l = hyper_params[0]
  if any(params > r) or any(params < l): 
        return -np.inf
  else: 
      return 0

def unif_prior(params, hyper_params):
  if any(np.abs(params)>hyper_params): 
        return -np.inf
  else: 
      return 0
  
##     
## --------------------------------------------------------------------- ##



## --------------------------------------------------------------------- ##
## Computes inverse gamma distribution with hyper.params (alpha, beta).   #
##

def invGamma_prior(params, hyper_params):
  tau_sqd = params
  alpha = hyper_params[0]
  beta = hyper_params[1]
  if tau_sqd < 0:
      return -np.inf
  else:
      return (-alpha-1)*np.log(tau_sqd)-beta/tau_sqd

##     
## --------------------------------------------------------------------- ##







## --------------------------------------------------------------------- ##
##  Computes the Huser-wadsworth prior for R. (one dim)                    #
def huser_wadsworth_prior(params, hyper_params):
   R = params
   n_t = 1
   
   if R<1:
       return -np.inf
   else:
       dens = n_t*np.log((1-hyper_params)/hyper_params)-np.sum(np.log(R))/hyper_params  
       return dens

def R_powered_prior(params, hyper_params):
   x = params #R_powered with one element
   phi = hyper_params[0]
   s = hyper_params[1]  #gamma
   x_phi = x**(1/phi)
   
   if x<0:
       return -np.inf
   else:
       dens = np.log(s/(2 * np.pi))/2 - 3 * np.log(x_phi)/2 - s/(2 * x_phi) + (1/phi-1)*np.log(x)-np.log(phi)
   return dens

##
## --------------------------------------------------------------------- ##