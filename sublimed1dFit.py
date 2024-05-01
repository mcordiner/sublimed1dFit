#! /usr/bin/env python

##############################################################
#  Levenberg-Marquardt least squares coma spectrum fitter.   #
#  Finds best-fitting SUBLIMED1D model.                      #
#  Based on sublimedFit_v2.py                                #
##############################################################

import os,sys
import numpy as np
from mpfit import mpfit
from scipy.interpolate import interp1d
import pylab as plt
from scipy.stats import chi2
from astropy.convolution import convolve, Gaussian1DKernel

if len(sys.argv) != 2: sys.exit("Usage: sublimed1dFit.py sublimed1dFit.par \n Observed spectrum must be in units of K vs. km/s")

c = 2.99792458e8

modelFile = "sublimed1dFit_input.par"


#####################################################################
# Function to interface with MPFIT and return normalised residuals  #
#####################################################################
def getChisq(p, fjac=None, functkw=None):

   global modelTI,mvel,mT
   
   rresiduals=[]
   model=[]
   
   print(p)
   
#  Parameters for this run            
   abund,tkin,vexp,vsource,dopplerb = p

#  Generate model image        
   sublimed1dModel(Q,abund,tkin,vexp,rNuc,betamol,rH,delta,lp,spec,chwid,nchan,lamFile,pumpFile,trans,collPartId,xne,vsource,dopplerb,npix,imgres,units,freqAx,modelFile)
   
   os.system("sublimed1dc %s" % (modelFile))
   mvel,mT=convImgSpec(fitsFile,hpbwx,hpbwy,pa,eta,units,xoff=0,yoff=0,freqAx=freqAx,drop=True,outfile=fitsFile[:-5]+"_conv_spec.txt",integrate=True)
   
   if rez != 0:
#  Convolve the model spectrum to the required resolution
      sigmachans = rezchans/2.3548
      mTc=convolve(mT, Gaussian1DKernel(sigmachans))
   else:
      mTc = mT
 
   modelTI=interp1d(mvel,mTc,kind='cubic',fill_value=0.,bounds_error=False)
   
#  Subtract model from observation to generate residuals  
   residuals = yobs - modelTI(xobs)
   
   status = 0
   
   return ([status, residuals/rms])


###########################
# Sigma clipping function #
###########################
def sigClip(img,nSigma,DELTARMS=1.01,posneg='both'):
    sigma = np.nanstd(img)
    sigma0 = sigma*2.
    imgmasked = img
    while sigma0/sigma > DELTARMS:
       if posneg == 'positive':
           imgmasked = np.ma.masked_where((img-np.nanmean(imgmasked))>nSigma*sigma,img)
       if posneg == 'negative':
           imgmasked = np.ma.masked_where((img-np.nanmean(imgmasked))<(-nSigma)*sigma,img)
       if posneg == 'both':
           imgmasked = np.ma.masked_where(np.abs(img-np.nanmean(imgmasked))>nSigma*sigma,img)       
       sigma0 = sigma
       sigma=np.nanstd(imgmasked)
       if imgmasked.mask.all(): print("All data points have been rejected, cannot continue")
    return imgmasked


#############################
# Write two-column spectrum #
#############################
def write2col(data1,data2,outfile):
    f = open(outfile,'w')
    for i in range(len(data1)):
        f.write("%12.8f  %14.7e\n" %(data1[i], data2[i]))
    f.close()

###############
# 2D Gaussian #
###############
def gauss2d(xlen,ylen,fwhm_x,fwhm_y,dx,theta=0.,x0=0.,y0=0.,norm=True):
    #Returns the value of the 2d elliptical gaussian for grid of size xlen x ylen (pixels). Theta defined as angle in radians from x axis to semimajor axis (CCW). dx is the pixel scale (in same units as fwhm). x0 and y0 are offsets in units of dx
    
    x,y = np.indices([xlen,ylen])
    x = dx * (x-(x.max()-x.min())/2.0)
    y = dx * (y-(y.max()-y.min())/2.0)

    
    sig_x = fwhm_x / (2*np.sqrt(2*np.log(2))) #sig is the standard deviation
    sig_y = fwhm_y / (2*np.sqrt(2*np.log(2)))
    
    if norm:
        # Normalize the Gaussian to unit volume by pixel number
        A = 1. / (2.*np.pi*sig_x*sig_y/dx**2)
    else:
        A = 1.   
    
    # Rotated elliptical Gaussian
    a1 = np.cos(theta)**2/(2*sig_x**2) + np.sin(theta)**2/(2*sig_y**2)
    b1 = -np.sin(2*theta)/(4*sig_x**2) + np.sin(2*theta)/(4*sig_y**2)
    c1 = np.sin(theta)**2/(2*sig_x**2) + np.cos(theta)**2/(2*sig_y**2)
    g = A*np.exp(-(a1*(x-x0)**2 - 2*b1*(x-x0)*(y-y0) + c1*(y-y0)**2))
    return g


####################################
# Write the sublimed1dc input file #
####################################
def sublimed1dModel(Q,abund,tkin,vexp,rNuc,betamol,rH,delta,lp,spec,chwid,nchan,lamFile,pumpFile,trans,collPartId,xne,vsource,dopplerb,npix,imgres,units,freqAx,modelFile):

   vexp = vexp * 1000.   # To convert to SI as required by LIME
   dopplerb = dopplerb * 1000.   # To convert to SI as required by LIME
   vsource = vsource * 1000.   # To convert to SI as required by LIME
   lp = lp * 1000. # To convert to SI as required by LIME

   # This change allows the model to be centered on an arbitrary frequency
   if freqAx:
      fString = "freq     = %d;   // Central channel frequency" %(trans * 1e9)
   else:
      fString = "trans    = %d;   // Zero-indexed J quantum number" % (trans)


   limeInput="""#SUBLIME-D1D-C input file

runname		= SUBLIMED1DFIT_%s;			// Prefix for output files
moldatfile	= %s;			// Molecular data file (LAMDA format)
girdatfile	= %s;	// Effective solar pumping rates (at 1 AU)

Qwater		= %8.3e;  	//  H2O production rate from the nucleus (mol/s)
vexp		   = %f;		//  Coma expansion velocity (m/s)
deldot      = %f;    // Velocity shift (m/s)
tkin		   = %f;		   //  Kinetic temperature (K)
abund		   = 0;	   //  Parent molecular abundance
dAbund      = %12.6e;    //  Daughter molecular abundance
lp          = %f;     // Production scale length 
dopplerb	   = %f;		// Turbulent Doppler broadening parameter (m/s)
betamol		= %8.3e;   //  Molecular photolysis rate at 1 AU (s^-1)
xne 		= %f;		// Electron density scaling factor - default value is 0.2
rhelio	= %f;         //  Heliocentric distance (AU)
rnuc     = %f;         //  Nucleus radius (m)
radius   = 1e9;         //  Model coma outer radius (m) 
collPartId	= %d;		   //  Collision partner ID (in moldatfile)

unit		= %d;		   //  Output intensity units - 0=Kelvin, 1=Jansky/pixel, 2=SI, 3=Lsun/pixel
velres	= %f;	   //  Channel resolution (m/s)
nchan		= %d;		   //  Number of channels
pxls		= %d;		//  Pixels per dimension
imgres	= %f;      //  Pixel size (arcsec)
delta		= %f;   	//  Distance from observer (AU)
%s
   """

   f=open(modelFile,'w')
   f.write(limeInput %(spec,lamFile,pumpFile,Q,vexp,vsource,tkin,abund,lp,dopplerb,betamol,xne,rH,rNuc,collPartId,units,chwid,nchan,npix,imgres,delta,fString))
   f.close()


#######################################################################################
# Multiply fits cube by Gaussian beam and extract the summed (beam-weighted) spectrum #
#######################################################################################

def convImgSpec(infits,hpbwx,hpbwy,theta,eta,units,xoff=0,yoff=0,freqAx=False,drop=True,outfile=None,integrate=False):
   # 1D Spectrum is extracted from a 3D image cube after applying a Gaussian beam shape (hpbwx,hpbwy are the major and minor axes in arcsec, theta is the position angle in degrees (CCW from the x axis), and xoff,yoff are the beam offset positions in arcsec). If drop=True, then only the first stokes axis of the cube is used. This (multiplicative) algorithm is much faster than a full convolution, but only applies for images in flux/pixel units. If the image units were Kelvin (surface brightness), we normalize the Gaussian so that the beam sum becomes an average.
   from scipy.constants import c
   from astropy.io import fits
   import sys   
      
      
   f = fits.open(infits)
   header = f[0].header
        
   naxis1 = header['NAXIS1']
   naxis2 = header['NAXIS2']

   cdelt1 = header['CDELT1']

   if drop:
      data = f[0].data[0]
   else:
      data = f[0].data

   if units == 0:
      norm = True  # Kelvin scale
   else: 
      norm = False # Jansky scale

   g=gauss2d(naxis1,naxis2,hpbwx,hpbwy,abs(cdelt1*3600),theta=np.deg2rad(theta+90),x0=xoff,y0=yoff,norm=norm)
   
   dg = data * g
   spectrum = np.sum(dg,axis=(1,2)) * eta
     
   # Generate frequency grid
   nspec = len(spectrum)
   f0 = header['CRVAL3']
   df = header['CDELT3']/1e3
   i0 = header['CRPIX3']
   xgrid = ((np.arange(nspec) - i0 + 1) * df + f0)
 
   if freqAx:
     xgrid = (1. - (xgrid/(c/1e3))) * (header['RESTFRQ']/1e9)

   if outfile:
      write2col(xgrid,spectrum,outfile)

   if integrate:
   #Get integrated intensity
      Tdv = sum(spectrum) * df
      sys.stdout.write ( "Integrated intensity = %.3f\n" % Tdv)


   return xgrid, spectrum


###################################################################
#  MAIN PROGRAM                                                   #
###################################################################

# Load input parameters
exec(compile(open(sys.argv[1], "rb").read(), sys.argv[1], 'exec'))

fitsFile = "SUBLIMED1DFIT_"+spec+".fits"

if 'hpbwx' not in globals():
   print("Only one beam dimension given --> assuming hpbwx=hpbwy=hpbw")
   hpbwx = hpbw
   hpbwy = hpbw
   pa = 0.
   
if 'rez' not in globals():
   rez = 1

if 'rmsFactor' not in globals():
   rmsFactor = 1.

if 'lp' not in globals():
   lp = 0.0

xobs,yobs = np.loadtxt(obsSpec,unpack=1)

#Test if spectral axis is velocity or frequency and set up model parameters accordingly. If frequency axis, then the model will be centered on the central channel and no transition index is needed 
if xobs.min() > 1.0:
   freqAx = True
   trans = 0.5*(xobs[-1]+xobs[0])
#Calculate spectral resolution of data in units of model channels
   rezchans = rez * c * abs(xobs[1]/xobs[0]-1)/chwid
   nchan = c * abs(xobs[-1]-xobs[0]) / trans / chwid
   dnuFactor = (c/1e3)/trans
   print("INFO: The observed spectrum is on a frequency axis (centered on %f). The model will cover the entire range, using %d channels" %(trans,nchan)) 
else:
   freqAx = False
   rezchans = rez * abs(xobs[1]-xobs[0])/(chwid/1e3)
   dnuFactor = 1.


# Model sampling warnings
if rez != 0:
   if rezchans < 2:
      print("WARNING: The model channel width (%.0f m/s) is more than half the spectral resolution (%.0f m/s) --> the spectral convolution kernel will be poorly sampled" %(chwid,rezchans*chwid))
      if rezchans < 0.25:
         print("WARNING: In fact, the model channel width is %.1f times the spectral resolution --> the model spectrum will contain significant cubic interpolation artifacts" %(1/rezchans))
   if nchan/rezchans < 4:
       sys.exit("FATAL: The model domain is too narrow for the requested spectral convolution kernel --> increase the number of channels")  
      
if imgres > min([hpbwx,hpbwy])/2:
   print("WARNING: The model pixels are more than half the beam size --> the model image convolution kernel will be poorly sampled")

if imgres * npix < max([hpbwx,hpbwy]) * 3:
   print("WARNING: The model image size is only %.1f times the beam fwhm --> edge effects may be significant in the convolved model image" % (imgres * npix / max([hpbwx,hpbwy])))



rms = rmsFactor * np.std(sigClip(yobs,3))

parinfo=[]

#  Chi-squared tolerance between successive iterations
ftol = 1.0e-5
#  Parameter value tolerance between successive iterations
xtol = 1.0e-4

for i in range(len(p0)):
   parinfo.append({'value':p0[i], 'fixed':pfix[i], 'limited':[0,0], 'limits':[0.,0.], 'relstep':0.1})


# abundance limits
parinfo[0]['limited'] = [1,1]
parinfo[0]['limits'] = [1.0e-6,1.0]

# Temperature limits
parinfo[1]['limited'] = [1,1]
parinfo[1]['limits'] = [10.0,350.0]

# vexp limits
parinfo[2]['limited'] = [1,1]
parinfo[2]['limits'] = [0.1,2.0]

# vsource limits
parinfo[3]['limited'] = [1,1]
parinfo[3]['limits'] = [-100,100]
del(parinfo[3]['relstep'])
parinfo[3]['step'] = vexp/6.

# doppler limits
parinfo[4]['limited'] = [1,1]
parinfo[4]['limits'] = [0.00,0.5]



#  If some parameters are free, do the fitting
if sum(pfix) != len(pfix):
   print("Starting MPFIT...")
   result = mpfit(getChisq, p0, parinfo=parinfo, xtol=xtol,ftol=ftol)
   
   #  Exit and print any error messages
   print('MPFIT exit status = ', result.status)
   if (result.status <= 0): 
      print('MPFIT', result.errmsg)
      sys.exit(result.status)

   #  Get final parameters
   abund,tkin,vexp,vsource,doppler = result.params   
   sigma_abund,sigma_tkin,sigma_vexp,sigma_vsource,sigma_doppler = result.perror
   
   # Reduced chi square
   DOF = len(yobs)-(len(p0)-sum(pfix))
   print('\nReduced Chisq X_r = %.4f'% (float(result.fnorm) / DOF))
   print('P (probability that model is different from data due to chance) = %.3f' % chi2.sf(result.fnorm,DOF))

   print('\nBest-fit parameters and 1-sigma covariance errors:')
   print('Abund. = %8.3e +- %8.3e' %(abund, sigma_abund))
   print('T_kin = %.2f +- %.2f K' %(tkin, sigma_tkin))
   print('v_exp = %.3f +- %.3f km/s' %(vexp,sigma_vexp))
   print('deldot = %.3f +- %.3f km/s' %(vsource,sigma_vsource))
   print('doppler = %.3f +- %.3f km/s' %(doppler, sigma_doppler))
 

#  If all fixed, calculate profile anyway
else:
   print("All parameters fixed...")
   status,resids = getChisq(p0)
   abund,tkin,vexp,vsource,doppler = p0

   print('Abund. = %8.3e' %(abund))
   print('T_kin = %.2f K' %(tkin))
   print('v_exp = %.3f km/s' %(vexp))
   print('deldot = %.3f km/s' %(vsource))
   print('doppler = %.3f km/s' %(doppler))
   
#Integrate the observed spectrum over the model range
yobsm=np.ma.array(yobs,mask=((xobs<np.min(mvel)) | (xobs>np.max(mvel))))
vobsm=np.ma.array(xobs,mask=((xobs<np.min(mvel)) | (xobs>np.max(mvel))))
Tdv = np.trapz(yobsm.compressed(),vobsm.compressed())
if xobs[0]>xobs[-1]: Tdv = -Tdv
sigma_Tdv = rms * (np.max(mvel) - np.min(mvel)) / np.sqrt(len(yobsm.compressed()))

print('\nTdv (obs) = %.3f +- %.3f K km/s\n' %(Tdv * dnuFactor, sigma_Tdv * dnuFactor))

if(showplots):
#  Plot the final model
   print("Close the plot window(s) to exit...")
   plt.plot(xobs,yobs, drawstyle = 'steps-mid', label='Observation')                 
   plt.plot(mvel,mT, label='Raw model')
   plt.plot(xobs,modelTI(xobs), drawstyle = 'steps-mid', label='Convolved model')                                                                                              
   #write2col(xobs,modelTI(xobs),"ACA_HC15N_2_SUBLIMED1D_conv_spec_conv.txt")
   plt.title("Final model")
   plt.legend(loc='upper right')                           
   plt.show()                                                                 
