# SUBLIMED1DFIT

This is a python code that interfaces with SUBLIME-D1DC to fit a NLTE radiative transfer coma model to an observed spectrum, using the Levenberg-Marquardt lest-squares algorithm.  Parameter uncertainties are derived from the covariance matrix, based on the RMS noise of the input spectrum.

Example input file (sublimed1dFit.par) is in the example/ folder, and contains initial guesses for the molecular abundance, kinetic temperature, outflow velocity, Doppler shift and (turbulent) line broadening parameter. Molecular data must be supplied in the form of a LAMDA input file and (optional) effective pumping rates. Observed data should be on a velocity scale (km/s). The number of spectral channels in the model and the flux units (Jy or K) should be set appropriately. Parameter values can be held fixed using the pfix array.

This python code should be run as an executable from the command line, with the input file as the argument. Requires a working installation of SUBLIME-D1DC (accessible on the command line). Python dependencies include numpy, scipy, matplotlib, astropy and mpfit.