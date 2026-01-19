# postEoR
postEoR is a semi-analytic simulation that builds off of [21cmFAST][https://github.com/21cmfast/21cmFAST] to model the neutral hydrogen field from $3 \lesssim z \lesssim 7$. This enables simulation of the transition from the end of reionization to late-time large-scale structure formation. 

# Features
* Simple generation of lightcones and coeval boxes into a Ltcone or Box object, with all associated data saved in this object
* Built-in analysis functions for the calculation and plotting of the power spectrum (spherical and cylindrical), halo mass function
* Generation of surveys using SKA-Low stages or custom telescopes, built-in calculation of thermal noise power
* Calculation of foreground avoidance power spectra for given regime and survey

# Install requirements
* py21cmfast commit b3f8e619d0d3fd53dff2b0ef1fe4d37d1adfc7eb
	* In inputs.py, min_redshift in _node_redshifts_default needs to be changed from 5.5 to 2.99 (to enable evolution of matter field past the end of the EoR)
* scipy
* numpy
* skimage
* matplotlib
* os
* abc
* hmf
* camb
* ska_ost_array_config
* astropy
* hickle

# Tutorial
See Notebooks.
