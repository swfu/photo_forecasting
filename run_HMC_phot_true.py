from pathlib import Path
import sys

drive_dir = Path('drive/My Drive')
payne_dir = drive_dir.joinpath('DPayne')
sys.path.append(str(payne_dir))

import argparse
from pathlib import Path
import yaml
import pandas as pd
from DPayne.utils import split_data
from DPayne.neural_networks import load_trained_model
from DPayne.utils import scale_labels, rescale_labels

import theano.tensor as tt
import pymc3 as pm
from pymc3.backends import HDF5
from pymc3.backends.tracetab import trace_to_dataframe
from scipy.stats import truncnorm
import numpy as np

model_name = 'R6500p'

nn_dir = drive_dir.joinpath('DPayne/DPayne')
NN_model = load_trained_model(model_name, nn_dir, theano_wrap=True)

elements_to_fit = ['Fe', 'Ca', 'Ni', 'Si', 'Ti', 'Co', 'Mg'] # this will be toggle-able (less elementstho)
other_to_fit = ['Teff', 'logg', 'v_micro'] # this will be hard-coded

labels_to_fit = other_to_fit + elements_to_fit
assert set(labels_to_fit) <= set(NN_model.labels),\
    f'{set(labels_to_fit)- set(NN_model.labels)} not label(s) in the model'
assert set(priors.keys()) <= set(NN_model.labels), \
    f'Priors {set(labels_to_fit) - set(NN_model.labels)} not label(s) in the model'
assert set(priors.keys()) <= set(labels_to_fit), \
    f'Priors {set(priors.keys()) - set(labels_to_fit)} not being fit'

filtersfile = drive_dir.joinpath('galaxy_widget/filters.h5')
fullthroughput = pd.read_hdf(filtersfile,'throughput')
fullwav = pd.read_hdf(filtersfile,'wave_eff')

def calc_photometry(filter_name, model_wavelength, model_flux,
	throughput_file=fullthroughput,wave_eff_file=fullwav):
	'''
	inputs
	filter: filter name. this will be fed in as the column name of the fullthroughput and fullwav data frames 
	model_wavelength: from the neural net. hopefully lower resolution than transmission curves 
	model_flux: uhh just comes from the neural net (does not correct for radius or distance or things like that yet.)
	throughput_file: table corresponding to transmission curves of a filter 
	wave_eff_file: the table corresponding to effective wavelength of a filter
	--
	outputs
	-flux value in a specific photometric band
	-effective wavelength of that flux (in angstroms)
	'''

	transmission_interp = np.interp(model_wavelength, fullthroughput.index, fullthroughput[filter_name])
	numerator = np.trapz(np.multiply(np.multiply(model_flux, transmission_interp), model_wavelength),model_wavelength)
	denominator = np.trapz(np.multiply(model_wavelength,transmission_interp),model_wavelength)
	flux = numerator/denominator 
	return flux, wave_eff_file[filtername] # these are scalars ... i hope. 

def calc_multiple_filters(filter_list, model_wavelength, model_flux):
	'''
	given a set of filters, calculates photometry of the model_wavelength and model_flux spectra 
	the last two arguments, as usual, usually come from a neural net. 
	need to have a throughput and wave_eff array already read in.  
	'''
	wave_eff = []
	flux_array = []
	flux_err_array = []
	for i in range(len(filter_list)):
		photo_output = calc_photometry(filter_list[i],model_wavelength, model_flux)
		flux_array.append(photo_output[0])
		flux_err_array.append(photo_output[1])
		wave_eff.append(photo_output[2])
	return np.array(wave_eff), np.array(flux) # these are arrays

hmc_dir = drive_dir.joinpath('galaxy_widget/hmc_scratch')

# saving results of hmc -- don't think i need this chunk yet 
# hmc_trace = hmc_dir.joinpath(f'{model_name}_snr{snr:03d}_trace.h5') # don't know if i need this chunk
# hmc_samples = hmc_dir.joinpath(f'{model_name}_snr{snr:03d}_samples.h5')
# hmc_corner = hmc_dir.joinpath(f'{model_name}_snr{snr:03d}_corner.png')
# hmc_truespec = hmc_dir.joinpath(f'{model_name}_snr{snr:03d}_spec.npy') # save jic? 

theta_true = np.zeros(NN_model.dim_in)
spec_true = NN_model.nn_tt(theta_true).eval() # so THIS is a spectrum, comes in a numpy array 

wave = np.linspace(6500,9000,spec_true.size,endpoint=True) 
# i'm defining a fake spectrum here just because. this is in angstroms

phot_wave_eff, phot_true = calc_multiple_filters(filt_list, SN_list, wave, spec_true)

phot_noise = phot_true + phot_err 
# phot_noise = phot_true + np.random.normal(0,scale=phot_true/snr,size=phot_true.shape[0])

# noisy_spec = spec_true + np.random.normal(0, scale=spec_true/snr, size=spec_true.shape[0])                                                                                          
# np.save(hmc_truespec, spec_true)

'''                                                                                                                                                                                  
Scale Priors; hard-code this                                                                                                                                                                        
'''

tmp = [NN_model.labels.index(item) for item in labels_to_fit]
x_min_fit = NN_model.x_min[tmp]
x_max_fit = NN_model.x_max[tmp]
truth_fit = theta_true[tmp]

scaled_priors = {}
if priors:
    for label in priors:
        lab_ind = np.argwhere(np.array(labels_to_fit) == label)[0][0]
        lab = rescale_labels(theta_true, NN_model.x_min, NN_model.x_max)[lab_ind] + priors[label]
        scaled_priors[label] = scale_labels(lab, NN_model.x_min, NN_model.x_max)[lab_ind]

# this is the TRAINING. ie i need to change this up. 

with pm.Model() as model:
    # Priors                                                                                                                                                                         
    theta_list = []
    for label in NN_model.labels:
        if label in labels_to_fit:
            if label in priors:
                theta_list.append(pm.Bound(pm.Normal, lower=-2.0, upper=2.0)(label, mu=0.0, sigma=scaled_priors[label]))
            else:
                theta_list.append(pm.Uniform(label, lower=-2.0, upper=2.0))
        else:
            theta_list.append(0.0)
    theta = tt.stack(theta_list)
    # Model                                                                                                                                                                          
    model_spec = NN_model.nn_tt(theta) # this has the SAME WAVELENGTHS as the fake wavelength from above
    model_phot, model_wave_eff = calc_multiple_filters(filt_list, wave, model_spec) 
    # this should be saved as a np array... from the photometry 
    
    # Likelihood                                                                                                                                                                     
    phot = pm.Normal('phot', mu=model_phot, sd=phot_true/snr, observed=phot_noise)
    # Sampling                                                                                                                                                                       
    backend = HDF5(hmc_trace)
    trace = pm.sample(nsamples, tune=ntune, chains=chains, cores=cores, trace=backend)


samples = pd.DataFrame(columns=labels_to_fit)
samples[labels_to_fit] = trace_to_dataframe(trace, varnames=labels_to_fit)
samples = rescale_labels(samples, x_min_fit, x_max_fit)
samples.to_hdf(hmc_samples, f'SNR={snr}')



