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

priors = {}
random_state = 3457 # hard coded
cores = 24 # umm hard-coded? 
chains = 24 # int toggle-able
ntune = 2000 # int toggle-able
nsamples = 500 # int toggle-able

labels_to_fit = other_to_fit + elements_to_fit
assert set(labels_to_fit) <= set(NN_model.labels),\
    f'{set(labels_to_fit)- set(NN_model.labels)} not label(s) in the model'
assert set(priors.keys()) <= set(NN_model.labels), \
    f'Priors {set(labels_to_fit) - set(NN_model.labels)} not label(s) in the model'
assert set(priors.keys()) <= set(labels_to_fit), \
    f'Priors {set(priors.keys()) - set(labels_to_fit)} not being fit'

hmc_dir = drive_dir.joinpath('galaxy_widget/hmc_scratch')

# saving results of hmc -- don't think i need this chunk yet 
hmc_trace = hmc_dir.joinpath(f'{model_name}_snr{snr:03d}_trace.h5') # don't know if i need this chunk
hmc_samples = hmc_dir.joinpath(f'{model_name}_snr{snr:03d}_samples.h5')
hmc_corner = hmc_dir.joinpath(f'{model_name}_snr{snr:03d}_corner.png')
hmc_truespec = hmc_dir.joinpath(f'{model_name}_snr{snr:03d}_spec.npy') # save jic? 

theta_true = np.zeros(NN_model.dim_in)
spec_true = NN_model.nn_tt(theta_true).eval()
noisy_spec = spec_true + np.random.normal(0, scale=spec_true/snr, size=spec_true.shape[0])                                                                                          
np.save(hmc_truespec, spec_true)

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
    model_spec = NN_model.nn_tt(theta)
    # Likelihood                                                                                                                                                                     
    spec = pm.Normal('spec', mu=model_spec, sd=spec_true/snr, observed=noisy_spec)
    # Sampling                                                                                                                                                                       
    backend = HDF5(hmc_trace)
    trace = pm.sample(nsamples, tune=ntune, chains=chains, cores=cores, trace=backend)


samples = pd.DataFrame(columns=labels_to_fit)
samples[labels_to_fit] = trace_to_dataframe(trace, varnames=labels_to_fit)
samples = rescale_labels(samples, x_min_fit, x_max_fit)
samples.to_hdf(hmc_samples, f'SNR={snr}')

# plotting results 

ndim = len(labels_to_fit)
fig = corner(samples, labels=labels_to_fit, truths=rescale_labels(truth_fit, x_min_fit, x_max_fit),
             show_titles=True, quantiles=(0.16, 0.50, 0.84),
             range=list(zip(x_min_fit, x_max_fit)), scale_hist=True,
             max_n_ticks=3,
             label_kwargs=dict(size=24),
             hist_kwargs=dict(density=True))
axes = np.array(fig.axes).reshape((ndim, ndim))
for i, label in enumerate(labels_to_fit):  # Overplot priors                                                                                                                         
    ax = axes[i, i]
    if label in priors:
        a = (x_min_fit[i] - rescale_labels(truth_fit, x_min_fit, x_max_fit)[i]) / priors[label]
        b = (x_max_fit[i] - rescale_labels(truth_fit, x_min_fit, x_max_fit)[i]) / priors[label]
        x = np.linspace(x_min_fit[i], x_max_fit[i], 1000)
        prior_dist = truncnorm.pdf(x, a, b,
                                   loc=rescale_labels(truth_fit, x_min_fit, x_max_fit)[i],
                                   scale=priors[label])
        ax.plot(x, prior_dist)
plt.savefig(hmc_corner)
