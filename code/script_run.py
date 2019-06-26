# -*- coding: utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('Agg')
import importlib
import gpc_DataGen
import gpc_utils
import gpc_module_gp_combo
# import module_HMC_orig

# useful to have reloads when developing
importlib.reload(gpc_DataGen)
importlib.reload(gpc_utils)
importlib.reload(gpc_module_gp_combo)
# importlib.reload(module_HMC_orig)

from gpc_DataGen import DataGenerator
from gpc_utils import *
import gpc_module_gp_combo
# import module_HMC_orig as module_HMC

import numpy as np
import tensorflow as tf
import datetime
import pickle

start_time = datetime.datetime.now()
print_w_time('started')

# avoid the dreaded type 3 fonts...
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 10})
plt.rcParams['text.usetex'] = True

# this script also requires:
# gpc_DataGen.py
# gpc_utils.py
# gpc_module_gp_combo.py

# this script produces the GP Figures from paper, 
# Expressive Priors in Bayesian Neural Networks: Kernel Combinations and Periodic Functions
# Pearce et al.
# https://arxiv.org/abs/1905.06076

# WARNING: this is horrible spaghetti code, apologies in advance...

# for 1-D prior plots, Figure 1,
# set data_set = 'favourite_fig', 
# and edit code in method combo_inner, 
# of gpc_module_gp_combo.py (around line 500)

# for 2-D prior plots, Figure 2
# set data_set = 'test_2D'
# and edit code in method combo_inner of gpc_module_gp_combo,
# uncommenting lines 524, 555

# for timeseries GP plots, Figure 3
# set data_set = 'mauna' or 'airline'
# and edit code in method combo_inner of gpc_module_gp_combo,
# use: self.k_a_type = 'relu' and self.k_b_type = 'per'
# and: self.per_p = 1.0/10
# I haven't synced all the hyperparams with those in timeseries_hmc_05.py
# so you'll have to tweak them to get an exact match

# generally speaking...
# using: activation_fn = 'combo' is the one you'll normally want to play with
# can also try relu, ERF, rbf for basic BNNs


# -- inputs --
data_set = 'favourite_fig' 	# favourite_fig, test_2D, mauna, airline
n_samples = 12
activation_fn = 'combo' 		# activation type - relu, erf, rbf, softplus, Lrelu
									# for GP only: cosine, SE, periodic, linear, add_lin_per, combo, relu_per 
									# combo
data_noise = 0.005 #0.001 			# data noise variance
b_0_var = 1				# var of b_0
w_0_var = b_0_var			# var of w_0
# w_0_var = 4			
u_var = 2					# var for rbf params as -> inf, goes to stationary cov dist
g_var = .5					# var for rbf params
n_runs = 1					# no. runs to average over

# -- NN model inputs --
optimiser_in = 'adam' 		# optimiser: adam, SGD
learning_rate = 0.01		# learning rate 0.01
decay_rate = 0.995			# decay rate
hidden_size = 50			# no. hidden neurons
n_epochs = 2000				# no. epochs to train for
cycle_print = n_epochs/10 	# print info every cycle no. of epochs
batch_size = 64
n_ensembles = 5				# no. NNs in ensemble

# -- HMC model inputs --
step_size = 0.001			# size of 'leapfrog steps' for HMC inference
n_steps = 150				# no. 'leapfrog steps' in between samples for HMC inference
burn_in = 500				# no. samples to drop
n_predict_hmc = 50				# no. samples used to make predictions
# n_samples_hmc = burn_in+n_predict	# total no. of samples to collect - should be > burn_in
n_samples_hmc = 1000

# -- VI model inputs --
n_predict_vi = 50			# no. samples for predictions
n_iter_vi = 2000			# how long to run for
n_samples_vi = 300			# no. samples from variational model for calculating stochastic gradients

# -- misc model inputs --
single_data_n_std = 0.05 	# when training a single NN, const. data noise std dev!
single_lambda_mod = 1		# multiply by lambda

# plotting options
is_try_plot = True
is_save_graphs = 0


if activation_fn == 'erf':
	type='panel/favfig_erf_low_noise_'
# elif activation_fn == 'relu':
# 	type='panel/favfig_relu_low_noise_'
elif activation_fn == 'Lrelu':
	type='panel/favfig_Lrelu_low_noise_'
elif activation_fn == 'rbf':
	type='panel/favfig_rbf_low_noise_'
elif data_set == 'mauna':
	type='mauna'
elif data_set == 'airline':
	type='airline'
else:
	type='other'

# which to run
is_gp_run = 1
is_hmc_run = 0

is_single_run = 0
is_deep_NN = 0 # whether to make a 2 layer NN

gp_results=[]; ens_results=[]; single_results=[]; mc_results=[]; 
hmc_results=[]; sk_results=[]; unc_ens_results=[]; 
run_kls=[]
for run_ in range(n_runs):
	print('\n\n  ====== run:',run_, '======\n')

	# -- create data --
	Gen = DataGenerator(type_in=data_set)
	X_train, y_train, X_val, y_val = Gen.CreateData(n_samples=n_samples, seed_in=run_+10114, # was 1011
		train_prop=0.9)

	# do cos transformation
	# X_train = np.concatenate([np.cos(X_train),np.sin(X_train)],axis=-1)
	# X_val = np.concatenate([np.cos(X_val),np.sin(X_val)],axis=-1)

	# X_train = X_train[0:2]
	# y_train = y_train[0:2]
	n_samples = X_train.shape[0]
	X_dim = X_train.shape[1]
	y_dim = 1

	# this lets us test how gp changes when data duplicated
	# the effect is the same as over optimising a NN
	# which suggests there is a need to do early stopping
	# X_train_orig = X_train.copy()
	# y_train_orig = y_train.copy()
	# for i in range(50):
	# 	X_train = np.concatenate((X_train,X_train_orig))
	# 	y_train = np.concatenate((y_train,y_train_orig))

	# mesh the input space for evaluations
	if X_dim == 1:
		if type=='panel/favfig_rbf_low_noise_':
			X_grid = np.atleast_2d(np.linspace(-8, 8, 200)).T
		else:
			# X_grid = np.atleast_2d(np.linspace(-3, 3, 100)).T

			X_grid = np.atleast_2d(np.linspace(-6*0.9, 6*0.9, 200)).T # prior cartoon
			if data_set == 'mauna':
				X_grid = np.atleast_2d(np.linspace(0, 20/10, 200)).T # mauna, 400 for plots
			if data_set == 'airline':
				X_grid = np.atleast_2d(np.linspace(0, 20/10, 300)).T # airline

		X_val = X_grid
		y_val = np.expand_dims(X_grid[:,0],1)
	elif X_dim == 2:
		x_s = np.atleast_2d(np.linspace(-2, 2, 30)).T # 30
		X_grid = np.array(np.meshgrid(x_s,x_s))
		X_grid = np.stack((X_grid[1].ravel(), X_grid[0].ravel()),axis=-1)
		X_val = X_grid
		y_val = np.expand_dims(X_grid[:,0],1)
	else:
		X_grid = X_val


	if is_gp_run:
		# -- gp model --
		gp = gpc_module_gp_combo.gp_model(kernel_type=activation_fn, data_noise=data_noise, 
			b_0_var=b_0_var, w_0_var=w_0_var, u_var=u_var, g_var=g_var)
		y_pred_mu, y_pred_std = gp.run_inference(x_train=X_train, y_train=y_train, x_predict=X_val, print=False)
		if is_try_plot and X_dim == 1: gp.priors_visualise(n_draws=2, is_save=is_save_graphs)
		if is_try_plot and X_dim == 2: gp.priors_visualise_2d(n_draws=2, is_save=is_save_graphs)

		metrics_calc(y_val, y_pred_mu, y_pred_std, Gen.scale_c, b_0_var, w_0_var, data_noise, gp, is_print=True)
		gp_results.append(np.array((gp.mse_unnorm, gp.rmse, gp.nll)))
		if data_set == 'mauna' or data_set == 'airline':
			if is_try_plot: try_plot(X_dim, X_grid, y_pred_mu, y_pred_std, X_train, y_train, gp, save=is_save_graphs, type=type)


	if is_hmc_run:
		# -- hmc model --
		hmc = module_HMC.hmc_model(activation_fn=activation_fn, data_noise=data_noise, 
			b_0_var=b_0_var, w_0_var=w_0_var, u_var=u_var, g_var=g_var, hidden_size = hidden_size,
			step_size=step_size, n_steps=n_steps, burn_in=burn_in, n_samples=n_samples_hmc, n_predict=n_predict_hmc, deep_NN = is_deep_NN)

		hmc.train(X_train=X_train, y_train=y_train, X_val=X_val,is_print=False)

		y_preds, y_pred_mu, y_pred_std = hmc.predict(X_val)

		metrics_calc(y_val, y_pred_mu, y_pred_std, Gen.scale_c, b_0_var, w_0_var, data_noise, hmc, is_print=True)
		hmc_results.append(np.array((hmc.mse_unnorm, hmc.rmse, hmc.nll)))
		if is_try_plot: try_plot(X_dim, X_grid, y_pred_mu, y_pred_std, X_train, y_train, hmc, save=is_save_graphs, type=type)#, y_preds)

if is_gp_run:
	gp_results = np.array(gp_results)
	print('\n\n___ GP RESULTS ___')
	print('data', data_set, ', act_fn', activation_fn, ', b_0_var', b_0_var, 'd_noise',data_noise)
	metric_names= ['MSE_un','RMSE', 'NLL']
	print('runs\tensemb')
	print(n_runs, '\t', n_ensembles)
	print('\tavg\tstd_err\tstd_dev')
	for i in range(0,len(metric_names)): 
		avg = np.mean(gp_results[:,i])
		std_dev = np.std(gp_results[:,i], ddof=1)
		std_err = std_dev/np.sqrt(n_runs)
		print(metric_names[i], '\t', round(avg,3), 
			'\t', round(std_err,3),
			'\t', round(std_dev,3))


# -- tidy up --
print_w_time('finished')
end_time = datetime.datetime.now()
total_time = end_time - start_time
print('seconds taken:', round(total_time.total_seconds(),1),
	'\nstart_time:', start_time.strftime('%H:%M:%S'), 
	'end_time:', end_time.strftime('%H:%M:%S'))


# ------------------------------------------------------------------------------------------------
# settings

# -- mauna --
# data_set = 'mauna' # 'drunk_bow_tie' '~boston' favourite_fig, toy_2, bias_fig, 
# activation_fn = 'combo' 

# data_noise = 0.005

# self.add_or_mult = '+' # *, +
# self.k_a_type = 'relu' # relu, erf, per, se
# self.k_b_type = 'per'

# # components of each
# if self.add_or_mult == '+':
# 	self.comp_a = 1.0
# 	self.comp_b = 2. - self.comp_a
# else:
# 	self.comp_a = 1.0
# 	self.comp_b = 1.0

# # -- first kernel --

# # relu, erf
# self.b_0_var = 500.0
# self.w_0_var = self.w_0_var
# self.w_1_var = 0.05

# # periodic
# self.per_p = 4.0
# self.per_const = 1.0
# self.per_l = 0.5

# # squared exp
# self.SE_const = 1.0
# self.SE_l = 10

# # -- second kernel --

# # relu, erf
# self.b_0_var = 1000.0
# self.w_0_var = self.w_0_var
# self.w_1_var = 5.0

# # periodic
# self.per_p = 1.0 # smaller is faster 
# self.per_const = 1.0
# self.per_l = 4.0 # how noisy fn is, larger = smoother

# # squared exp
# self.SE_const = 1.0
# self.SE_l = 10

# -- airline --

# data_noise = 0.01 #0.001 
# self.add_or_mult = '*' # *, +
# otherwise same as for mauna

