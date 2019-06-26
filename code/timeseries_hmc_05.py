import theano
floatX = theano.config.floatX
import pymc3 as pm
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.special import erf
from sklearn.datasets import fetch_openml

# this script produces the BNN parts of Figure 3 from paper, 
# Expressive Priors in Bayesian Neural Networks: Kernel Combinations and Periodic Functions
# Pearce et al.
# https://arxiv.org/abs/1905.06076
#
# change activation_fn and data_set to get the different permutations of experiments
#
# the script requires /01_data/international-airline-passengers.csv
#
# note that hyperparameters in this script have been optimised for speed rather than
# to reproduce, exactly, the distributions found in the paper, so some minor
# differences may exist.

# avoid the dreaded type 3 fonts...
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 10})
plt.rcParams['text.usetex'] = True

# inputs
activation_fn = 'combo_bnn' # relu, periodic, combo_bnn, erf
data_set = 'mauna' # mauna, airline

if data_set == 'mauna':
	data_noise = 0.005 # variance, mauna=0.005 mauna_relu=0.001 || airline=0.005 airline_relu = 0.001
	n_hidden = 50 # mauna=50, mauna_relu = 100 || airline=50
	n_inf_samples = 700 # number samples to store during inference
	n_pred_samples = 200 # number samples to use during prediction
	w1_var = 1
	b1_var = w1_var
	w1_var_2 = 5. # mauna = 5, mauna_relu = 50 || airline = 1, airline_relu = 20
	b1_var_2 = w1_var_2
	w2_var_a = 1. # final layer weight variance # mauna = 1, mauna_per = 20 || airline = 1, airline_per = 10
	w2_var_b = 1.
	b2_var = 1.

elif data_set == 'airline':
	data_noise = 0.001 # variance, mauna=0.005 mauna_relu=0.001 || airline=0.005 airline_relu = 0.001
	n_hidden = 50 # mauna=50, mauna_relu = 100 || airline=50
	n_inf_samples = 700 # number samples to store during inference
	n_pred_samples = 200 # number samples to use during prediction
	w1_var = 1.0
	b1_var = w1_var
	w1_var_2 = 0.1 # mauna = 5, mauna_relu = 50 || airline = 1, airline_relu = 20
	b1_var_2 = w1_var_2
	w2_var_a = 1. # final layer weight variance # mauna = 1, mauna_per = 20 || airline = 1, airline_per = 10
	w2_var_b = 1.
	b2_var = 1.

is_save = 0 # whether to save final plot
is_plot_data_only = 0 # whether to plot data set by itself
is_plot_priors = 0 # whether to plot a prior draw

# rbf
var_u = 1.0 # spread of relative points
var_g = 1.0 # how smooth fn is (larger=smoother), mauna = 1 || airline = 1
estimate_per = 1/10 # expected period of repetition
l = np.sqrt(2*var_g + var_g**2 / var_u)
var_e = 1/(2/var_g + 1/var_u)
var_m = 2*var_u + var_g
var_f = w2_var_a * (var_e/var_u) * np.exp(-1/var_m)

# create data
def load_mauna_loa_atmospheric_co2():
	# routine for downloading and processing mauna data set
	ml_data = fetch_openml(name='mauna-loa-atmospheric-co2')
	months = []
	ppmv_sums = []
	counts = []

	y = ml_data.data[:, 0]
	m = ml_data.data[:, 1]
	month_float = y + (m - 1) / 12
	ppmvs = ml_data.target

	for month, ppmv in zip(month_float, ppmvs):
		if not months or month != months[-1]:
			months.append(month)
			ppmv_sums.append(ppmv)
			counts.append(1)
		else:
			# aggregate monthly sum to produce average
			ppmv_sums[-1] += ppmv
			counts[-1] += 1

	months = np.asarray(months).reshape(-1, 1)
	avg_ppmvs = np.asarray(ppmv_sums) / counts
	return months, avg_ppmvs

if data_set == 'mauna':
	X_train, Y_train = load_mauna_loa_atmospheric_co2()
	X_train = (X_train - X_train[0])/10
	X_train = np.concatenate([X_train[0:36], X_train[60:120]]) # leave out a chunk in middle
	Y_train = np.concatenate([Y_train[0:36], Y_train[60:120]])
	Y_train = (Y_train - np.mean(Y_train))/np.std(Y_train)
	Y_train = np.atleast_2d(Y_train).T

if data_set == 'airline':
	# we use the airline dataset already downloaded
	Y_train = np.genfromtxt('01_data/international-airline-passengers.csv', usecols=1,skip_header=1,skip_footer=1,delimiter=',')
	Y_train = np.atleast_2d(Y_train).T
	Y_train = (Y_train - np.mean(Y_train))/np.std(Y_train) #+2

	n_points = Y_train.shape[0] 
	X_train = np.linspace(0,n_points/12,n_points)/10
	X_train = np.atleast_2d(X_train).T

	X_train = np.concatenate([X_train[0:36], X_train[60:120]]) # leave out a chunk in middle
	Y_train = np.concatenate([Y_train[0:36], Y_train[60:120]])

def pre_process_rbf_per(x):
	# do cos and sin transformations so will become periodic kernel
	# do this to inputs rather than in BNN routine as pymc is a pain to figure out
	return np.concatenate([np.cos(x*2*np.pi/estimate_per),np.sin(x*2*np.pi/estimate_per)],axis=-1)

# pre process inputs for rbf periodic BNN
X_train_per = pre_process_rbf_per(X_train)

if is_plot_data_only:
	# can plot dataset only here
	fig = plt.figure(figsize=(5, 4))
	ax = fig.add_subplot(111)
	ax.plot(X_train[:,0], Y_train, 'r.', markersize=14, label=u'Observations', markeredgecolor='k',markeredgewidth=0.5)
	fig.show()

# create validation data - i.e. a grid over 1-d input
X_grid = np.atleast_2d(np.linspace(-6*0.9, 6*0.9, 200)).T # prior cartoon
if data_set == 'mauna':
	X_grid = np.atleast_2d(np.linspace(0, 20/10, 400)).T # mauna
if data_set == 'airline':
	X_grid = np.atleast_2d(np.linspace(0, 20/10, 400)).T # airline
X_grid_per = pre_process_rbf_per(X_grid)
Y_grid = np.ones_like(X_grid)

# set up
ann_input = theano.shared(X_train)
ann_input_per = theano.shared(X_train_per)
ann_output = theano.shared(Y_train)
total_size = len(Y_train)
n_in = X_train.shape[1]
n_in_per = X_train_per.shape[1]

# Initialize random weights between each layer
comp_a = 1.; comp_b = 1.;

priors=[]
for i in range(1):
	init_w1 = np.random.normal(loc=0, scale=np.sqrt(w1_var), size=[n_in, n_hidden]).astype(floatX)
	init_b1 = np.random.normal(loc=0, scale=np.sqrt(b1_var), size=[n_hidden]).astype(floatX)
	# init_out = np.random.normal(loc=0, scale=np.sqrt(np.sqrt(comp_a)/n_hidden), size=[n_hidden,1]).astype(floatX)
	init_out = np.random.normal(loc=0, scale=np.sqrt(w2_var_a/n_hidden), size=[n_hidden,1]).astype(floatX)

	init_w1_2 = np.random.normal(loc=0, scale=np.sqrt(w1_var_2), size=[n_in, n_hidden]).astype(floatX)
	init_b1_2 = np.random.normal(loc=0, scale=np.sqrt(b1_var_2), size=[n_hidden]).astype(floatX)
	# init_out_2 = np.random.normal(loc=0, scale=np.sqrt(np.sqrt(comp_b)/n_hidden), size=[n_hidden,1]).astype(floatX)
	init_out_2 = np.random.normal(loc=0, scale=np.sqrt(w2_var_b/n_hidden), size=[n_hidden,1]).astype(floatX)

	# final bias
	init_b2 = np.random.normal(loc=0, scale=np.sqrt(b2_var), size=[1]).astype(floatX)

	# rbf, need init_rbf to be size (n_hidden, n_samples, n_dims)
	init_rbf_orig = np.random.normal(loc=0, scale=np.sqrt(var_u), size=[n_hidden,n_in_per]).astype(floatX)
	init_rbf_orig = np.expand_dims(init_rbf_orig,1)
	init_rbf = np.repeat(init_rbf_orig,total_size,axis=1)
	rbf_input = theano.shared(init_rbf)

	# prior plot
	if is_plot_priors:
		init_rbf_pred = np.repeat(init_rbf_orig,X_grid.shape[0],axis=1)
		np_act_diffs = X_grid_per - init_rbf_pred
		np_h_a = np.exp(-np.linalg.norm(np_act_diffs,axis=-1)/(2*var_g)).T
		np_f_a = np.matmul(np_h_a, init_out)

		np_h_b = np.maximum(np.matmul(X_grid, init_w1_2) + init_b1_2,0)
		np_f_b = np.matmul(np_h_b, init_out_2)

		# prior_out = np_f_b
		# prior_out = np.matmul(np_h_a * np_h_b, init_out)
		if data_set == 'mauna':
			prior_out = np_f_a + np_f_b
		elif data_set == 'airline':
			prior_out = np.matmul(np_h_a * np_h_b, init_out_2)
		priors.append(prior_out)

if is_plot_priors: # plot priors
	priors=np.squeeze(np.array(priors))
	fig = plt.figure(figsize=(5, 4))
	ax = fig.add_subplot(111)
	ax.plot(priors.T,'k',alpha=1.0)
	# ax.plot(priors.T,'k',alpha=0.1)
	fig.show()

def build_model(ann_output):
	with pm.Model() as model:

		# first head of NN
		weights_in_w1 = pm.Normal('w_in_1', 0, sd=np.sqrt(w1_var),
								 shape=(n_in, n_hidden),
								 testval=init_w1)

		weights_in_b1 = pm.Normal('b_in_1', 0, sd=np.sqrt(b1_var),
								 shape=(n_hidden),
								 testval=init_b1)

		weights_2_out = pm.Normal('w_2_out', 0, sd=np.sqrt(w2_var_a/n_hidden),
								  shape=(n_hidden,1),
								  testval=init_out)

		weights_in_b2 = pm.Normal('b_in_2', 0, sd=np.sqrt(b2_var),
								 shape=(1),
								 testval=init_b2)

		if activation_fn in ['mixed', 'cos_lin', 'combo_bnn']: 
			# second head of NN
			weights_in_w1_2 = pm.Normal('w_in_1_2', 0, sd=np.sqrt(w1_var_2),
									 shape=(n_in, n_hidden),
									 testval=init_w1_2)

			weights_in_b1_2 = pm.Normal('b_in_1_2', 0, sd=np.sqrt(b1_var_2),
									 shape=(n_hidden),
									 testval=init_b1_2)

			weights_2_out_2 = pm.Normal('w_2_out_2', 0, sd=np.sqrt(w2_var_b/n_hidden),
								  shape=(n_hidden,1),
								  testval=init_out_2)

			# could learn the weighting between components, although I don't use it here
			comp_a_learn = pm.Uniform('comp_a_learn',lower=0,upper=2,testval=1.)
			comp_b_learn = 2 - comp_a_learn
			# comp_b_learn = pm.Uniform('comp_b_learn',lower=0,upper=1)


		# Build neural-network using tanh activation function
		if activation_fn == 'relu':
			act_1 = pm.math.maximum(pm.math.dot(ann_input, weights_in_w1) + weights_in_b1,0)
			act_out = pm.math.dot(act_1, weights_2_out)
		if activation_fn == 'periodic':
			act_diffs = ann_input_per - rbf_input
			h_a = pm.math.exp(-act_diffs.norm(L=2,axis=-1)/(2*var_g)).T
			f_a = pm.math.dot(h_a, weights_2_out)
			act_out = f_a
		elif activation_fn =='tanh':
			act_1 = pm.math.tanh(pm.math.dot(ann_input, weights_in_w1) + weights_in_b1)
			act_out = pm.math.dot(act_1, weights_2_out)
		elif activation_fn =='erf':
			act_1 = pm.math.erf(pm.math.dot(ann_input, weights_in_w1) + weights_in_b1)
			act_out = pm.math.dot(act_1, weights_2_out)
		elif activation_fn =='cosine':
			act_1 = pm.math.cos(pm.math.dot(ann_input, weights_in_w1) + weights_in_b1)
			act_out = pm.math.dot(act_1, weights_2_out)
		elif activation_fn =='linear':
			act_1 = pm.math.dot(ann_input, weights_in_w1) + weights_in_b1
			act_out = pm.math.dot(act_1, weights_2_out)
		elif activation_fn =='mixed':
			# out_a = pm.math.dot(pm.math.erf(pm.math.dot(ann_input, weights_in_w1) + weights_in_b1),weights_2_out)
			# out_b = pm.math.dot(pm.math.erf(pm.math.dot(ann_input, weights_in_w1_2) + weights_in_b1_2),weights_2_out_2)
			# act_out = out_a * out_b

			out_a = pm.math.erf(pm.math.dot(ann_input, weights_in_w1) + weights_in_b1)
			# out_b = pm.math.erf(pm.math.dot(ann_input, weights_in_w1_2) + weights_in_b1_2)
			out_b = pm.math.maximum(pm.math.dot(ann_input, weights_in_w1_2) + weights_in_b1_2,0)
			# act_out = pm.math.dot(out_a + out_b, weights_2_out)
			# act_out = pm.math.dot(out_a * out_b, weights_2_out)
			act_out = pm.math.dot(np.sqrt(comp_a)*out_a * np.sqrt(comp_b)*out_b, weights_2_out)
			# act_out = 1.*pm.math.dot(out_a, weights_2_out) + 1.*pm.math.dot(out_b, weights_2_out_2)
			# act_out = 1.*pm.math.dot(out_a, weights_2_out) * 1.*pm.math.dot(out_b, weights_2_out_2)
			# act_out = pm.math.dot(out_a, weights_2_out)
			# act_out = pm.math.dot(out_b, weights_2_out_2)

		elif activation_fn =='cos_lin':
			out_a = pm.math.cos(pm.math.dot(ann_input, weights_in_w1) + weights_in_b1)
			# out_b = pm.math.dot(ann_input, weights_in_w1_2) + weights_in_b1_2
			out_b = pm.math.maximum(pm.math.dot(ann_input, weights_in_w1_2) + weights_in_b1_2,0)
			f_a = pm.math.dot(out_a, weights_2_out)
			f_b = pm.math.dot(out_b, weights_2_out_2)
			act_out = f_a #+ f_b
			# act_out = comp_a_learn*f_a + comp_b_learn*f_b

			# act_out = pm.math.dot(np.sqrt(comp_a_learn)*out_a + np.sqrt(comp_b_learn)*out_b, weights_2_out)
		elif activation_fn =='combo_bnn':

			# after mapping inputs to cos and sin
			act_diffs = ann_input_per - rbf_input
			h_a = pm.math.exp(-act_diffs.norm(L=2,axis=-1)/(2*var_g)).T
			f_a = pm.math.dot(h_a, weights_2_out)

			if False:
				h_a = pm.math.erf(pm.math.dot(ann_input, weights_in_w1) + weights_in_b1)
				# h_a = pm.math.maximum(pm.math.dot(ann_input, weights_in_w1) + weights_in_b1,0)
				f_a = pm.math.dot(h_a, weights_2_out)

			# take raw inputs
			# h_b = pm.math.erf(pm.math.dot(ann_input, weights_in_w1_2) + weights_in_b1_2)
			h_b = pm.math.maximum(pm.math.dot(ann_input, weights_in_w1_2) + weights_in_b1_2,0)
			f_b = pm.math.dot(h_b, weights_2_out_2)

			# could use individual BNN outputs
			# act_out = f_a
			# act_out = f_b
			if data_set == 'mauna':
				act_out = f_a + f_b
			elif data_set == 'airline':
				h_combo = h_a*h_b
				act_out = pm.math.dot(h_combo, weights_2_out_2)
		
		out = pm.Normal('out', act_out,sd=np.sqrt(data_noise),
						   observed=ann_output,
						   total_size=total_size)

	return model, out


# build BNN
BNN, out = build_model(ann_output)
 
# run inference with neural_network
print('\nStarting sampling. If it hangs early on (<20 samples) try restarting script or reducing path_length.\n')
step = pm.HamiltonianMC(path_length=0.4, adapt_step_size=True, step_scale=0.04,
	gamma=0.05, k=0.9, t0=1, target_accept=0.9, model=BNN)
trace = pm.sample(n_inf_samples, step=step, model=BNN, chains=1, n_jobs=1, tune=300)

# make predictions - need to update inputs in this way with validation data
ann_input.set_value(X_grid.astype('float32'))
ann_input_per.set_value(X_grid_per.astype('float32'))
init_rbf_pred = np.repeat(init_rbf_orig,X_grid.shape[0],axis=1)
rbf_input.set_value(init_rbf_pred.astype('float32'))
ann_output.set_value(X_grid.astype('float32'))

ppc = pm.sample_ppc(trace, model=BNN, samples=n_pred_samples) # this does new set of preds per point
y_preds = ppc['out']
y_pred_mu = y_preds.mean(axis=0)
y_pred_std = y_preds.std(axis=0)

# add on data noise
y_pred_std = np.sqrt(np.square(y_pred_std) + data_noise)

# plot predictions
x_s = X_grid; y_mean = y_pred_mu; y_std = y_pred_std; X_train_plt = X_train

# add years on x axis
if data_set == 'mauna':
	yr_start = 1949; x_s = x_s*12 + yr_start; X_train_plt = X_train*12 + yr_start
elif data_set == 'airline':
	yr_start = 1958; x_s = x_s*12 + yr_start; X_train_plt = X_train*12 + yr_start


fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(111)
ax.plot(x_s, y_mean + 3 * y_std, 'b', linewidth=0.5)
ax.plot(x_s, y_mean - 3 * y_std, 'b', linewidth=0.5)
ax.fill(np.concatenate([x_s, x_s[::-1]]),
		 np.concatenate([y_mean - 3 * y_std,
						(y_mean + 3 * y_std)[::-1]]),
		 alpha=1, fc='lightskyblue', ec='None')
ax.plot(x_s, y_mean, 'b-', linewidth=1.,label=u'Prediction')
if data_set == 'mauna':
	ax.plot(X_train_plt[:,0], Y_train, 'r.', markersize=8, label=u'Observations', markeredgecolor='k',markeredgewidth=0.5)
	ax.set_ylim(-2.2, 5.5)
elif data_set == 'airline':
	ax.plot(X_train_plt[:,0], Y_train, 'r.', markersize=8, label=u'Observations', markeredgecolor='k',markeredgewidth=0.5)
	ax.set_ylim(-2, 5)
else:
	ax.plot(X_train_plt[:,0], Y_train, 'r.', markersize=14, label=u'Observations', markeredgecolor='k',markeredgewidth=0.5)
ax.set_xlim(yr_start,x_s.max())
if is_save:
	ax.set_yticklabels([])
	# ax.set_xticklabels([])
	ax.set_yticks([])
	# ax.set_xticks([])
fig.show()

if is_save:
	title = datetime.datetime.now().strftime('%m-%d-%H-%M-%S') + data_set + '-BNN'
	fig.savefig('00_outputs_graph/'+data_set+'/' + title +'.eps', format='eps', dpi=1000, bbox_inches='tight')



