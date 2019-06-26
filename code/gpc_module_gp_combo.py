# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import importlib

import gpc_utils
importlib.reload(gpc_utils)
from gpc_utils import *


# --- step fn kernel ---

def np_k_0(x,x2):
	k = b_0 + w_0*(x*x2)
	return k

def np_step(x,x2):
	# do the kernel for two scalars
	# only appropriate for 1-D
	# return 1 - np.arccos(x * x2 / (np.linalg.norm(x) * np.linalg.norm(x2)))/np.pi
	k_s = np_k_0(x,x2) / np.sqrt( (np_k_0(x,x) * np_k_0(x2,x2)) )
	theta = np.arccos(k_s)
	w_1 = 1.0
	return w_1*(1 - theta/np.pi)

def np_step_kernel(X,X2=None):
	# cho and saul step kernel?
	cov = np.zeros([X.shape[0],X2.shape[0]])
	if X2 is None:
		X2 = X
	for i in range(X.shape[0]):
		for j in range(X2.shape[0]):
			cov[i,j] = np_step(X[i],X2[j])
	return cov



# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

class gp_model:
	def __init__(self, kernel_type, data_noise, b_0_var=1., w_0_var=1., u_var=1., g_var=1.):

		self.kernel_type = kernel_type
		self.data_noise = data_noise
		self.name_ = 'GP'

		# variance for step fn, relu, erf
		self.b_0_var = b_0_var # first layer bias variance		
		self.w_0_var = w_0_var # first layer weight variance

		# variance for rbf - we use williams 1996 notation
		# i.e. node = exp(-(x-u)^2 / 2*var_g)
		self.g_var = g_var # param of rbf fn (fixed)
		self.u_var = u_var # var of centers, as -> inf, goes to stationary cov dist

		# place holders
		self.mse_unnorm = 0.
		self.rmse = 0.
		self.nll = 0.

		return



	def __relu_kernel(self, X, X2=None):
		# relu kernel from cho and saul, 
		# also Lee 2010 (2018?) helped communicate where to put bias and w_0 variance
		# eq. 6 & 11, Lee, Bahri et al. 2018, could add + b_1

		def relu_inner(x, x2):
			# actually these should be 1/d_in going by Lee. But we leave it normal
			# to be equivalent to our NN ens implementation
			k_x_x = self.b_0_var + self.w_0_var*(np.matmul(x,x.T))/1#/d_in
			k_x2_x2 = self.b_0_var + self.w_0_var*(np.matmul(x2,x2.T))/1
			k_x_x2 = self.b_0_var + self.w_0_var*(np.matmul(x,x2.T))/1

			k_s = k_x_x2 / np.sqrt(k_x_x * k_x2_x2)
			if k_s>1.0: k_s=1.0 # occasionally get some overflow errors

			theta = np.arccos(k_s)
			
			x_bar = np.sqrt(k_x_x)
			x2_bar = np.sqrt(k_x2_x2)

			w_1 = 1.0#self.w_1_var # variance of last layers
			b_1 = 0.0

			return b_1 + w_1/(2*np.pi) * x_bar * x2_bar * (np.sin(theta) + (np.pi-theta)*np.cos(theta))
			
		if X2 is None:
			same_inputs=True
			X2 = X
		else:
			same_inputs=False

		cov = np.zeros([X.shape[0],X2.shape[0]])

		if not same_inputs:
			for i in range(X.shape[0]):
				if i % 10 == 0:
					print('compiling cov, row... '+str(i) + ' / ' + str(X.shape[0]),end='\r')
				for j in range(X2.shape[0]):
					cov[i,j] = relu_inner(X[i], X2[j])
		else: # use symmetry
			for i in range(X.shape[0]):
				if i % 10 == 0:
					print('compiling cov, row... '+str(i) + ' / ' + str(X.shape[0]),end='\r')
				for j in range(i+1):
					cov[i,j] = relu_inner(X[i], X2[j])
			cov += np.tril(cov,k=-1).T
		return cov

	def __relu_per_kernel(self, X, X2=None):
		# relu kernel applied to cos sin warping

		def relu_inner(x, x2):
			# main = np.sin(x - x2) + (np.pi - (x-x2))*np.cos(x-x2)
			# main = (np.pi - (x-x2))
			# const = 1/np.pi

			w_1 = 1.0 # variance of last layers
			b_1 = 0.0

			# return b_1 + w_1*const*main
			p = 1
			# theta = 2*np.pi/p * (x - x2)
			
			if 1:
				# relu with warping first
				self.b_0_var = 1
				self.w_0_var = 0.1 # if set b_0_var = w_0_var, it has no effect
				# theta = np.arccos(np.cos(2*np.pi/p * (x - x2)))

				# theta = np.arccos(  (self.b_0_var + self.w_0_var*np.cos(2*np.pi/p * (x - x2))) / (2*self.b_0_var + 2*self.w_0_var) )

				if 1:
					# usual relu periodic with [cos x, sin x] warping:
					theta = np.arccos(  (self.b_0_var + self.w_0_var*np.cos(2*np.pi/p * (x - x2)) ) / (self.b_0_var + self.w_0_var) )
				else:
					# 'deluxe' relu periodic with [cos x, sin x, x] warping
					cos_inner = (self.b_0_var + self.w_0_var*np.cos(2*np.pi/p * (x - x2)) + self.w_0_var*np.matmul(x,x2.T)) \
								/ np.sqrt(  \
									(self.b_0_var + self.w_0_var*(1+np.matmul(x,x.T))) * \
									(self.b_0_var + self.w_0_var*(1+np.matmul(x2,x2.T))) \
									)
					eps = 1e-6
					cos_inner = np.clip(cos_inner, -1+eps, 1-eps)
					theta = np.arccos(cos_inner)

				return b_1 + w_1/(np.pi) * (np.sin(theta) + (np.pi - theta)*np.cos(theta))
			else:
				# erf with warping first
				# const = 
				return b_1 + w_1*2/np.pi * np.arcsin(np.cos(2*np.pi/p * (x-x2)))



			# return 2*np.pi - theta

			# # check can output arccos cos x
			# x = np.linspace(-20,20,400)
			# y = np.arccos(np.cos(x))
			# y_ = x % (2*np.pi)
			# x % (2*np.pi) < np.pi
			# np.floor(x / (2*np.pi))

			# fig = plt.figure(figsize=(5, 4))
			# ax = fig.add_subplot(111)
			# ax.plot(x,y)
			# fig.show()
			
		if X2 is None:
			same_inputs=True
			X2 = X
		else:
			same_inputs=False

		cov = np.zeros([X.shape[0],X2.shape[0]])

		if not same_inputs:
			for i in range(X.shape[0]):
				if i % 10 == 0:
					print('compiling cov, row... '+str(i) + ' / ' + str(X.shape[0]),end='\r')
				for j in range(X2.shape[0]):
					cov[i,j] = relu_inner(X[i], X2[j])
		else: # use symmetry
			for i in range(X.shape[0]):
				if i % 10 == 0:
					print('compiling cov, row... '+str(i) + ' / ' + str(X.shape[0]),end='\r')
				for j in range(i+1):
					cov[i,j] = relu_inner(X[i], X2[j])
			cov += np.tril(cov,k=-1).T
		return cov


	def __Lrelu_kernel(self, X, X2=None):
		# leaky relu kernel from Tsuchida, 2018, eq. 6

		def Lrelu_inner(x, x2):
			# actually these should be 1/d_in going by Lee. But we leave it normal
			# to be equivalent to our NN ens implementation
			k_x_x = self.b_0_var + self.w_0_var*(np.matmul(x,x.T))/1#/d_in
			k_x2_x2 = self.b_0_var + self.w_0_var*(np.matmul(x2,x2.T))/1
			k_x_x2 = self.b_0_var + self.w_0_var*(np.matmul(x,x2.T))/1

			k_s = k_x_x2 / np.sqrt(k_x_x * k_x2_x2)
			theta = np.arccos(k_s)
			
			x_bar = np.sqrt(k_x_x)
			x2_bar = np.sqrt(k_x2_x2)

			w_1 = 1.0 #self.w_1_var # variance of last layers
			b_1 = 0.0

			a = 0.2 # leaky param

			return b_1 + w_1 * x_bar * x2_bar * ( np.square(1-a)/(2*np.pi) * (np.sin(theta) + (np.pi-theta)*np.cos(theta)) + a*np.cos(theta))
			
		if X2 is None:
			same_inputs=True
			X2 = X
		else:
			same_inputs=False

		cov = np.zeros([X.shape[0],X2.shape[0]])

		if not same_inputs:
			for i in range(X.shape[0]):
				for j in range(X2.shape[0]):
					cov[i,j] = Lrelu_inner(X[i], X2[j])
		else: # use symmetry
			for i in range(X.shape[0]):
				if i % 10 == 0:
					print('compiling cov, row... '+str(i) + ' / ' + str(X.shape[0]),end='\r')
				for j in range(i+1):
					cov[i,j] = Lrelu_inner(X[i], X2[j])
			cov += np.tril(cov,k=-1).T
		return cov


	def __erf_kernel(self, X, X2=None):
		# erf kernel from Williams 1996, eq. 11

		def erf_inner(x,x2):	
			# actually these should be 1/d_in
			k_x_x = 2*(self.b_0_var + self.w_0_var*(np.matmul(x,x.T)))
			k_x2_x2 = 2*(self.b_0_var + self.w_0_var*(np.matmul(x2,x2.T)))
			k_x_x2 = 2*(self.b_0_var + self.w_0_var*(np.matmul(x,x2.T)))

			if 0:
				# hack to show 2d with ARD
				if x.shape[0] != 2:
					raise ValueError('Only works with 2d inputs')
				w1_A = 10. # 10
				w1_B = 0.1 # 0.1
				k_x_x = 2*(self.b_0_var + w1_A*x[0]*x[0] + w1_B*x[1]*x[1])
				k_x2_x2 = 2*(self.b_0_var + w1_A*x2[0]*x2[0] + w1_B*x2[1]*x2[1])
				k_x_x2 = 2*(self.b_0_var + w1_A*x[0]*x2[0] + w1_B*x[1]*x2[1])

			a = k_x_x2 / np.sqrt((1+k_x_x)*(1+k_x2_x2))
			
			w_1 = 1.0 #self.w_1_var # variance of last layers
			b_1 = 0.0

			return b_1 + w_1*2*np.arcsin(a)/np.pi

		if X2 is None:
			same_inputs=True
			X2 = X
		else:
			same_inputs=False

		cov = np.zeros([X.shape[0],X2.shape[0]])

		if not same_inputs:
			for i in range(X.shape[0]):
				for j in range(X2.shape[0]):
					cov[i,j] = erf_inner(X[i],X2[j])
		else:
			for i in range(X.shape[0]):
				for j in range(i+1):
					cov[i,j] = erf_inner(X[i],X2[j])
			# now just reflect - saves recomputing half the matrix
			cov += np.tril(cov,k=-1).T
		return cov

	def __cosine_kernel(self, X, X2=None):
		# kernel for when activations are cosine
		# I haven't seen this derived anywhere,
		# I used e^(-u^2/c)*sin(x*u)*sin(z*u)
		# in https://www.integral-calculator.com/
		# result: (sqrt(pi)*sqrt(c)*(e^(c*x*z)-1)*e^(-(c*z^2)/4-(c*x*z)/2-(c*x^2)/4))/2

		def cosine_inner(x,x2):	
			t = 4.# upper and lower bounds for uniform of weights
			# if set t at same frequency as data, works well...
			s = 0.02 # upper and lower for bias
			# t = 20, s = 0.1 for fav fig data
			if x2 == x:
				# print_w_time('same inputs')
				x2 = x2 + 0.000001
			elif x2 == -x:
				# print_w_time('same inputs negative')
				x2 = x2 + 0.000001

			b = (np.cos(t*(x2+x)+2*s)-np.cos(t*(x2+x)-2*s))/(2*(x2+x))+(2*s*np.sin(t*(x2-x)))/(x2-x)
			b = b/(2*t) # normalise
			# b = b/(2*s)

			if False:
				# or using a uniform dist (basically delta) over w_1 and uniform over b_1 as before
				t = 4.01
				s = 0.00002
				delt = 0.02
				b = ((x2-x)*np.cos((t-delt)*x2+(t-delt)*x+2*s) + (x-x2)*np.cos((t-delt)*x2+(t-delt)*x-2*s) + (-4*s*x2-4*s*x)*np.sin((t-delt)*x2+(delt-t)*x) + (x-x2)*np.cos(t*x2+t*x+2*s) + (x2-x)*np.cos(t*x2+t*x-2*s) + (4*s*x2+4*s*x)*np.sin(t*x2-t*x) ) / (4*(x2**2-x**2))
				# b = b/delt # normalise
				# b = b/(2*s)

			w_1 = 5.0 # variance of last layers
			b_1 = 0.0

			return b_1 + w_1*b

		if X2 is None:
			same_inputs=True
			X2 = X
		else:
			same_inputs=False

		cov = np.zeros([X.shape[0],X2.shape[0]])

		if not same_inputs:
			for i in range(X.shape[0]):
				for j in range(X2.shape[0]):
					cov[i,j] = cosine_inner(X[i],X2[j])
		else:
			for i in range(X.shape[0]):
				for j in range(i+1):
					cov[i,j] = cosine_inner(X[i],X2[j])
			# now just reflect - saves recomputing half the matrix
			cov += np.tril(cov,k=-1).T
		return cov

	def __rbf_kernel(self, X, X2=None):
		# rbf kernel from Williams, 1996, eq. 13
		# don't think we use biases here, not sure about input weights

		def rbf_inner(x, x2):
			# do the kernel for two scalars
			# only appropriate for 1-D, for now...

			var_e = 1/(2/self.g_var + 1/self.u_var)
			var_s = 2*self.g_var + (self.g_var**2)/self.u_var
			var_m = 2*self.u_var + self.g_var

			# williams eq 13
			term1 = np.sqrt(var_e/self.u_var)
			term2 = np.exp(-np.matmul(x,x.T)/(2*var_m))
			term3 = np.exp((np.matmul((x-x2),(x2-x).T))/(2*var_s))
			term4 = np.exp(-np.matmul(x2,x2.T)/(2*var_m))

			w_1 = 1.0 # variance of last layers
			b_1 = 0.0

			return b_1 + w_1*term1*term2*term3*term4
			# return only term3 gives stationary
		
		if X2 is None:
			same_inputs=True
			X2 = X
		else:
			same_inputs=False

		cov = np.zeros([X.shape[0],X2.shape[0]])

		if not same_inputs:
			for i in range(X.shape[0]):
				for j in range(X2.shape[0]):
					cov[i,j] = rbf_inner(X[i],X2[j])
		else:
			for i in range(X.shape[0]):
				for j in range(i+1):
					cov[i,j] = rbf_inner(X[i],X2[j])
			cov += np.tril(cov,k=-1).T
		return cov


	def __SE_kernel(self, X, X2=None):
		# these kernels taked from 
		# Automatic Model Construction with Gaussian Processes, Duvenaud, 2014, p.9
		# also https://scikit-learn.org/stable/modules/gaussian_process.html

		def SE_inner(x, x2):
			var_f = self.SE_const
			length = self.SE_l
			return var_f*np.exp(-np.square(x-x2)/(2*length**2))

		if X2 is None:
			X2 = X

		cov = np.zeros([X.shape[0],X2.shape[0]])

		for i in range(X.shape[0]):
			for j in range(X2.shape[0]):
				cov[i,j] = SE_inner(X[i],X2[j])

		return cov


	def __periodic_kernel(self, X, X2=None):
		# try with periodic_1 in DataGen, p=2

		def periodic_inner(x, x2):
			var_f = self.per_const
			length = self.per_l # longer is smoother
			p = self.per_p # periodicity (how often repeats on x axis)
			return var_f * np.exp(-2/(length**2) * np.square(np.sin(np.pi*(x-x2)/p)))
			# return var_f * np.exp(-2/(length**2) * np.square(np.sin(np.pi*(x[0]-x2[0])/p)))

		if X2 is None:
			X2 = X

		cov = np.zeros([X.shape[0],X2.shape[0]])

		for i in range(X.shape[0]):
			for j in range(X2.shape[0]):
				cov[i,j] = periodic_inner(X[i],X2[j])

		return cov


	def __linear_kernel(self, X, X2=None):

		def linear_inner(x, x2):
			var_f = 1.0
			c =  1.0 # constant (goes through this x point when y=0)
			return var_f * np.matmul((x - c),(x2 - c))

		if X2 is None:
			X2 = X

		cov = np.zeros([X.shape[0],X2.shape[0]])

		for i in range(X.shape[0]):
			for j in range(X2.shape[0]):
				cov[i,j] = linear_inner(X[i],X2[j])

		return cov


	def __add_lin_per_kernel(self, X, X2=None):
		# try 

		def inner(x, x2):
			var_f = 1.0
			c = 0 # constant
			length = 0.9
			p = 2.0 # periodicity
			# I haven't written a separate method for each combo, just tinker with the below line
			return var_f * np.matmul((x - c),(x2 - c)) + var_f * np.exp(-2/(length**2) * np.square(np.sin(np.pi*(x-x2)/p)))
			# return var_f * np.matmul((x - c),(x2 - c)) * var_f * np.exp(-2/(length**2) * np.square(np.sin(np.pi*(x-x2)/p)))

		if X2 is None:
			X2 = X

		cov = np.zeros([X.shape[0],X2.shape[0]])

		for i in range(X.shape[0]):
			for j in range(X2.shape[0]):
				cov[i,j] = inner(X[i],X2[j])

		return cov

	def __combo_kernel(self, X, X2=None):
		# rbf kernel from Williams, 1996, eq. 13
		# don't think we use biases here, not sure about input weights

		def combo_inner(x, x2):
			# whether to add or multiply
			self.add_or_mult = '+' # *, +
			self.k_a_type = 'per' # relu, erf, per, se
			self.k_b_type = 'relu'

			# components of each
			if self.add_or_mult == '+':
				self.comp_a = 0.7
				self.comp_b = 2. - self.comp_a
			else:
				self.comp_a = 1.0
				self.comp_b = 1.0

			# -- first kernel --

			# relu, erf
			self.b_0_var = 1.0
			self.w_0_var = self.b_0_var
			# self.w_1_var = 0.05

			# periodic
			self.per_p = 1.2 # 1.0
			self.per_const = 3
			self.per_l = 0.9

			# squared exp
			self.SE_const = 1.0
			self.SE_l = 10

			# rbf NN
			self.g_var = 1.0
			self.u_var = 1.0


			if self.k_a_type == 'relu':
				k_a = self.comp_a * self.__relu_kernel(np.atleast_2d(x),np.atleast_2d(x2))
			elif self.k_a_type == 'leaky':
				k_a = self.comp_a * self.__Lrelu_kernel(np.atleast_2d(x),np.atleast_2d(x2))
			elif self.k_a_type == 'erf':
				k_a = self.comp_a * self.__erf_kernel(np.atleast_2d(x),np.atleast_2d(x2))
			elif self.k_a_type == 'per':
				k_a = self.comp_a * self.__periodic_kernel(x,x2)
				# uncomment below for 2-d separation of inputs example:
				#k_a = self.comp_a * self.__periodic_kernel(np.atleast_2d(x[0]),np.atleast_2d(x2[0]))
			elif self.k_a_type == 'se':
				k_a = self.comp_a * self.__SE_kernel(x,x2)
			elif self.k_a_type == 'rbf':
				k_a = self.comp_a * self.__rbf_kernel(np.atleast_2d(x),np.atleast_2d(x2))
			else:
				raise Exception('not implemented')

			# -- second kernel --

			# relu, erf
			self.b_0_var = 10.0 # checked 
			self.w_0_var = self.b_0_var
			# self.w_1_var = 1.0

			# periodic
			self.per_p = 1.0/10 # smaller is faster 
			self.per_const = 0.2388
			self.per_l = 1.732 # how noisy fn is, larger = smoother

			# squared exp
			self.SE_const = 1.0
			self.SE_l = 10

			# rbf NN
			self.g_var = 1.0
			self.u_var = 10.0

			if self.k_b_type == 'relu':
				k_b = self.comp_b * self.__relu_kernel(np.atleast_2d(x),np.atleast_2d(x2))
				# uncomment below for 2-d separation of inputs
				#k_b = self.comp_b * self.__relu_kernel(np.atleast_2d(x[1]),np.atleast_2d(x2[1]))
			elif self.k_b_type == 'leaky':
				k_b = self.comp_b * self.__Lrelu_kernel(np.atleast_2d(x),np.atleast_2d(x2))
			elif self.k_b_type == 'erf':
				k_b = self.comp_b * self.__erf_kernel(np.atleast_2d(x),np.atleast_2d(x2))
			elif self.k_b_type == 'per':
				k_b = self.comp_b * self.__periodic_kernel(x,x2)
			elif self.k_b_type == 'se':
				k_b = self.comp_b * self.__SE_kernel(x,x2)
			elif self.k_b_type == 'rbf':
				k_b = self.comp_b * self.__rbf_kernel(np.atleast_2d(x),np.atleast_2d(x2))
			else:
				raise Exception('not implemented')

			if self.add_or_mult == '+':
				return k_a + k_b
				# return k_a
				# return (k_a + k_b) * self.__relu_kernel(np.atleast_2d(x),np.atleast_2d(x2)) # (erf + erf)*relu
				# return k_a
			elif self.add_or_mult == '*':
				return k_a * k_b
				# self.per_l = 0.6
				# self.per_p = 1.0/2
				# return k_a * k_b + 10*self.__periodic_kernel(x,x2) # relu*relu + per
			else:
				raise Exception('not implemented')
		
		if X2 is None:
			same_inputs=True
			X2 = X
		else:
			same_inputs=False

		cov = np.zeros([X.shape[0],X2.shape[0]])

		if not same_inputs:
			for i in range(X.shape[0]):
				for j in range(X2.shape[0]):
					cov[i,j] = combo_inner(X[i],X2[j])
		else:
			for i in range(X.shape[0]):
				for j in range(i+1):
					cov[i,j] = combo_inner(X[i],X2[j])
			cov += np.tril(cov,k=-1).T
		return cov




	def run_inference(self, x_train, y_train, x_predict, print=False):
		''' this is why we're here - do inference '''

		if self.kernel_type == 'relu' or self.kernel_type == 'softplus':
			kernel_fn = self.__relu_kernel
			# kernel_fn = self.__relu_kernel_tf
		elif self.kernel_type == 'Lrelu':
			kernel_fn = self.__Lrelu_kernel
		elif self.kernel_type == 'rbf':
			kernel_fn = self.__rbf_kernel
		elif self.kernel_type == 'erf':
			kernel_fn = self.__erf_kernel
		elif self.kernel_type == 'step':
			kernel_fn = step_kernel
		elif self.kernel_type == 'cosine': # cos activation
			kernel_fn = self.__cosine_kernel
		elif self.kernel_type == 'SE': # squared exp
			kernel_fn = self.__SE_kernel
		elif self.kernel_type == 'periodic': # periodic
			kernel_fn = self.__periodic_kernel
		elif self.kernel_type == 'linear': # linear kernel
			kernel_fn = self.__linear_kernel
		elif self.kernel_type == 'add_lin_per': # combos of kernels
			kernel_fn = self.__add_lin_per_kernel
		elif self.kernel_type == 'combo': # flexible combos of kernels
			kernel_fn = self.__combo_kernel
		elif self.kernel_type == 'relu_per': # flexible combos of kernels
			kernel_fn = self.__relu_per_kernel

		# d is training data, x is test data
		if print: print_w_time('beginning inference')
		cov_dd = kernel_fn(x_train) + np.identity(x_train.shape[0])*self.data_noise
		if print: print_w_time('compiled cov_dd')
		cov_xd = kernel_fn(x_predict, x_train)
		if print: print_w_time('compiled cov_xd')
		cov_xx = kernel_fn(x_predict,x_predict)
		if print: print_w_time('compiled cov_xx')

		# if print: print_w_time('inverting matrix dims: '+ str(cov_dd.shape))
		# cov_dd_inv = np.linalg.inv(cov_dd) # could speed this up w cholesky or lu decomp
		# if print: print_w_time('matrix inverted')

		# cov_pred = cov_xx - np.matmul(np.matmul(cov_xd,cov_dd_inv),cov_xd.T)
		# y_pred_mu = np.matmul(np.matmul(cov_xd,cov_dd_inv),y_train)
		# # y_pred_var = np.atleast_2d(np.diag(cov_pred)).T
		# y_pred_var = np.atleast_2d(np.diag(cov_pred) + self.data_noise).T
		# y_pred_std = np.sqrt(y_pred_var)
		# print_w_time(cov_dd)
		# print_w_time(cov_xd)

		# p 19 of rasmussen
		L = np.linalg.cholesky(cov_dd)
		alpha = np.linalg.solve(L.T,np.linalg.solve(L,y_train))
		y_pred_mu = np.matmul(cov_xd,alpha)
		v = np.linalg.solve(L,cov_xd.T)
		cov_pred = cov_xx - np.matmul(v.T,v)

		# print_w_time(cov_pred)

		y_pred_var = np.atleast_2d(np.diag(cov_pred) + self.data_noise).T
		# print_w_time(y_pred_var)
		# y_pred_var = np.abs(y_pred_var)
		y_pred_std = np.sqrt(y_pred_var)


		if print: print_w_time('calculating log likelihood')
		# marg_log_like = - np.matmul(y_train.T,np.matmul(cov_dd_inv,y_train))/2 - np.log(np.linalg.det(cov_dd))/2 - x_train.shape[0]*np.log(2*np.pi)/2
		# have problems with this going to zero

		# marg_log_like = - np.matmul(y_train.T,np.matmul(cov_dd_inv,y_train))/2 - np.sum(np.log(np.diag(L))) - x_train.shape[0]*np.log(2*np.pi)/2
		marg_log_like = - np.matmul(y_train.T,alpha)/2 - np.sum(np.log(np.diag(L))) - x_train.shape[0]*np.log(2*np.pi)/2
		
		# print_w_time(L)
		# a = np.sum(np.log(np.diag(L)))
		# print_w_time(a)
		# a = np.matmul(y_train.T,np.matmul(cov_dd_inv,y_train))/2
		# print_w_time(a)
		# a = np.linalg.det(cov_dd) # this goes to zero sometimes...
		# print_w_time(a)
		# a = np.log(np.linalg.det(cov_dd))/2 
		# print_w_time(a)
		if print: print_w_time('matrix ops complete')

		self.cov_xx = cov_xx
		self.cov_dd = cov_dd
		# self.cov_dd_inv = cov_dd_inv
		self.cov_xd = cov_xd
		self.cov_xx = cov_xx
		self.cov_pred = cov_pred
		self.y_pred_mu = y_pred_mu
		self.y_pred_std = y_pred_std
		self.y_pred_var = y_pred_var
		self.x_train = x_train
		self.y_train = y_train
		self.x_predict = x_predict

		self.y_pred_mu = y_pred_mu
		self.y_pred_std = y_pred_std
		self.marg_log_like = marg_log_like

		return y_pred_mu, y_pred_std


	def cov_visualise(self):
		''' display heat map of cov matrix over 1-d input '''

		# plot cov matrix
		fig = plt.figure()
		plt.imshow(self.cov_xx, cmap='hot', interpolation='nearest')
		if self.kernel_type != 'rbf':
			title = self.kernel_type + ', cov matrix, b_0: ' + str(self.b_0_var) + ', w_0: ' + str(self.w_0_var)
		else:
			title = self.kernel_type + ', cov matrix, g_var: ' + str(self.g_var) + ', u_var: ' + str(self.u_var)
		plt.title(title)
		plt.colorbar()
		fig.show()

		return


	def priors_visualise(self, n_draws=10, is_save=False):
		# 1-D only, plot priors
		# we currently have data noise included in this, could remove it to get smooth

		from cycler import cycler
		# col_list = ['fuchsia','mediumblue','dodgerblue','g','r','k']
		col_list = ['mediumblue','fuchsia','plum','grey','blue']
		plt.rcParams['axes.prop_cycle'] = cycler(color=col_list)

		print_w_time('getting priors')
		# get some priors
		y_samples_prior = np.random.multivariate_normal(
			np.zeros(self.x_predict.shape[0]), self.cov_xx, n_draws).T # mean, covariance, size

		# plot priors
		fig = plt.figure(figsize=(5, 3))
		ax = fig.add_subplot(111)
		ax.plot(self.x_predict, y_samples_prior, lw=2.0, label='Priors')
		# ax.plot(self.x_predict, y_samples_prior, 'k', alpha=0.1)
		# plt.xlabel('$x$')
		# plt.ylabel('$f(x)$')
		title = get_time() + self.kernel_type + ', priors, b_0: ' + str(self.b_0_var) + ', w_0: ' + str(self.w_0_var)
		title = title.replace('_','-')    
		# plt.title(title)
		ax.set_xlim(self.x_predict.min(), self.x_predict.max())
		fig.show()

		if is_save:
			ax.set_yticklabels([])
			ax.set_xticklabels([])
			ax.set_yticks([])
			ax.set_xticks([])
			fig.savefig('00_outputs_graph/' + title +'.eps', format='eps', dpi=500, bbox_inches='tight')

		return

	def priors_visualise_2d(self, n_draws=10, is_save=False):
		# 2-D only, plot priors

		print_w_time('getting priors')
		# get some priors
		self.y_samples_prior = np.random.multivariate_normal(
			np.zeros(self.x_predict.shape[0]), self.cov_xx, 1).T # mean, covariance, size

		# plot priors
		fig = plt.figure(figsize=(8, 6))

		ax = fig.gca(projection='3d')

		n = int(np.sqrt(self.x_predict.shape[0]))
		xs = self.x_predict[:,0].reshape((-1, n))
		ys = self.x_predict[:,1].reshape((-1, n))
		zs = self.y_samples_prior[:,0].reshape((-1, n))

		# plot the surface

		if 1:
			import matplotlib
			surf = ax.plot_surface(xs, ys, zs,color='aqua', # aqua lightskyblue
								linewidth=0.2, alpha=1.0, edgecolors='k', 
								antialiased=True, rstride=1, cstride=1, shade=True, 
								lightsource=matplotlib.colors.LightSource(altdeg=40))
		else:
			surf = ax.plot_surface(xs, ys, zs,cmap=plt.cm.cool, # cool, spring, coolwarm, jet, summer
								linewidth=0.2, alpha=1.0, edgecolors='k', 
								antialiased=True, rstride=1, cstride=1)
		

		# ax.plot_trisurf(self.x_predict[:,0], self.x_predict[:,1], self.y_samples_prior[:,0], \
		# 	cmap=plt.cm.jet, alpha=1.0, shade=False, linewidth=0.9, color='k', linecolor='k')

		# Customize the z axis.
		# ax.set_zlim(-1.01, 1.01)

		fig.show()

		title = get_time() + self.kernel_type + ', priors, b_0: ' + str(self.b_0_var) + ', w_0: ' + str(self.w_0_var)
		title = title.replace('_','-')  

		if 1: # turn off axes

			# make the panes transparent
			ax.xaxis.set_pane_color((0.1, 0.1, 0.6, 0.))
			ax.yaxis.set_pane_color((0.1, 0.1, 0.6, 0.))
			ax.zaxis.set_pane_color((0.1, 0.1, 0.6, 0.1))
			# ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

			# make the grid lines transparent
			ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
			ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
			ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

			# turn off everything
			# ax.set_axis_off()

			ax.set_yticklabels([])
			ax.set_xticklabels([])
			ax.set_zticklabels([])
			# ax.set_yticks([])
			# ax.set_xticks([])
			ax.set_zticks([])

			ax.w_zaxis.line.set_lw(0.)

		if is_save:
			# fig.savefig('00_outputs_graph/' + title +'.eps', format='eps', dpi=500, bbox_inches='tight')
			fig.savefig('00_outputs_graph/' + title +'.pdf', format='pdf', dpi=500, bbox_inches='tight')

		return


	def posts_draw_visualise(self, n_draws=10, is_graph=True):
		# 1-D only, plot posteriors
		# we currently have data noise included in this, could remove it to get smooth

		# sample from posterior
		y_samples_post = np.random.multivariate_normal(
			self.y_pred_mu.ravel(), self.cov_pred, n_draws).T # mean, covariance, size

		# plot priors
		if is_graph:
			fig = plt.figure()
			plt.plot(self.x_predict, y_samples_post, color='k',alpha=0.5,lw=0.5, label=u'Priors')
			plt.plot(self.x_train, self.y_train, 'r.', markersize=14, label=u'Observations', markeredgecolor='k',markeredgewidth=0.5)
			plt.xlabel('$x$')
			plt.ylabel('$f(x)$')
			title = self.kernel_type + ', posteriors, b_0: ' + str(self.b_0_var) + ', w_0: ' + str(self.w_0_var)
			plt.title(title)
			# plt.xlim(-6, 6)
			fig.show()

		self.y_preds = y_samples_post.T

		y_pred_mu_draws = np.mean(self.y_preds,axis=0)
		y_pred_std_draws = np.std(self.y_preds,axis=0, ddof=1)

		# add on data noise
		# do need to add on for GP!
		y_pred_std_draws = np.sqrt(np.square(y_pred_std_draws) + self.data_noise)

		self.y_pred_mu_draws = y_pred_mu_draws
		self.y_pred_std_draws = y_pred_std_draws

		return





