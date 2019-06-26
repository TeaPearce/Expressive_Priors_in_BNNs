import numpy as np
import gym
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from scipy import stats
from keras import backend as K
import time

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 10})
plt.rcParams['text.usetex'] = True

start_time = datetime.datetime.now()
print('\nstart_time:', start_time.strftime('%H:%M:%S'))

is_return_raw_theta = True
env = gym.make('Pendulum-v0')
env.env.tp_theta_init_max = 3*np.pi # np.pi is default
env.env.dynamics_mode = 'thread' # pure, thread
env.env.is_return_raw_theta = is_return_raw_theta
env._max_episode_steps = 200 # 200 by default

# env.seed(1)


model_type='periodic_times_tanh' # theta_raw, periodic, periodic_times_tanh
if is_return_raw_theta and model_type == 'theta_raw':
	state_size = 2
else:
	state_size = 3

n_actions = 3 #
n_episodes = 2000 # 1500

n_ens = 5 # number of NNs in the ensemble
what_reg = 'anc' # reg free anc - regularisation type for ensemble
n_hidden = 50 # no. hidden nodes per layer
n_layers = 2 # no. hidden layers
l_rate = 0.01 # 0.01 w 100 rollouts find for 1 NN, but ensemble needs gentler
gamma = 0.98
n_eps_rollout = 20
rollouts_size = int((n_eps_rollout-1)*env._max_episode_steps) # number of time steps before training
batch_size = 100 # normal batch size for NN
train_reps = 400 # how many batches to pull out after every rollouts_size episodes
n_burn_in = 20-1 # no epsidodes to randomly sample from before training begins
n_runs = 3 # no. runs to repeat over

buff_max = 40000 # experience buffer size
n_target_updates = 1-1 # number of training updates to wait for before updating target NN

data_noise_var = 0. #1 #25 # # 1.0
activation_fn = tf.nn.relu # tf.nn.tanh relu

move_scale = 1.5 # how much to scale movement by (0-1), was 0.5
move_noise = 0. # 0.01 # how much noise to add - stdev 0.6
reward_noise_std = 0. # can learn with 0.4 just about, 1.0 ends up struggling
if n_ens>1:
	prob_rand_action=0.
else:
	prob_rand_action = 0.5 # probability of selecting a random action after burn in

is_render = 0
is_save_graph = 0
is_save_results = 0
render_every = 500 # render a few episodes every now and then to check on learning

action_list = ['fwd', 'bwd', 'pause']


def obs_tidy(obs_in):
	# could do preprocessing of inputs here
	return obs_in


def vis_current_state(episodes=10):
	# have a look at what it's learnt
	for episode in range(1,episodes):
		print('\t\t  fwd    bwd    l    r    up    dwn')
		obs = env.reset()
		eval_rewards=[]
		time_step = 0
		while True:
			time.sleep(0.005)
			time_step+=1
			env.render()
			obs = obs_tidy(obs)

			# NN action selection, avg over all NN policies
			action_pred_q_ens=[]
			for m in range(n_ens):
				action_pred_q_ens.append(NNs[m].predict(sess,obs))
			action_pred_q = np.mean(action_pred_q_ens,axis=0)
			action_pred_q_std = np.std(action_pred_q_ens,axis=0)

			# select best action
			id = np.argmax(action_pred_q)

			# action selection - select the best action, no randomness
			# id = np.argmax(action_pred_probs) 

			if id == 0: # clockwise
				action = np.array([1.])
			elif id == 1: # anticlockwise
				action = np.array([-1.])
			elif id == 2: # nothing
				action = np.array([0.])

			# scale movement
			action = action* move_scale 

			# make actions stochastic
			# if id != 6: # no noise if stationary
			if id != 100: # always add noise
				action_noise = np.random.normal(loc=action,scale=move_noise,size=action.shape)
			else:
				action_noise = action.copy()

			# implement action
			obs, reward, done, info = env.step(action_noise)
			# eval_rewards.append(reward)
			eval_rewards.append(reward) # add noise to reward

			# for m in range(n_ens):
			# 	print(np.round(action_pred_q_ens[m],3))
			# print(np.round(action_pred_q,3),end='\r')
			print('obs',np.round(obs_tidy(obs),3),'| action',np.round(action_pred_q,3),end='\r')

			if done:
				print('\n',time_step)
				print('\nrewards:',np.round(np.mean(eval_rewards),3))
				break
	return

class NN():
	def __init__(self, n_hidden, n_layers, reg_in):

		self.reg_in = reg_in

		# get initialisations, and regularisation values
		self.W1_var = 1.0/state_size
		self.W1_lambda = data_noise_var/self.W1_var
		self.W1_ancs = np.random.normal(loc=0,scale=np.sqrt(self.W1_var),size=[state_size,n_hidden])
		self.W1_init = np.random.normal(loc=0,scale=np.sqrt(self.W1_var),size=[state_size,n_hidden])

		self.b1_var = self.W1_var
		self.b1_lambda = data_noise_var/self.b1_var
		self.b1_ancs = np.random.normal(loc=0,scale=np.sqrt(self.b1_var),size=[n_hidden])
		self.b1_init = np.random.normal(loc=0,scale=np.sqrt(self.b1_var),size=[n_hidden])

		self.W2_var = 1/n_hidden
		self.W2_lambda = data_noise_var/self.W2_var
		self.W2_ancs = np.random.normal(loc=0,scale=np.sqrt(self.W2_var),size=[n_hidden,n_hidden])
		self.W2_init = np.random.normal(loc=0,scale=np.sqrt(self.W2_var),size=[n_hidden,n_hidden])

		self.b2_var = self.W2_var
		self.b2_lambda = data_noise_var/self.b2_var
		self.b2_ancs = np.random.normal(loc=0,scale=np.sqrt(self.b2_var),size=[n_hidden])
		self.b2_init = np.random.normal(loc=0,scale=np.sqrt(self.b2_var),size=[n_hidden])

		self.W3_var = 10/n_hidden
		self.W3_lambda = data_noise_var/self.W3_var
		self.W3_ancs = np.random.normal(loc=0,scale=np.sqrt(self.W3_var),size=[n_hidden, n_actions])
		self.W3_init = np.random.normal(loc=0,scale=np.sqrt(self.W3_var),size=[n_hidden, n_actions])

		print('W1_lambda',self.W1_lambda)
		print('W2_lambda',self.W2_lambda)
		print('W3_lambda',self.W3_lambda)

		if is_return_raw_theta: # two inputs
			self.input_ = tf.placeholder(tf.float32, [None, 2])
		else:	# three inputs
			self.input_ = tf.placeholder(tf.float32, [None, 3])
		self.actions_ = tf.placeholder(tf.float32, [None, n_actions])
		self.q_target = tf.placeholder(tf.float32, [None], name='target')
		self.N_ = tf.placeholder(tf.float32, [1]) # this is total size of data
		# self.discounted_rewards_ = tf.placeholder(tf.float32, [None,])



		# warp input
		if model_type == 'theta_raw':
			self.input_warped = self.input_
		elif model_type == 'periodic':
			self.th_ = self.input_[:, 0] 		# raw angle
			self.th_dot_ = self.input_[:, -1] 	# angular velocity
			p = 2*np.pi # here we just want the period to be 2 pi as is in radians
			p_const = 2*np.pi/p
			self.input_warped = tf.stack([tf.cos(self.th_*p_const),tf.sin(self.th_*p_const),self.th_dot_], axis=-1)
		elif model_type == 'periodic_times_tanh':
			self.th_ = self.input_[:, 0] 		# raw angle
			self.th_dot_ = self.input_[:, -1] 	# angular velocity
			p = 2*np.pi # here we just want the period to be 2 pi as is in radians
			p_const = 2*np.pi/p
			# self.input_warped = tf.stack([tf.cos(self.th_*p_const),tf.sin(self.th_*p_const)], axis=-1)
			self.input_warped = tf.stack([tf.cos(self.th_*p_const),tf.sin(self.th_*p_const),self.th_dot_], axis=-1)
		else:
			raise Exception('not implemented')

		self.layer_1_w = tf.layers.Dense(n_hidden,
			activation=activation_fn,
			kernel_initializer=tf.keras.initializers.Constant(value=self.W1_init),
			bias_initializer=tf.keras.initializers.Constant(value=self.b1_init))
		self.layer_1 = self.layer_1_w.apply(self.input_warped)
		# self.layer_1 = self.layer_1_w.apply(self.input_)

		self.layer_2_w = tf.layers.Dense(n_hidden,
			activation=activation_fn,
			kernel_initializer=tf.keras.initializers.Constant(value=self.W2_init),
			bias_initializer=tf.keras.initializers.Constant(value=self.b2_init))
		self.layer_2 = self.layer_2_w.apply(self.layer_1)

		if model_type == 'periodic_times_tanh':
			print('connecting sub BNN')
			# create a second single layer NN
			self.layer_1_w_b = tf.layers.Dense(n_hidden,
				activation=tf.tanh,
				kernel_initializer=tf.initializers.random_normal(mean=0.0,stddev=0.2/n_actions),
				bias_initializer=tf.initializers.random_normal(mean=0.0,stddev=0.2/n_actions))
			self.layer_1_b = self.layer_1_w_b.apply(tf.expand_dims(self.th_,axis=-1))

			# element wise multiplication at output of hidden layers
			self.layer_2_combo = tf.multiply(self.layer_1_b, self.layer_2)
			# self.layer_2_combo = self.layer_2
		else:
			self.layer_2_combo = self.layer_2

		self.output_w = tf.layers.Dense(n_actions, 
			activation=None, use_bias=False,
			kernel_initializer=tf.keras.initializers.Constant(value=self.W3_init))

		if n_layers == 1:
			self.fc_out = self.output_w.apply(self.layer_1) # skip this layer
		elif n_layers == 2:
			self.fc_out = self.output_w.apply(self.layer_2_combo)
		else:
			raise Exception('>2 layers not implemented')
		
		self.q = tf.reduce_sum(tf.multiply(self.fc_out, self.actions_), axis=1)
		self.loss_ = tf.reduce_mean(tf.square(self.q_target - self.q))
		# self.train_opt = tf.train.AdamOptimizer(l_rate).minimize(self.loss_)
		# self.train_opt = tf.train.GradientDescentOptimizer(l_rate).minimize(self.loss)

		return

	def anchor(self):
		'''method to set loss to account for anchoring'''

		if self.reg_in == 'free':
			# no regularisation
			print('unconstraining')
			self.W1_lambda = 0.
			self.b1_lambda = 0.
			self.W2_lambda = 0.
			self.b2_lambda = 0.
			self.W3_lambda = 0.

		elif self.reg_in == 'reg':
			# do normal regularisation
			print('regularising')
			self.W1_ancs = np.zeros_like(self.W1_ancs)
			self.b1_ancs = np.zeros_like(self.b1_ancs)
			self.W2_ancs = np.zeros_like(self.W2_ancs)
			self.b2_ancs = np.zeros_like(self.b2_ancs)
			self.W3_ancs = np.zeros_like(self.W3_ancs)

		elif self.reg_in == 'anc':
			print('anchoring')

		# set squared loss around it
		self.loss_anchor =  self.W1_lambda * tf.reduce_sum(tf.square(self.W1_ancs - self.layer_1_w.kernel))
		self.loss_anchor += self.b1_lambda * tf.reduce_sum(tf.square(self.b1_ancs - self.layer_1_w.bias))
		self.loss_anchor += self.W2_lambda * tf.reduce_sum(tf.square(self.W2_ancs - self.layer_2_w.kernel))
		self.loss_anchor += self.b2_lambda * tf.reduce_sum(tf.square(self.b2_ancs - self.layer_2_w.bias))
		self.loss_anchor += self.W3_lambda * tf.reduce_sum(tf.square(self.W3_ancs - self.output_w.kernel))

		# combine with original loss
		# self.loss_ = self.loss_ + self.loss_anchor / tf.shape(self.input_, out_type=tf.int64)[0] 
		# self.loss_ = self.loss_ + self.loss_anchor / tf.to_float(tf.shape(self.input_, out_type=tf.int64))[0] 
		self.loss_ = self.loss_ + self.loss_anchor / self.N_

		# reset optimiser
		self.train_opt = tf.train.AdamOptimizer(l_rate).minimize(self.loss_)
		return

	def train(self, sess, feed_b):
		loss_, _ = sess.run([self.loss, self.train_opt], feed_dict=feed_b)
		return loss_

	def predict(self, sess, obs):
		if is_return_raw_theta:
			return sess.run(self.fc_out, feed_dict={self.input_: obs.reshape([1,2])})
		else:
			return sess.run(self.fc_out, feed_dict={self.input_: obs.reshape([1,3])})

results_train_hist=[]
for run in range(n_runs):
	print('\n\n-- run', run+1, 'of', n_runs, ' --\n\n')

	# trainable variables was not resetting properly after each run
	tf.reset_default_graph()

	# live NNs
	NNs=[]
	for m in range(n_ens):
		NNs.append(NN(n_hidden, n_layers, what_reg))

	# create a copy of target NNs
	NNs_freeze=[]
	for m in range(n_ens):
		NNs_freeze.append(NN(n_hidden, n_layers, 'free'))

	# set anchoring
	for m in range(n_ens):
		NNs[m].anchor()

	# https://stackoverflow.com/questions/42061224/duplicate-a-network-in-tensorflow/42076464#42076464
	vars = tf.trainable_variables()
	# copy_ops = [vars[ix+len(vars)//2].assign(var.value()) for ix, var in enumerate(vars[0:len(vars)//2])]
	copy_ops=[]
	half_vars_len = len(vars)//2
	for i in range(half_vars_len):
		copy_ops.append(vars[half_vars_len+i].assign(vars[i]))
		# copies first half of variables to second half, which is exactly what I want

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	sess.run(copy_ops) # make target and live NNs are equal at beginning

	# run loop
	all_time_taken = []
	batch_states, batch_actions, batch_rewards = [],[],[]
	training_hist = []
	experience_replay = [] # q learning
	target_count = 0 # how often to update target NN
	for episode in range(1,n_episodes+1):
		print(episode,end='\r')
		
		obs = env.reset()
		obs = obs_tidy(obs)

		# NN_choose = np.random.randint(0,n_ens) # follow a single NN over the whole episode
		NN_choose = 1

		episode_rewards = []
		action = np.array([0.,0.,0.,0.]) # init action for sticky keys
		done = False
		t_step=0
		while True:
			action_prev = action.copy() # store action as prev action
			t_step += 1

			if is_render: 
				env.render()
			elif episode %render_every == 0: # visualise what it's doing every now and then
				vis_current_state(episodes=5)

			# NN action selection, avg over all NN policies
			action_pred_q_ens=[]
			for m in range(n_ens):
				action_pred_q_ens.append(NNs[m].predict(sess,obs))
			action_pred_q = np.mean(action_pred_q_ens,axis=0)
			action_pred_q_std = np.std(action_pred_q_ens,axis=0)

			# if model each action q dist as gaussian, take one sample and see which largest
			gauss_sample = np.random.normal(loc=action_pred_q,scale=action_pred_q_std)

			# select best action
			id = np.argmax(gauss_sample) # also works for one NN
			# id = np.argmax(action_pred_q)

			# select action according to prob is best
			# id = np.random.choice(range(n_actions), p=action_pred_probs.ravel())

			# choose a random action with some prob - epsilon greedy
			if episode < n_burn_in:
				prob_rand = 1.
			else:
				prob_rand = prob_rand_action
			if (1. - prob_rand) < np.random.rand():
				id = np.random.randint(0,n_actions)

			# convert index of action to movement
			if id == 0: # clockwise
				action = np.array([1.])
			elif id == 1: # anticlockwise
				action = np.array([-1.])
			elif id == 2: # nothing
				action = np.array([0.])

			# scale movement
			action = action*move_scale 

			# make actions stochastic			
			# if id != 6: # no noise if stationary
			if id != 100:
				action_noise = np.random.normal(loc=action,scale=move_noise,size=action.shape)
			else:
				action_noise = action.copy()

			# one hot encoding for training NN - selected action
			action_one_hot = np.zeros((1, n_actions))
			action_one_hot[0, id] = 1

			# sticky action
			if False:
				stick = 0.25 # prob of repeating action
				repeat_action = np.random.rand()
				if repeat_action < stick:
					action = action_prev.copy()

			batch_states.append(obs) # current state at t
			batch_actions.append(action_one_hot) # current action at t

			obs_1, reward, done, info = env.step(action_noise)
			obs_1 = obs_tidy(obs_1)

			batch_rewards.append(reward) # reward after action at t+1
			episode_rewards.append(reward)

			# experience_replay.append([obs, action_one_hot, reward, obs_1]) # noiseless reward
			experience_replay.append([obs, action_one_hot, np.random.normal(loc=reward,scale=reward_noise_std), obs_1])

			obs = obs_1.copy()

			if done:
				
				batch_rewards_avg = np.mean(batch_rewards)
				all_time_taken.append(t_step) # only if quite after found target
				avg_time_taken = np.mean(all_time_taken[-100:])
				episode_rewards = np.sum(episode_rewards)
				
				# print("episode: ", episode, ", reward: ", round(episode_rewards_sum,4), ", last 100:", round(recent_rewards,4), ", avg_time_taken:", round(avg_time_taken,4))
				# print("episode: ", episode, ", batch_rewards_avg:", round(batch_rewards_avg,4), ", episode_rewards:", round(episode_rewards,4))
				
				# train on 100 batch at a time
				if len(batch_states) > rollouts_size and episode > n_burn_in:
					print("episode: ", episode, ", batch_rewards_avg:", round(batch_rewards_avg,4))
					print('buffer length:',len(experience_replay))
					# print('\n- training batch -\n')
					# disc_norm_rewards = discount_and_normalize_rewards(batch_rewards)

					if len(experience_replay) > buff_max:
						# delete the first chunk of time steps
						# del experience_replay[0:10000]
						del experience_replay[0:int(buff_max/3)]

					# update target NNs params
					target_count += 1
					if target_count > n_target_updates:
						if 1:
							print('-- updating target NN --')

							# print('-- params pre update --')
							# print('\norig NN\n',sess.run(vars[1]))
							# print('\ntarget NN\n',sess.run(vars[half_vars_len+1]))

							# map(lambda x: sess.run(x), copy_ops)
							sess.run(copy_ops)
							target_count=0

							# print('-- params after update --')
							# print('\norig NN\n',sess.run(vars[1]))
							# print('\ntarget NN\n',sess.run(vars[half_vars_len+1]))

					for rep in range(train_reps): # repeat training multiple times

						# should code this to select batches properly later
						samples_i = np.random.choice(range(len(experience_replay)),size=batch_size,replace=False)
						samples=[]

						# other samples
						for k in samples_i:
							samples.append(np.array(experience_replay[k]))
						samples = np.array(samples)

						replay_obs = np.vstack(samples[:,0])
						replay_actions = np.vstack(samples[:,1])
						replay_rewards = np.array(samples[:,2],dtype='float32')
						replay_obs_1 = np.vstack(samples[:,3])
						# episode_ends = (replay_observations_ == np.zeros(replay_observations[0].shape)).all(axis=1) # in case final episode

						# first generate targets
						ops = []; feed = {}
						for j in range(0,n_ens):
							feed[NNs_freeze[j].input_] = replay_obs_1
							# feed[NNs[j].input_] = replay_obs_1
							ops.append(NNs_freeze[j].fc_out)
							# ops.append(NNs[j].fc_out)
						target_qs = sess.run(ops, feed_dict=feed)
						target_qs = np.array(target_qs)
						target_qs_mean = np.mean(target_qs,axis=0)

						# compare to live NN
						if False:
							if rep == 0:
								ops = []; feed = {}
								for j in range(0,n_ens):
									feed[NNs[j].input_] = replay_obs_1
									ops.append(NNs[j].fc_out)
								target_qs_live = sess.run(ops, feed_dict=feed)
								target_qs_live = np.array(target_qs_live)

								print('len(vars)',len(vars))
								print('\ntarget_qs\n',target_qs[0,0:4],'\n')
								print('\ntarget_qs_live\n',target_qs_live[0,0:4],'\n')
								sess.run(vars[0])


						# targets = (replay_rewards+1) + gamma * np.max(target_qs, axis=-1) # target is individ. NN q value
						# targets = (replay_rewards+1) + gamma * np.max(target_qs_mean, axis=-1) # target is ens q value
						targets = replay_rewards/10 + gamma * np.max(target_qs_mean, axis=-1) # target is ens q value

						# target_qs[:,episode_ends,:] = (0, 0)
						# targets = replay_rewards + gamma * np.max(target_qs, axis=-1)
						# targets = (replay_rewards-1) + gamma * np.max(target_qs, axis=2)

						# we should really have targets generated by a separate 'target' NN
						# only updated periodically

						# print('\n\ntarget_qs',target_qs)
						# print('targets',targets)

						# then train on targets
						ops = []; feed = {}
						for j in range(0,n_ens):
							feed[NNs[j].input_] = replay_obs
							# feed[NNs[j].q_target] = targets[j] # target is individ. NN q value
							feed[NNs[j].q_target] = targets # target is ens q value
							feed[NNs[j].actions_] = replay_actions
							feed[NNs[j].N_] = np.array([len(experience_replay)])
							np.hstack(replay_actions)

							ops.append(NNs[j].loss_)
							ops.append(NNs[j].train_opt)
						loss_and_blank = sess.run(ops, feed_dict=feed)
						l = np.mean(loss_and_blank[0::2])  # find mean loss
						# print('rep',rep,', loss',l)

					training_hist.append(batch_rewards_avg)

					# after training, find out our residuals - just use last batch

					# need Q value for (state, action) output by live NNs
					# compare to our target
					ops = []; feed = {}
					for j in range(0,n_ens):
						feed[NNs[j].input_] = replay_obs
						feed[NNs[j].actions_] = replay_actions
						ops.append(NNs[j].q)
					live_q = sess.run(ops, feed_dict=feed)
					live_q = np.array(live_q)
					live_q_mean = np.mean(live_q,axis=0)

					# compare with what we're aiming for - targets
					residuals_q = live_q_mean - targets

					if False:
						fig = plt.figure(figsize=(5, 4))
						ax = fig.add_subplot(111)
						ax.hist(residuals_q,bins=10)
						# ax.set_xlabel('Episode')
						# ax.set_ylabel('Average rewards')
						fig.show()

					print('residuals_q.mean(): ',np.round(residuals_q.mean(),4))
					print('residuals_q.std(): ',np.round(residuals_q.std(),4))

					# plot the reward surface learnt for one dimension and action
					if False:
						replay_obs_dummy = []
						replay_actions_dummy = []
						n_step = 100
						max_min = 1.
						action_dummy = [0,0,0,0,1.,0,0]
						for i in range(100):
							replay_obs_dummy.append([0.0,0.0,i*2*max_min/n_step-max_min])
							replay_actions_dummy.append(action_dummy)
							# replay_actions_dummy.append([0,0,0,0,0,1.,0])
						replay_obs_dummy = np.array(replay_obs_dummy)
						replay_actions_dummy = np.array(replay_actions_dummy)

						ops = []; feed = {}
						for j in range(0,n_ens):
							feed[NNs[j].input_] = replay_obs_dummy
							feed[NNs[j].actions_] = replay_actions_dummy
							ops.append(NNs[j].q)
						pred_q = sess.run(ops, feed_dict=feed)
						pred_q = np.array(pred_q)
						pred_q_mean = np.mean(pred_q,axis=0)
						pred_q_std = np.std(pred_q,axis=0)

						# related points in replay buffer
						experience_replay_arr = np.array(experience_replay)
						tol = 0.02
						relevant_buff=[]; relevant_input=[]
						while len(relevant_buff) < 1:
							for i in range(len(experience_replay)):
								obs_i = experience_replay[i][0]
								if obs_i[0] < tol and obs_i[0] > -tol:
									if obs_i[1] < tol and obs_i[1] > -tol:
										relevant_buff.append(obs_i)
										relevant_input.append(action_dummy)
							tol+=0.01
						print('relevant samples: ', len(relevant_buff))
						relevant_buff = np.array(relevant_buff)

						# see what would have predicted
						ops = []; feed = {}
						for j in range(0,n_ens):
							feed[NNs[j].input_] = relevant_buff
							feed[NNs[j].actions_] = relevant_input
							ops.append(NNs[j].q)
						dummy_pred_q = sess.run(ops, feed_dict=feed)
						dummy_pred_q = np.array(dummy_pred_q)
						dummy_pred_q_mean = np.mean(dummy_pred_q,axis=0)

						fig = plt.figure(figsize=(5, 4))
						ax = fig.add_subplot(111)
						ax.plot(replay_obs_dummy[:,-1],pred_q_mean + pred_q_std,'r:')
						ax.plot(replay_obs_dummy[:,-1],pred_q_mean - pred_q_std,'r:')
						ax.plot(replay_obs_dummy[:,-1],pred_q_mean,'k')
						# ax.scatter(relevant_buff[:,-1],np.zeros_like(relevant_buff[:,-1]))
						ax.scatter(relevant_buff[:,-1],dummy_pred_q_mean)
						ax.set_xlabel('x displacement')
						ax.set_ylabel('pred. Q value')
						ax.set_xlim([-max_min,max_min])
						fig.show()

					# training_hist.append(batch_rewards_avg)
					batch_states, batch_actions, batch_rewards = [],[],[]

				break # onto next episode
	results_train_hist.append(training_hist)
	# see what it's learnt
	# vis_current_state()
	if run != n_runs-1:
		print('closing sess')
		sess.close()

if is_save_results:
	import pickle
	name = str(datetime.datetime.now().strftime('%H-%M-%S')) + '_NNs' + str(n_ens) + 'noise' + str(round(move_noise,2)) + 'type' + what_reg
	name = '_try_2_' + model_type
	pickle.dump(results_train_hist, open('00_outputs_graph/rl_pendulum_data/results_train_' + name + '.p', "wb"))
	pickle.dump(training_hist, open('00_outputs_graph/rl_pendulum_data/training_hist' + name + '.p', "wb"))
	print('\n\n -- saved results ', name, '--\n')


is_compare = 0
if is_compare:
	# import results for comparison
	name='00_outputs_graph/rl_pendulum_data/' + 'results_train__try_2_periodic_times_tanh.p'
	results_train_hist_compare = pickle.load(open(name, 'rb'))
	results_train_hist_compare = np.array(results_train_hist_compare)
	N=10
	y_all_smooth_comp=[]
	for i in range(n_runs):
		y_all_smooth_comp.append(np.convolve(results_train_hist_compare[i], np.ones((N,))/N, mode='valid'))
	y_all_smooth_comp=np.array(y_all_smooth_comp)
	y_mean_comp_2 = np.mean(y_all_smooth_comp,axis=0)
	y_std_comp_2 = np.std(y_all_smooth_comp,axis=0)/np.sqrt(n_runs)

# training curve
results_train_hist=np.array(results_train_hist)
y_mean = np.mean(results_train_hist,axis=0)
y_std = np.std(results_train_hist,axis=0)/np.sqrt(n_runs)

# moving average
N=10
y_all_smooth=[]
for i in range(n_runs):
	y_all_smooth.append(np.convolve(results_train_hist[i], np.ones((N,))/N, mode='valid'))
y_all_smooth=np.array(y_all_smooth)

# x_s = np.arange(0,results_train_hist.shape[1])
# y_mean_smooth = np.convolve(y_mean, np.ones((N,))/N, mode='valid')
# y_std_smooth = np.convolve(y_std, np.ones((N,))/N, mode='valid')

y_mean_smooth = np.mean(y_all_smooth,axis=0)
y_std_smooth = np.std(y_all_smooth,axis=0)/np.sqrt(n_runs)

x_s = np.linspace(0,results_train_hist.shape[1],results_train_hist.shape[1])*n_eps_rollout
x_s_smooth = x_s[0:y_mean_smooth.shape[0]]
x_s_smooth = x_s_smooth/x_s_smooth.max() * x_s.max()

if model_type == 'theta_raw': col='g'; 
elif model_type == 'periodic': col='b'; 
elif model_type == 'periodic_times_tanh': col='r'; 
if col=='g':
	col_2 = 'palegreen'
elif col=='r':
	col_2 = 'mistyrose'
elif col=='b':
	col_2 = 'lightskyblue'

is_fill = 0; n_stds=1

fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(111)
if 0:
	for run in range(n_runs):
		# ax.plot(x_s,results_train_hist[run,:], 'k-', linewidth=0.2,label=u'run ' + str(run))
		ax.plot(x_s_smooth,y_all_smooth[run,:], 'k-', linewidth=0.2,label=u'run ' + str(run))

ax.plot(x_s_smooth, y_mean_smooth, col, linewidth=2.,label=u'ReLU')
# ax.plot(x_s, y_mean, col, linewidth=2.,label=str(n_ens) + u'xNNs')
if is_fill:
	# ax.plot(x_s_smooth, y_mean_smooth + n_stds * y_std_smooth, col, linewidth=0.3)
	# ax.plot(x_s_smooth, y_mean_smooth - n_stds * y_std_smooth, col, linewidth=0.3)
	ax.fill(np.concatenate([x_s_smooth, x_s_smooth[::-1]]),
					 np.concatenate([y_mean_smooth - n_stds * y_std_smooth,
									(y_mean_smooth + n_stds * y_std_smooth)[::-1]]),
					 alpha=1, fc=col_2, ec='None')

if 0:
	ax2 = ax.twinx()
	ax2.plot(x_s,np.cumsum(y_mean), 'k:', linewidth=2.,label=u'Regret')
	ax2.set_ylabel('Regret')

# ax.fill(np.concatenate([x_s, x_s[::-1]]),
# 		 np.concatenate([y_mean - 2 * y_std,
# 						(y_mean + 2 * y_std)[::-1]]),
# 		 alpha=1, fc='lightskyblue', ec='None')
is_compare=1
if is_compare:
	ax.plot(x_s_smooth, y_mean_comp_2, 'r-', linewidth=2.,label=u'Periodic x TanH')
	ax.plot(x_s_smooth, y_mean_comp_1, 'b-', linewidth=2.,label=u'Periodic')
	# ax.plot(x_s_smooth, np.convolve(y_mean_comp_2, np.ones((N,))/N, mode='valid'), 'b-', linewidth=2.,label=u'Periodic')
	# ax.plot(x_s_smooth, np.convolve(y_mean_comp_1, np.ones((N,))/N, mode='valid'), 'r-', linewidth=2.,label=u'Periodic x TanH')
	if is_fill:
		# ax.plot(x_s_smooth, y_mean_comp_1 + n_stds * y_std_comp_1, 'r-', linewidth=0.3)
		# ax.plot(x_s_smooth, y_mean_comp_1 - n_stds * y_std_comp_1, 'r-', linewidth=0.3)
		ax.fill(np.concatenate([x_s_smooth, x_s_smooth[::-1]]),
					 np.concatenate([y_mean_comp_2 - n_stds * y_std_comp_1,
									(y_mean_comp_2 + n_stds * y_std_comp_1)[::-1]]),
					 alpha=1, fc='mistyrose', ec='None')

		# ax.plot(x_s_smooth, y_mean_comp_2 + n_stds * y_std_comp_2, 'b-', linewidth=0.3)
		# ax.plot(x_s_smooth, y_mean_comp_2 - n_stds * y_std_comp_2, 'b-', linewidth=0.3)
		ax.fill(np.concatenate([x_s_smooth, x_s_smooth[::-1]]),
					 np.concatenate([y_mean_comp_1 - n_stds * y_std_comp_2,
									(y_mean_comp_1 + n_stds * y_std_comp_2)[::-1]]),
					 alpha=1, fc='lightskyblue', ec='None')


ax.set_xlim([0,x_s_smooth.max()])
ax.set_ylim([-8,-2])
ax.set_ylim([-7.5,-2.8])
# ax.set_ylim([-8,-2])
ax.set_xlabel('Episode')
ax.set_ylabel('Average Rewards')
ax.legend()
# plt.legend(loc='upper left')
title = str(datetime.datetime.now().strftime('%H-%M-%S')) + '-NNs' + str(n_ens) + 'noise' + str(round(move_noise,2)) + 'type' + what_reg + 'n-actions' + str(n_actions)
# ax.set_title(title)
fig.show()

is_save_graph=1
if is_save_graph:
	title = str(datetime.datetime.now().strftime('%H:%M:%S')) + '_NNs_' + str(n_ens) + '_modeltype_' + model_type + '_reg_' + what_reg
	fig.savefig('00_outputs_graph/rl_pendulum/train_' + title +'.eps', format='eps', dpi=1000, bbox_inches='tight')




# -- examine loss surface --
obs_grid = np.linspace(-np.pi*5,np.pi*5,800)
obs_grid = np.atleast_2d(obs_grid).T
obs_grid = np.concatenate([obs_grid,obs_grid],axis=-1)
obs_grid[:,-1] = 0.0 # angular vel theta_dot

grid_mus=[]; grid_stds=[]
for i in range(obs_grid.shape[0]):
	# evaluate distribution would have taken
	if is_return_raw_theta:
		obs_in = obs_grid[i]
	else:
		obs_in = np.array([np.cos(obs_grid[i]),np.sin(obs_grid[i]),obs_grid[-1]])
	action_pred_q_ens=[]
	for m in range(n_ens):
		action_pred_q_ens.append(NNs[m].predict(sess,obs_in))
	action_pred_q_mu = np.mean(action_pred_q_ens,axis=0)
	action_pred_q_std = np.std(action_pred_q_ens,axis=0)

	action_pred_q_mu = np.squeeze(action_pred_q_mu)
	action_pred_q_std = np.squeeze(action_pred_q_std)
	# action_pred_q_ens = np.array(action_pred_q_ens)
	# action_pred_q_ens = np.squeeze(action_pred_q_ens).T
	grid_mus.append(action_pred_q_mu)
	grid_stds.append(action_pred_q_std)

grid_mus=np.array(grid_mus)
grid_stds=np.array(grid_stds)

# advantage of moving one direction rather than the other
grid_mus_adv = grid_mus[:,0] - grid_mus[:,1]
grid_stds_adv = np.sqrt(np.square(grid_stds[:,0]) + np.square(grid_stds[:,1]))

if model_type == 'theta_raw': col='g'; 
elif model_type == 'periodic': col='b'; 
elif model_type == 'periodic_times_tanh': col='r'; 
if col=='g':
	col_2 = 'palegreen'
elif col=='r':
	col_2 = 'mistyrose'
elif col=='b':
	col_2 = 'lightskyblue'

# action dist figure
x_s = obs_grid[:,0]
action_i = 2 # which action to plot
fig = plt.figure(figsize=(6, 2))
ax = fig.add_subplot(111)
if 1: # plot one action
	ax.plot(x_s, grid_mus[:,action_i], col, linewidth=2.,label=u'Prediction')
	# ax.plot(x_s, grid_mus[:,action_i] + 2 * grid_stds[:,action_i], 'b', linewidth=0.5)
	# ax.plot(x_s, grid_mus[:,action_i] - 2 * grid_stds[:,action_i], 'b', linewidth=0.5)
	ax.plot(x_s, grid_mus[:,action_i] + 4 * grid_stds[:,action_i], col, linewidth=0.5)
	ax.plot(x_s, grid_mus[:,action_i] - 4 * grid_stds[:,action_i], col, linewidth=0.5)
	ax.fill(np.concatenate([x_s, x_s[::-1]]),
					 np.concatenate([grid_mus[:,action_i] - 4 * grid_stds[:,action_i],
									(grid_mus[:,action_i] + 4 * grid_stds[:,action_i])[::-1]]),
					 alpha=1, fc=col_2, ec='None')
	# ax.fill(np.concatenate([x_s, x_s[::-1]]),
	# 				 np.concatenate([grid_mus[:,action_i] - 2 * grid_stds[:,action_i],
	# 								(grid_mus[:,action_i] + 2 * grid_stds[:,action_i])[::-1]]),
	# 				 alpha=1, fc='deepskyblue', ec='None')
elif 0: # plot advantage of clockwise vs anticlockwise
	ax.plot(x_s, grid_mus_adv, 'b-', linewidth=1.,label=u'Prediction')
	ax.plot(x_s, grid_mus_adv + 2 * grid_stds_adv, 'b', linewidth=0.5)
	ax.plot(x_s, grid_mus_adv - 2 * grid_stds_adv, 'b', linewidth=0.5)
	ax.fill(np.concatenate([x_s, x_s[::-1]]),
				 np.concatenate([grid_mus_adv - 2 * grid_stds_adv,
								(grid_mus_adv + 2 * grid_stds_adv)[::-1]]),
				 alpha=1, fc='lightskyblue', ec='None')
else: # q values for all actions
	ax.plot(x_s, grid_mus[:,0], 'b-', linewidth=1.,label=u'+1 Torque')
	ax.plot(x_s, grid_mus[:,1], 'r-', linewidth=1.,label=u'-1 Torque')
	ax.plot(x_s, grid_mus[:,2], 'g-', linewidth=1.,label=u'0  Torque')
ax.set_ylim([-40,10]) # 4 actions
# ax.set_xlim([np.min(x_s),np.max(x_s)])
ax.set_xlim([-10,10])
ax.set_xlabel('Observed Angle')
# ax.set_yticklabels([])
# ax.set_xticklabels([-np.pi,0,np.pi])
# ax.set_yticks([])
ax.set_ylabel('Q-Value')
# ax.legend(loc='upper left') # 4 actions
# ax.legend(loc='upper right') # 7 actions
fig.show()

if is_save_graph:
	title = datetime.datetime.now().strftime('%m-%d-%H-%M-%S') + '_' + model_type
	fig.savefig('00_outputs_graph/rl_pendulum/q_surface' + title + '.eps', format='eps', dpi=1000, bbox_inches='tight')


if False:
	# quick single plot if cancelled training halfway
	fig = plt.figure(figsize=(5, 4))
	ax = fig.add_subplot(111)
	ax.plot(np.linspace(0,len(training_hist),len(training_hist))*n_eps_rollout,training_hist)
	ax.set_xlabel('Episode')
	ax.set_ylabel('Average rewards')
	fig.show()



# env.close()
# sess.close()
is_save_model=0
if is_save_model:
	saver = tf.train.Saver() 
	saver.save(sess, './00_outputs/00_tf_NNs/model_4_actions_anchored_2.ckpt')  
	
end_time = datetime.datetime.now()
total_time = end_time - start_time
print('seconds taken:', round(total_time.total_seconds(),1),
	'\nstart_time:', start_time.strftime('%H:%M:%S'), 
	'end_time:', end_time.strftime('%H:%M:%S'))





