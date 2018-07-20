import numpy as np
import pickle
import gym

def preprocess_obs(input_obs, prev_processed_obs, input_dim):

	processed_obs = input_obs[35 : 195]
	processed_obs = processed_obs[::2, ::2, 0]
	processed_obs[processed_obs == 144] = 0
	processed_obs[processed_obs == 109] = 0
	processed_obs[processed_obs != 0] = 1
	processed_obs = processed_obs.astype(np.float).ravel()

	if prev_processed_obs is not None:
		input_obs = processed_obs - prev_processed_obs
	else:
		input_obs = np.zeros(input_dim)

	prev_processed_obs = processed_obs

	return input_obs, prev_processed_obs

def relu(mat):
	mat[mat < 0] = 0
	return mat

def sigmoid(z):
	return 1.0/(1.0 + np.exp(-z))

def apply_nn(obs_matrix, weights):
	hidden_layer_values = np.dot(weights['1'], obs_matrix)
	hidden_layer_values = relu(hidden_layer_values)
	output = np.dot(weights['2'], hidden_layer_values)
	output = sigmoid(output)
	return hidden_layer_values, output

def choice(p):
	if np.random.uniform() < p:
		return 2 # 2 = UP
	else:
		return 3 # 3 = DOWN

def disc_with_rewards(gradient_log_p, epi_rewards, gamma):
	disc_rewards = np.zeros_like(epi_rewards)
	running_add = 0
	for t in reversed(range(0, epi_rewards.size)):
		if epi_rewards[t] != 0:
			running_add = 0
		running_add = running_add * gamma + epi_rewards[t]
		disc_rewards[t] = running_add
	
	disc_rewards -= np.mean(disc_rewards)
	disc_rewards /= np.std(disc_rewards)
	return gradient_log_p * disc_rewards

def compute_gradient(gradient_log_p, hidden_layer_values, obs_values, weights):

	# backbackbackback 
	delta_l = gradient_log_p
	dC_dw2 = np.dot(hidden_layer_values.T, delta_l).ravel()

	delta_l2 = np.outer(delta_l, weights['2'])
	delta_l2 = relu(delta_l2)
	dC_dw1 = np.dot(delta_l2.T, obs_values)

	return {
		'1': dC_dw1,
		'2': dC_dw2
	}

def update_weights(weights, expectation_g_squared, g_dict, decay_rate, learning_rate):
	#   check out rmsprop
	epsilon = 1e-5
	for layer_name in weights.keys():
		g =  g_dict[layer_name]
		expectation_g_squared[layer_name] = decay_rate * expectation_g_squared[layer_name] + (1 - decay_rate) * g**2
		weights[layer_name] += (learning_rate * g)/(np.sqrt(expectation_g_squared[layer_name]) + epsilon)
		g_dict[layer_name] = np.zeros_like(weights[layer_name]) 

def main():
	resume = True

	env = gym.make("Pong-v0")
	obs = env.reset()
	
	batch_size = 10
	gamma = 0.99
	decay_rate = 0.99
	n_hidden_layer = 200
	input_dim = 80 * 80
	learning_rate = 1e-3
	epsilon = 1e-5
	epi_n = 0
	reward_sum = 0
	running_rewards = None
	prev_processed_obs = None

	weights = {
		'1' : np.random.randn(n_hidden_layer, input_dim) / np.sqrt(input_dim),
		'2' : np.random.randn(n_hidden_layer) / np.sqrt(n_hidden_layer)
	}

	if resume == True:
		weights = pickle.load(open('model2.p', 'rb'))
	#rmsprop
	expectation_g_squared = {}
	g_dict = {}
	for layer_name in weights.keys():
		expectation_g_squared[layer_name] = np.zeros_like(weights[layer_name])
		g_dict[layer_name] = np.zeros_like(weights[layer_name])

	epi_hidden_layer_values, epi_obs, epi_gradient_log_ps, epi_rewards = [], [], [], []
	
	
	while True:
		env.render()
		processed_obs, prev_processed_obs = preprocess_obs(obs, prev_processed_obs, input_dim)
		hidden_layer_values, up_probablity = apply_nn(processed_obs, weights)
		
		if epi_obs == []:
			epi_obs = processed_obs
		else:
			epi_obs = np.vstack((epi_obs, processed_obs))
		
		if epi_hidden_layer_values == []:
			epi_hidden_layer_values = hidden_layer_values
		else:
			epi_hidden_layer_values = np.vstack((epi_hidden_layer_values, hidden_layer_values))

		action = choice(up_probablity)

		
		fake_label = 1 if action == 2 else 0

		loss_func_gradient = fake_label - up_probablity
		
		if epi_gradient_log_ps == []:
			epi_gradient_log_ps = loss_func_gradient
		else:
			epi_gradient_log_ps = np.vstack((epi_gradient_log_ps, loss_func_gradient))

		# done ~ termination
		obs, reward, done, info = env.step(action)

		reward_sum += reward
		if epi_rewards == []:
			epi_rewards = reward
		else:
			epi_rewards = np.vstack((epi_rewards, reward))

		#have a look at this, don't understand it fully


		if done:
			epi_n = epi_n + 1
  
			# epi_hidden_layer_values = np.vstack(epi_hidden_layer_values)
			# epi_obs = np.vstack(epi_obs)
			# epi_gradient_log_ps = np.vstack(epi_gradient_log_ps)
			# epi_rewards = np.vstack(epi_rewards)

			epi_gradient_log_ps_disc = disc_with_rewards(epi_gradient_log_ps, epi_rewards, gamma)

			gradient = compute_gradient(
				epi_gradient_log_ps_disc,
				epi_hidden_layer_values,
				epi_obs,
				weights
			)

			for layer_name in gradient:
				g_dict[layer_name] += gradient[layer_name]

			if epi_n % batch_size == 0:
				for layer_name in weights.keys():
					g =  g_dict[layer_name]
					expectation_g_squared[layer_name] = decay_rate * expectation_g_squared[layer_name] + (1 - decay_rate) * g**2
					weights[layer_name] += (learning_rate * g)/(np.sqrt(expectation_g_squared[layer_name]) + epsilon)
					g_dict[layer_name] = np.zeros_like(weights[layer_name]) 

			epi_hidden_layer_values, epi_obs, epi_gradient_log_ps, epi_rewards = [], [], [], [] 
			obs = env.reset() # reset env
			# read about thisk
			running_rewards = reward_sum if running_rewards is None else running_rewards * 0.99 + reward_sum * 0.01
			if epi_n % 10 == 0:
				print ("resetting env. episode reward total was ", reward_sum, ". running mean: ", running_rewards,"epi_n: ", epi_n)
			reward_sum = 0
			prev_processed_obs = None

			if epi_n % 100 == 0: pickle.dump(weights, open('model2.p', 'wb'))
			
main()
