import numpy as np
import numpy.random as rand
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import math

class MaxEnt:        
  def build_trans_mat_gridworld(self):
    # 5x5 gridworld laid out like: 
    # state 0 = (0,0)
    # state 1 = (0,1)
    # state 2 = (0,2)
    #.
    #.
    #.
    # 0  1  2  3  4
    # 5  6  7  8  9 
    # ...
    # 20 21 22 23 24
    # where 24 is a goal state that always transitions to a 
    # special zero-reward terminal state (25) with no available actions
    trans_mat = np.zeros((26,4,26))
    
    # NOTE: the following iterations only happen for states 0-23.
    # This means terminal state 25 has zero probability to transition to any state, 
    # even itself, making it terminal, and state 24 is handled specially below.
    
    # Action 0 = down
    for s in range(24):
      if s < 20:
        trans_mat[s,0,s+5] = 1
      else:
        trans_mat[s,0,s] = 1
        
    # Action 1 = up
    for s in range(24):
      if s >= 5:
        trans_mat[s,1,s-5] = 1
      else:
        trans_mat[s,1,s] = 1
        
    # Action 2 = left
    for s in range(24):
      if s%5 > 0:
        trans_mat[s,2,s-1] = 1
      else:
        trans_mat[s,2,s] = 1
        
   # Action 3 = right
    for s in range(24):
      if s%5 < 4:
        trans_mat[s,3,s+1] = 1
      else:
        trans_mat[s,3,s] = 1

    # Finally, goal state always goes to zero reward terminal state
    for a in range(4):
      trans_mat[24,a,25] = 1  
        
    return trans_mat


             
  def calcMaxEntPolicy(self, trans_mat, horizon, r_weights, state_features):
    """
    For a given reward function and horizon, calculate the MaxEnt policy that gives equal weight to equal reward trajectories
    
    trans_mat: an S x A x S' array of transition probabilites from state s to s' if action a is taken
    horizon: the finite time horizon (int) of the problem for calculating state frequencies
    r_weights: a size F array of the weights of the current reward function to evaluate
    state_features: an S x F array that lists F feature values for each state in S
    
    return: an S x A policy in which each entry is the probability of taking action a in state s
    """

    #BACKWARD PASS
    Z = np.zeros(np.shape(trans_mat)[0])
    #Z terminal = 1
    Z[len(Z) - 1] = float(1)
    #creates matrix of state action pairs
    Z_sa = np.zeros(np.shape(trans_mat)[0:2])
    #QUESTION: What are the initial Z_sk's?
    #loop for horizon steps
    for i in range(horizon):
      #go through every state
      for k in range(np.shape(Z_sa)[0]):
        #go through every action (at each state)
        for j in range(np.shape(Z_sa)[1]):
          #Question what are we summing across?
          #for every possible next state
          z_kj = 0
          for h in range(np.shape(trans_mat)[2]):
            #add transition probability
            z_kj = z_kj + trans_mat[k, j, h] * math.exp(np.dot(r_weights, state_features[k])) * Z[h]
          Z_sa[k, j] = z_kj
      #for every state
      for k in range(len(Z)):
        sum = 0
        #for every action at state k
        for j in range(np.shape(Z_sa)[1]):
          # print "zsa is " + str(Z_sa[k, j]) + " " + str(k) + " " + str(j)
          sum += Z_sa[k, j]
        #check is k is the terminal state (the last state)
        if (k == len(Z) - 1):
          sum += 1
        Z[k] = sum
    #LOCAL ACTION PROBABILITY COMPUTATION
    policy = np.zeros(np.shape(trans_mat)[0:2])
    for i in range(np.shape(policy)[0]):
      for k in range(np.shape(policy)[1]):
        policy[i, k] = Z_sa[i, k]/Z[i]
    return policy


    
  def calcExpectedStateFreq(self, trans_mat, horizon, start_dist, policy):
    """
    Given a MaxEnt policy, begin with the start state distribution and propagate forward to find the expected state frequencies over the horizon
    
    trans_mat: an S x A x S' array of transition probabilites from state s to s' if action a is taken
    horizon: the finite time horizon (int) of the problem for calculating state frequencies
    start_dist: a size S array of starting start probabilities - must sum to 1
    policy: an S x A array array of probabilities of taking action a when in state s
    
    return: a size S array of expected state visitation frequencies
    """
    #FORWARD PASS
    #Set the frequency to initial distribution
    D_st = np.zeros((np.shape(trans_mat)[0], horizon))
    #for every state
    for i in range(len(start_dist)):
      #for every time step
      for t in range(horizon):
        #initialize everything to start distribution
        D_st[i, t] = start_dist[i]
        if math.isnan(D_st[i, t]):
          print "the initial dist is nan here first"
          exit()
    # For time steps
    for t in range(horizon - 1):
    # For states 
      for k in range(len(start_dist)):
        #begin summing
        sum = 0
        #sum across states
        for i in range(len(start_dist)):
          #sum across actions
          for j in range(np.shape(trans_mat)[1]):
            #d_si,t * P_aij|s_i * P(q|s,a)
            sum += D_st[i, t] * policy[i, j] * trans_mat[i, j, k]
            np.nan_to_num(D_st)
        D_st[k, t + 1] = sum
    state_freq = np.zeros(len(start_dist))
    #for every state
    for i in range(len(start_dist)):
      sum = 0
      #for each time step for that state
      for t in range(horizon):
        np.nan_to_num(D_st)
        if math.isnan(D_st[i, t]):
          print "the initial dist is nan"
          # exit()
        sum += D_st[i, t]
        np.nan_to_num(sum)
        # if math.isnan(sum):
        #   print "sum is nan in calcExpectedStateFreq"
          # exit()
      state_freq[i] = sum
    return state_freq
    


  def maxEntIRL(self, trans_mat, state_features, demos, seed_weights, n_epochs, horizon, learning_rate):
    """
    Compute a MaxEnt reward function from demonstration trajectories
    
    trans_mat: an S x A x S' array that describes transition probabilites from state s to s' if action a is taken
    state_features: an S x F array that lists F feature values for each state in S
    demos: a list of lists containing D demos of varying lengths, where each demo is series of states (ints)
    seed_weights: a size F array of starting reward weights
    n_epochs: how many times (int) to perform gradient descent steps
    horizon: the finite time horizon (int) of the problem for calculating state frequencies
    learning_rate: a multiplicative factor (float) that determines gradient step size
    
    return: a size F array of reward weights
    """
    #QUESTION: What do we use as starting dist?
    avg_feature = self.get_avg_feature(demos, state_features, np.shape(state_features)[1])
    #Seeding with this start distribution because of demos, change to work for any demos
    start_dist = np.zeros(np.shape(trans_mat)[0])
    start_dist[0] = 1.0
    n_features = np.shape(state_features)[1]
    r_weights = seed_weights
    for n in range(n_epochs):
      policy = self.calcMaxEntPolicy(trans_mat, horizon, r_weights, state_features)
      # print policy
      state_freq = self.calcExpectedStateFreq(trans_mat, horizon, start_dist, policy)
      r_weights = self.gradient_descent(r_weights, state_freq, state_features, demos, learning_rate, avg_feature)
    return r_weights

  #computes the change in likelihood and increments the weights by change and returns 
  def gradient_descent(self, r_weights, state_freq, state_features, demos, learning_rate, avg_feature):
    fc_sum = np.zeros(np.shape(state_features)[1])
    for i in range(np.shape(state_features)[0]):
      fc_sum = np.add(fc_sum, state_freq[i] * state_features[i]) 
    gradient = np.subtract(avg_feature, fc_sum)
    weight_delta = learning_rate * gradient
    return np.add(r_weights, weight_delta)

  #computes f~
  def get_avg_feature(self, demos, state_features, num_features):
    m = len(demos)
    weight = 1/float(m)
    traj_features = np.zeros((len(demos), num_features))
    for i in range(len(demos)):
      for j in range(len(demos[i])):
        state = demos[i][j]
        traj_features[i] = np.add(traj_features[i], state_features[state])
    avg_feature = np.zeros(num_features)
    for i in range(m):
      avg_feature = np.add(avg_feature, weight * traj_features[i])
    if math.isnan(avg_feature[0]):
      print "we have a nan"
    return avg_feature

if __name__ == '__main__':
  
  maxent = MaxEnt()
  # Build domain, features, and demos
  trans_mat = maxent.build_trans_mat_gridworld()
  state_features = np.eye(26,25)  # Terminal state has no features, forcing zero reward 
  demos = [[0,1,2,3,4,9,14,19,24,25],[0,5,10,15,20,21,22,23,24,25],[0,5,6,11,12,17,18,23,24,25],[0,1,6,7,12,13,18,19,24,25]]
  seed_weights = np.zeros(25)
  
  # Parameters
  n_epochs = 100
  horizon = 10
  learning_rate = 1.6
  
  #print calcMaxEntPolicy(trans_mat, horizon, seed_weights, state_features)

  # Main algorithm call
  r_weights = maxent.maxEntIRL(trans_mat, state_features, demos, seed_weights, n_epochs, horizon, learning_rate)

  # Construct reward function from weights and state features
  reward_fxn = []
  for s_i in range(25):
    reward_fxn.append(np.dot(r_weights, state_features[s_i]))
  reward_fxn = np.reshape(reward_fxn, (5,5))
  #print reward_fxn
  # Plot reward function
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  X = np.arange(0, 5, 1)
  Y = np.arange(0, 5, 1)
  X, Y = np.meshgrid(X, Y)
  surf = ax.plot_surface(X, Y, reward_fxn, rstride=1, cstride=2, cmap=cm.coolwarm,
      linewidth=0, antialiased=False)
  fig.suptitle('Reward Function for Gridworld', fontsize=20)
  plt.xlabel('Row', fontsize=18)
  plt.ylabel('Column', fontsize=16)
  fig.savefig('reward.jpeg')
  plt.show()