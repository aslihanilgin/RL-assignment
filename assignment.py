# %% [markdown]
# # Assignment

# %%
# Import 

import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib
from degree_freedom_queen import *
from degree_freedom_king1 import *
from degree_freedom_king2 import *
from generate_game import *
from Chess_env import *


size_board = 4

# %% [markdown]
# ## The Environment
# 
# You can find the environment in the file Chess_env, which contains the class Chess_env. To define an object, you need to provide the board size considered as input. In our example, size_board=4. 
# Chess_env is composed by the following methods:
# 
# 1. Initialise_game. The method initialises an episode by placing the three pieces considered (Agent's king and queen, enemy's king) in the chess board. The outputs of the method are described below in order.
# 
#      S $\;$ A matrix representing the board locations filled with 4 numbers: 0, no piece in that position; 1, location of the 
#      agent's king; 2 location of the queen; 3 location of the enemy king.
#      
#      X $\;$ The features, that is the input to the neural network. See the assignment for more information regarding the            definition of the features adopted. To personalise this, go into the Features method of the class Chess_env() and change        accordingly.
#      
#      allowed_a $\;$ The allowed actions that the agent can make. The agent is moving a king, with a total number of 8                possible actions, and a queen, with a total number of $(board_{size}-1)\times 8$ actions. The total number of possible actions correspond      to the sum of the two, but not all actions are allowed in a given position (movements to locations outside the borders or      against chess rules). Thus, the variable allowed_a is a vector that is one (zero) for an action that the agent can (can't)      make. Be careful, apply the policy considered on the actions that are allowed only.
#      
# 
# 2. OneStep. The method performs a one step update of the system. Given as input the action selected by the agent, it updates the chess board by performing that action and the response of the enemy king (which is a random allowed action in the settings considered). The first three outputs are the same as for the Initialise_game method, but the variables are computed for the position reached after the update of the system. The fourth and fifth outputs are:
# 
#      R $\;$ The reward. To change this, look at the OneStep method of the class where the rewards are set.
#      
#      Done $\;$ A variable that is 1 if the episode has ended (checkmate or draw).
#      
#      
# 3. Features. Given the chessboard position, the method computes the features.
# 
# This information and a quick analysis of the class should be all you need to get going. The other functions that the class exploits are uncommented and constitute an example on how not to write a python code. You can take a look at them if you want, but it is not necessary.
# 
# 
# 
# 
# 

# %%
## INITIALISE THE ENVIRONMENT

env=Chess_Env(size_board)

# %%
# INITIALISE THE PARAMETERS OF YOUR NEURAL NETWORK AND...
# PLEASE CONSIDER TO USE A MASK OF ONE FOR THE ACTION MADE AND ZERO OTHERWISE IF YOU ARE NOT USING VANILLA GRADIENT DESCENT...
# WE SUGGEST A NETWORK WITH ONE HIDDEN LAYER WITH SIZE 200. 


S,X,allowed_a=env.Initialise_game()

N_a=np.shape(allowed_a)[0]   # TOTAL NUMBER OF POSSIBLE ACTIONS -> output
n_input_layer=np.shape(X)[0]    ## INPUT SIZE -> features
n_hidden_layer=200                ## NUMBER OF HIDDEN NODES

## INITALISE YOUR NEURAL NETWORK...

############## for one hidden layer
W1 = np.random.uniform(0,1,(n_hidden_layer, n_input_layer))
W2 = np.random.uniform(0,1,(N_a, n_hidden_layer))

# The following normalises the random weights so that the sum of each row =1
W1 = np.divide(W1,np.matlib.repmat(np.sum(W1,1)[:,None],1,n_input_layer))
W2 = np.divide(W2,np.matlib.repmat(np.sum(W2,1)[:,None],1,n_hidden_layer))


# Initialize the biases

bias_W1 = np.zeros((n_hidden_layer,))
bias_W2 = np.zeros((N_a,))

##############

# HYPERPARAMETERS SUGGESTED (FOR A GRID SIZE OF 4)

epsilon_0 = 0.2     # STARTING VALUE OF EPSILON FOR THE EPSILON-GREEDY POLICY
beta = 0.00005      # THE PARAMETER SETS HOW QUICKLY THE VALUE OF EPSILON IS DECAYING (SEE epsilon_f BELOW)
gamma = 0.85        # THE DISCOUNT FACTOR
eta = 0.0035        # THE LEARNING RATE

N_episodes = 100000 # THE NUMBER OF GAMES TO BE PLAYED 
# N_episodes = 10000 # THE NUMBER OF GAMES TO BE PLAYED 

# SAVING VARIABLES
R_save = np.zeros([N_episodes, 1]) # rewards
N_moves_save = np.zeros([N_episodes, 1])


# Keep track of the network inputs and average error per epoch
errors = np.zeros((N_episodes,))


# %%
## DEFINE THE EPSILON-GREEDY POLICY
# exploration vs exploitation 
def EpsilonGreedy_Policy(Qvalues:np.array, epsilon:float):
    
    N_a=np.shape(Qvalues)[0]
    rand_value=np.random.uniform(0,1)
    rand_a=rand_value<epsilon

    if rand_a==True: # take random action (exploration)
        a=np.random.randint(0,N_a)
    
    else: # exploitation
        a=np.argmax(Qvalues)
        
    return a

def sigmoid_activation(h:np.array):
    return 1/(1+np.exp(-h))


# %%
# TRAINING LOOP BONE STRUCTURE

for n in range(N_episodes):


    # debug
    print("----start of the episode")
    

    Qvals = np.zeros([np.shape(allowed_a)[0], 1]) # initiliase a Q values array

    epsilon_f = epsilon_0 / (1 + beta * n) # decaying epsilon
    Done = 0 # beginning of episode
    i = 1 # counter for number of actions per episode
    S, X, allowed_a = env.Initialise_game() # initialise game

    if i == 1: # if it's the first episode
        
        # NN forward prop
        h1 = np.dot(W1, X) + bias_W1 
        x1 = sigmoid_activation(h1)
        h2 = np.dot(W2, x1) + bias_W2
        Qvals = sigmoid_activation(h2)

    # SARSA: choose action from state using policy 
    poss_actions_indices, _ = np.where(allowed_a==1)
    poss_action_Qvals = np.copy(Qvals[poss_actions_indices])
    chosen_action_index = EpsilonGreedy_Policy(poss_action_Qvals, epsilon_f) # returns index of chosen action
    chosen_action = poss_actions_indices[chosen_action_index]
    chosen_action_Qval = Qvals[chosen_action] # get the Q value of action

    # play the game move by move
    while Done == 0: 
        # Initialise the gradients for each batch (game)
        dW1 = np.zeros(W1.shape)
        dW2 = np.zeros(W2.shape)

        dbias_W1 = np.zeros(bias_W1.shape)
        dbias_W2 = np.zeros(bias_W2.shape)

    
        S_next, X_next, allowed_a_next, R_next, Done = env.OneStep(chosen_action) # take action


        if Done == 1: # game has ended

            # debug
            print("game has ended.")
            
            # After each batch (game) update the weights using accumulated gradients
            W2 += eta*dW2/i
            W1 += eta*dW1/i

            bias_W1 += eta*dbias_W1/i
            bias_W2 += eta*dbias_W2/i

            # average reward and number of moves
            R_save[n]=np.copy(R_next)
            avg_reward = np.mean(R_save)

            N_moves_save[n]=np.copy(i)
            no_of_episodes = np.mean(N_moves_save)

            # debug
            print("avg_reward: {}, no of episodes: {}".format(avg_reward, no_of_episodes))

            # TODO: where did this come from???
            # Reward is the q-value of the final action. ???

            break


        else: # game is not over yet

            # Get the chosen step
            # NN forward prop
            h1 = np.dot(W1, X_next) + bias_W1 
            x1 = sigmoid_activation(h1)
            h2 = np.dot(W2, x1) + bias_W2
            Qvals = sigmoid_activation(h2)

            # SARSA: choose action' from state' using policy 
            poss_actions_indices, _ = np.where(allowed_a_next==1)
            poss_action_Qvals = np.copy(Qvals[poss_actions_indices])
            chosen_action_index = EpsilonGreedy_Policy(poss_action_Qvals, epsilon_f) # returns index of chosen action
            chosen_action_prime = poss_actions_indices[chosen_action_index]
            chosen_action_prime_Qval = Qvals[chosen_action_prime] # get the Q value of action'

            # backprop
            target_Qvals = Qvals.copy()
            np.put(target_Qvals, [chosen_action_index], [eta * (R_next + (gamma * chosen_action_prime_Qval) - chosen_action_Qval)])
            
            e_n = target_Qvals - Qvals # Compute the error signal
            
            # Backpropagation: output layer -> hidden layer
            delta2 = Qvals*(1-Qvals) * e_n
                
            dW2 += np.outer(delta2, x1)
            dbias_W2 += delta2

            # Backpropagation: hidden layer -> input layer
            delta1 = x1*(1-x1) * np.dot(W2.T, delta2)
            dW1 += np.outer(delta1,X_next)
            dbias_W1 += delta1

            # Store the error per epoch (step)
            errors[i] = errors[i] + 0.5*np.sum(np.square(e_n))/i 


        # next state becomes current state
        S=np.copy(S_next)
        X=np.copy(X_next)
        allowed_a=np.copy(allowed_a_next)
        chosen_action = chosen_action_prime
        
        i += 1  # UPDATE COUNTER FOR NUMBER OF ACTIONS


