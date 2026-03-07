#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Agent import BaseAgent

class SarsaAgent(BaseAgent):
    

    def update(self,s,a,r,s_next,a_next,done):
        if not done:
            self.Q_sa[s,a] = self.Q_sa[s,a] + self.learning_rate*(r + self.gamma*self.Q_sa[s_next,a_next] - self.Q_sa[s,a] )       
        else:
             self.Q_sa[s,a] = self.Q_sa[s,a] + self.learning_rate*(r - self.Q_sa[s,a] )    
        
def sarsa(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of SARSA
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = SarsaAgent(env.n_states, env.n_actions, learning_rate, gamma)
    env.start_location
    eval_timesteps = []
    eval_returns = []
    
    # TO DO: Write your SARSA algorithm here!
    pi.evaluate(eval_env)
    count = 0
         
    while count < n_timesteps:
        s = env.reset()
        a = pi.select_action(s, policy, epsilon, temp)
        terminal = False
        while not terminal and count < n_timesteps:
            s_prime, r, done = env.step(a)
            a_prime = pi.select_action(s_prime, policy, epsilon, temp)
            pi.update(s,a,r,s_prime,a_prime,done)
            terminal = done
            count = count + 1
            s = s_prime
            a = a_prime
            #if plot:
            #    env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.01) # Plot the Q-value estimates during SARSA execution   
            if count%eval_interval == 0:
                eval_timesteps.append(count)
                eval_returns.append(pi.evaluate(eval_env))
        
        

    # if plot:
    #    env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during SARSA execution
    

    return np.array(eval_returns), np.array(eval_timesteps) 
    

def test():
    n_timesteps = 10000
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'softmax' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True
    eval_returns, eval_timesteps = sarsa(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)   
    
if __name__ == '__main__':
    test()
