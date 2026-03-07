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

class NstepQLearningAgent(BaseAgent):
        
    def update(self, states, actions, rewards, done, n):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        # TO DO: Add own code

        T_ep = len(rewards)
    
        for t in range(T_ep):
            m = min(n, T_ep - t)
            
            Gt = 0
            for i in range(m):
                Gt += self.gamma**i * rewards[t + i]
            
            # Terminal condistion
            if(t + m == T_ep and done):
                is_terminal = True
            else:
                is_terminal = False

            if not is_terminal:
                Gt += (self.gamma**m) * np.max(self.Q_sa[states[t+m], :])

            Q = self.Q_sa[states[t], actions[t]]
            self.Q_sa[states[t], actions[t]] = Q + self.learning_rate * (Gt - Q)

def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, n=5, eval_interval=500):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    # TO DO: Write your n-step Q-learning algorithm here!    
    
    budget = 0
    while budget < n_timesteps:
        
        s = [env.reset()]
        a = []
        r = []

        eps_done = False
        
        for t in range(max_episode_length):
            
            if(budget % eval_interval == 0):
                eval_timesteps.append(budget)
                eval_returns.append(pi.evaluate(eval_env))
            
            
            action = pi.select_action(s= s[t], policy=policy, epsilon=epsilon, temp= temp)
            s_next, reward, done = env.step(action) 
           
            s.append(s_next)
            a.append(action)
            r.append(reward)
            
            budget += 1
            
            if done:
                eps_done = True
                break
            if budget >= n_timesteps:
                break
            
        pi.update(s, a, r, eps_done, n)
        
        if budget >= n_timesteps:
            break 
        
        
    eval_timesteps.append(n_timesteps)
    eval_returns.append(pi.evaluate(eval_env))
        
        
        
    # if plot:
    #    env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during n-step Q-learning execution
        
    return np.array(eval_returns), np.array(eval_timesteps) 

def test():
    n_timesteps = 10000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1
    n = 5
    
    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True
    n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, n=n)
    
    
if __name__ == '__main__':
    test()
