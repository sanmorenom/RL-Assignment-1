#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import argmax

class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        
    def select_action(self,s):
        ''' Returns the greedy best action in state s ''' 
        # TO DO: Add own code
        a = argmax(self.Q_sa[s]) 
        return a
        
    def update(self,s,a,p_sas,r_sas):
        ''' Function updates Q(s,a) using p_sas and r_sas '''
        qu = 0
        for state in range(self.n_states):
            qu += p_sas[s,a,state] * (r_sas[s,a,state]+self.gamma*np.max(self.Q_sa[state]))
        self.Q_sa[s,a] = qu
        pass
    
    
def Q_value_iteration(env, gamma=1.0, threshold=0.001):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''
    
    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)
 
    delta = 1.0
    i = 0
    while delta > threshold:
        delta = 0
        for s in range(QIagent.n_states):
            for a in range(QIagent.n_actions):
                q = QIagent.Q_sa[s,a]
                QIagent.update(s,a,env.p_sas,env.r_sas)
                delta = max(delta, abs(q-QIagent.Q_sa[s,a]))
        i +=1
        #env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=10)
        print("Q-value iteration, iteration {}, max error {}".format(i,delta))
    
        
    # Plot current Q-value estimates & print max error
    #env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.)
    #print("Q-value iteration, iteration {}, max error {}".format(i,max_error))
 
    return QIagent

def experiment():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    env.render()
    QIagent = Q_value_iteration(env,gamma,threshold)
    
    # view optimal policy
    done = False
    s = env.reset()
    reward = []
    while not done:
        a = QIagent.select_action(s)
        s_next, r, done = env.step(a)
        reward.append(r)
        env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.1)
        s = s_next
    print(f'Final State vlue (V*(s)) for initial state: {QIagent.Q_sa[3,3]}')

    # TO DO: Compute mean reward per timestep under the optimal policy
    
    print(f"Mean reward per timestep under optimal policy: {np.mean(reward)} \n final reward: {np.sum(reward)}")
    
if __name__ == '__main__':
    experiment()
