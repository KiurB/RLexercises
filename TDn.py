# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from typing import Callable
from itertools import count


#parameters
hgt = 15
wgt = 12
act_n = 3
np.set_printoptions(precision=2)

terminal_state = np.array([4,5])

#simulation
def left_policy(state):

    return np.array([0,-1])

def update(state,action):
    
    global hgt,wgt 
    global terminal_state
    
    n_state = state+action
    if (n_state < np.array([0,0])).any() or\
        (n_state >= np.array([hgt,wgt])).any():
            n_state = np.random.randint((0,0),(hgt,wgt),size=(1,2))
    if (n_state == terminal_state).all(): reward = 0
    else: reward = -1
    
    return n_state, reward
    


def n_step_TD(policy: Callable = left_policy,
              update: Callable = update,
              alpha: float = 0.5,
              gamma: float = 0.95,
              n: int = 5):
    
    global terminal_state
    global hgt,wgt
    
    value = np.random.uniform(-1,1,size=(hgt,wgt))
    tol = 3e-7
    times = []
    
    for k in count():
        
        rewards = np.array([[0]])
        states = np.array([[]],dtype=np.int32)
        
        state = np.random.randint((0,0),(hgt,wgt),size=(1,2))
        states = np.c_[states,state]
        T = 1000000
        old_value = value.copy()
        
        for t in count():
            
            if t<T:
                act = policy(state)
                state,r = update(state,act)
                rewards = np.append(rewards, r)
                states = np.r_[states,state]
                if (state == terminal_state).all(): T = t+1
            tau = t-n+1
            if tau >= 0:
                G = 0
                for i in range(tau+1,np.min((tau+n,T))+1):
                    G += gamma**(i-tau-1)*rewards[i]
                if tau+n < T: G += gamma**n*value[states[tau+n,0],
                                                 states[tau+n,1]]
                G = round(G,8)
                value[states[tau,0],states[tau,1]] +=\
                    alpha*(G-value[states[tau,0],states[tau,1]])
                value = np.round(value,8)
            if tau == T-1:
                break
        
        times.append(T)
        if (old_value-tol < value).all() and\
            (value < old_value+tol).all():
            break
        
    return value, sum(times)


def mod_n_step_TD(policy: Callable = left_policy,
              update: Callable = update,
              alpha: float = 0.5,
              gamma: float = 0.95,
              n: int = 5):
    
    global terminal_state,hgt,wgt
    
    value = np.random.uniform(-1,1,size=(hgt,wgt))
    tol = 3e-7
    times = []
    
    for k in count():
        
        rewards = np.array([[0]])
        states = np.array([[]],dtype=np.int32)
        
        state = np.random.randint((0,0),(hgt,wgt),size=(1,2))
        states = np.c_[states,state]
        T = 1000000
        old_value = value.copy()
        td_error = 0
        cum_gamma = 1
        
        for t in count():
            
            if t<T:
                act = policy(state)
                state,r = update(state,act)
                rewards = np.append(rewards, r)
                states = np.r_[states,state]
                if (state == terminal_state).all(): T = t+1
                td_error += r+gamma*value[states[-1][0],states[-1][1]]-\
                    value[states[-2][0],states[-2][1]]
                td_error *= cum_gamma
                td_error = np.round(td_error,8)
                cum_gamma *= gamma
            tau = t-n+1
            if t!=0 and t%n == 0:
                value[states[tau,0],states[tau,1]] +=\
                    alpha*td_error
                value = np.round(value,8)
                td_error = 0
                cum_gamma = 1
            if tau == T-1:
                break
        
        times.append(T)
        if (old_value-tol < value).all() and\
            (value < old_value+tol).all():
            break
        
    return value, sum(times)
            
            
#convergence is way faster and better too, requires less effort on
#initialization...
        
        
    
    







