from typing import Callable
import numpy as np
from itertools import count

#parameters
wdt = 12
hgt = 6
n_dir = 2
n_dim = 2
goal = np.array([3,3])
np.set_printoptions(precision=2)


def e_greedy(state: np.array, value: np.array, past: np.array = None) -> np.array:

    state_value = value[state[0],state[1],:,:]
    ind = np.unravel_index(np.argmax(state_value),state_value.shape)
    act_max = np.array(ind)
    coin = np.random.uniform(0,1)
    if coin < 0.95:
        return act_max
    else:
        return np.random.randint((0,0),(2,2),size=(2,))

def e_greedy_incentive(state: np.array, value: np.array, past: np.array):
    
    k = 0.3
    state_value = value[state[0],state[1],:,:]
    modifier = k*np.sqrt(past[state[0],state[1],:,:])
    ind = np.unravel_index(np.argmax(state_value+modifier),state_value.shape)
    act_max = np.array(ind)
    coin = np.random.uniform(0,1)
    if coin < 0.95:
        return act_max
    else:
        return np.random.randint((0,0),(2,2),size=(2,))


def reward(state: np.array) -> int:
    
    global goal
    if (state == goal).all(): return 0
    else: return -1

def update(state,action):

    global hgt,wgt,goal

    n_state = state+action
    if (n_state < np.array([0,0])).any() or\
        (n_state >= np.array([hgt,wdt])).any():
        n_state = np.random.randint((0,0),(hgt,wdt),size=(2,),dtype=np.int32)
    
    return n_state


def Q(alpha: float,
      gamma: float,
      tol: float,
      policy: Callable
) -> np.array:
    
    global goal,hgt,wdt,n_dir

    action_value = np.zeros(shape=(hgt,wdt,n_dir,n_dir))
    past = np.zeros(shape=(hgt,wdt,n_dir,n_dir))
    times = []

    for k in count():

        state = np.random.randint((0,0),(hgt,wdt),size=(2,),dtype=np.int32)
        if (state == goal).all():
            state += np.array([1,1])
        old_value = action_value.copy()
        
        for t in count():
            
            act = policy(state,action_value,past)
            past[state[0],state[1],act[0],act[1]] = 0
            r = reward(state)
            nstate = update(state,act)
            print(nstate)
            print(r)
            print(act)

            action_value[state[0],state[1],act[0],act[1]] += alpha*(r+
                gamma*np.max(action_value[nstate[0],nstate[1],:,:])-
                action_value[state[0],state[1],act[0],act[1]])
            state = nstate

            past += 1
            
            if (goal == state).all():
                break
        
        times.append(t)
        if (old_value-tol<action_value).all() and\
            (action_value<old_value+tol).all():
            break
 
    return np.array(times), action_value.mean(axis=(2,3))



