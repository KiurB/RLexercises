import numpy as np
from itertools import count

#parameters
wdt = 12
hgt = 6
n_dir = 2
n_dim = 2
goal = np.array([3,3])
np.set_printoptions(precision=2)


def behavior(state: np.array) -> np.array:

    act = np.zeros((2,),dtype=np.int32)
    act[1] = np.random.randint(0,2)

    return act


def target() -> np.array:

    return np.array([0,1])


def reward(state: np.array) -> int:
    
    global goal
    if (state == goal).all(): return 0
    else: return -1


def importance(action: np.array) -> int:

    if (action == target()).all(): return 2
    else: return 0


def update(state,action):

    global hgt,wgt,goal

    n_state = state+action
    if (n_state < np.array([0,0])).any() or\
        (n_state >= np.array([hgt,wdt])).any():
        n_state = np.random.randint((0,0),(hgt,wdt),size=(1,2),dtype=np.int32)
    
    return n_state


def sarsa_n(alpha: float,
            n: int,
            gamma: float,
            tol: float
) -> np.array:
    
    global goal,hgt,wdt,n_dir

    action_value = np.zeros(shape=(hgt,wdt,n_dir,n_dir))
    times = []

    for k in count():

        rewards = np.array([0])
        states = np.random.randint((0,0),(hgt,wdt),size=(1,2),dtype=np.int32)
        state = states[0]
        actions = np.array([behavior(state)],dtype=np.int32)
        act = actions[0]
        T = 10000000
        
        old_value = action_value.copy()

        for t in count():
            
            if t<T:
                
                state = update(state,act)
                states = np.vstack([states,state])
                r = reward(state)
                rewards = np.hstack([rewards,r])
                
                if (state == goal).all(): T = t+1
                else:
                    act = behavior(state)
                    actions = np.vstack([actions,act])
            tau = t-n+1
            if tau >= 0:
                rho = 1
                for i in range(tau+1,np.min((tau+n-1,T-1))+1):
                    rho *= importance(actions[i])
                G = 0
                for i in range(tau+1,np.min((tau+n,T))+1):
                    G += gamma**(i-tau-1)*rewards[i]
                if tau+n<T: G += gamma**n*action_value[states[tau+n,0],
                                                       states[tau+n,1],
                                                       actions[tau+n,0],
                                                       actions[tau+n,1]]
                G = round(G,4)
                action_value[states[tau,0],states[tau,1],actions[tau,0],actions[tau,1]] +=\
                    alpha*rho*(G-action_value[states[tau,0],states[tau,1],actions[tau,0],actions[tau,1]])
            if tau == T-1:
                break

        times.append(t)
        if (old_value-tol<action_value).all() and\
            (action_value<old_value+tol).all():
            break

    return action_value, sum(times)


def mod_sarsa_n(alpha: float,
            n: int,
            gamma: float,
            tol: float
) -> np.array:
    
    global goal,hgt,wdt,n_dir

    action_value = np.zeros(shape=(hgt,wdt,n_dir,n_dir))
    times = []

    for k in count():

        rewards = np.array([0])
        states = np.random.randint((0,0),(hgt,wdt),size=(1,2),dtype=np.int32)
        state = states[0]
        actions = np.array([behavior(state)],dtype=np.int32)
        act = actions[0]
        T = 10000000
        
        old_value = action_value.copy()

        for t in count():
            
            if t<T:
                
                state = update(state,act)
                states = np.vstack([states,state])
                r = reward(state)
                rewards = np.hstack([rewards,r])
                
                if (state == goal).all(): T = t+1
                else:
                    act = behavior(state)
                    actions = np.vstack([actions,act])
            tau = t-n+1
            if tau >= 0:
                values = action_value[:,:,0,1]
                print(values)
                if tau+n < T: G = action_value[states[tau+n,0],states[tau+n,1],actions[tau+n,0],actions[tau+n,1]]
                else: G = r
                for j in range(min(tau+n-1,T-1),tau,-1):
                    value = values[states[j][0],states[j][1]]
                    av = action_value[states[j,0],states[j,1],actions[j,0],actions[j,1]]
                    av = round(av,3)
                    G = rewards[j]+gamma*importance(actions[j])*(G-av)+gamma*value
                    G = round(G,3)
                print("End: " + str(G))
                action_value[states[tau,0],states[tau,1],actions[tau,0],actions[tau,1]] +=\
                    alpha*(G-action_value[states[tau,0],states[tau,1],actions[tau,0],actions[tau,1]])
                action_value = np.round(action_value,3)
            if tau == T-1:
                break

        times.append(t)
        if (old_value-tol<action_value).all() and\
            (action_value<old_value+tol).all():
            break

    return action_value, sum(times)


#honestly, both these methods have an enormous variance in the updates, but the second one is far better and it converges how it should. 
