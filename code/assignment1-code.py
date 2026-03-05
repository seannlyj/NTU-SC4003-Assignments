import numpy as np
import matplotlib.pyplot as plt

gamma = 0.99
p_intended = 0.8
p_side = 0.1

actions = ['U', 'D', 'L', 'R']
action_vectors = {
    'U': (-1,0),
    'D': (1,0),
    'L': (0,-1),
    'R': (0,1)
}

side_actions = {
    'U': ['L','R'],
    'D': ['L','R'],
    'L': ['U','D'],
    'R': ['U','D']
}

grid = [
    [1, None, 1, 0, 0, 1],
    [0,-1,0,1,None,-1],
    [0,0,-1,0,1,0],
    [0,0,0,-1,0,1],
    [0,None,None,None,-1,0],
    [0,0,0,0,0,0]
]

rows = 6
cols = 6

reward = np.zeros((rows,cols))

for r in range(rows):
    for c in range(cols):
        if grid[r][c] == None:
            reward[r,c] = np.nan
        elif grid[r][c] == 1:
            reward[r,c] = 1
        elif grid[r][c] == -1:
            reward[r,c] = -1
        else:
            reward[r,c] = -0.05

def valid(r,c):
    if r<0 or r>=rows or c<0 or c>=cols:
        return False
    if grid[r][c] is None:
        return False
    return True

def move(r,c,action):
    dr,dc = action_vectors[action]
    nr,nc = r+dr,c+dc
    if not valid(nr,nc):
        return r,c
    return nr,nc

def expected_utility(U,r,c,a):
    total = 0
    
    nr,nc = move(r,c,a)
    total += p_intended * U[nr,nc]
    
    for sa in side_actions[a]:
        nr,nc = move(r,c,sa)
        total += p_side * U[nr,nc]
        
    return total


# VALUE ITERATION
def value_iteration():
    
    # Initialize all U(s) = 0
    U = np.zeros((rows,cols))
    history = []
    
    # Iterate until convergence
    for iteration in range(200):
        U_new = U.copy()
        
        for r in range(rows):
            for c in range(cols):
                
                if grid[r][c] is None:
                    continue
                
                best = -1e9
                
                for a in actions:
                    val = expected_utility(U,r,c,a)     # expected utility of taking action a in state (r,c)
                    best = max(best,val)                # max(Summation(P(s'|s,a)*U(s')))
                
                U_new[r,c] = reward[r,c] + gamma*best   # U(s) = R(s) + gamma*max(Summation(P(s'|s,a)*U(s')))
        
        history.append(np.mean(U_new[~np.isnan(reward)]))
        
        # Check for convergence
        if np.max(np.abs(U_new-U)) < 1e-4:
            break
            
        U = U_new
        
    return U, history


def extract_policy(U):
    
    policy = [['' for _ in range(cols)] for _ in range(rows)]
    
    for r in range(rows):
        for c in range(cols):
            
            if grid[r][c] is None:
                policy[r][c] = 'W'
                continue
                
            best_action = None
            best_val = -1e9
            
            for a in actions:
                val = expected_utility(U,r,c,a)
                if val > best_val:
                    best_val = val
                    best_action = a
                    
            policy[r][c] = best_action
            
    return policy


# POLICY ITERATION
def policy_iteration():
    
    policy = np.random.choice(actions,(rows,cols))
    U = np.zeros((rows,cols))
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] is None:
                policy[r,c] = 'W'
    
    stable = False
    
    while not stable:
        
        # policy evaluation
        for _ in range(50):
            for r in range(rows):
                for c in range(cols):
                    
                    if grid[r][c] is None:
                        continue
                    
                    a = policy[r,c]
                    U[r,c] = reward[r,c] + gamma*expected_utility(U,r,c,a)  # U(s) = R(s) + gamma*[summation(P(s'|s,a)*U(s'))]
        
        
        stable = True
        
        # policy improvement
        for r in range(rows):
            for c in range(cols):
                
                if grid[r][c] is None:
                    continue
                
                old_action = policy[r,c]
                
                best_action = None
                best_val = -1e9
                
                for a in actions:
                    #Summation(P(s'|s,a)*U(s'))
                    val = expected_utility(U,r,c,a)
                    
                    # argmax(Summation(P(s'|s,a)*U(s')))
                    if val > best_val:
                        best_val = val
                        best_action = a
                
                policy[r,c] = best_action
                
                if best_action != old_action:
                    stable = False
                    
    return policy,U


U_vi, history = value_iteration()
policy_vi = extract_policy(U_vi)

policy_pi, U_pi = policy_iteration()

print("Utilities from Value Iteration")
print(np.round(U_vi,3))

print("\nOptimal Policy (Value Iteration)")
for row in policy_vi:
    print(row)

print("\nUtilities from Policy Iteration")
print(np.round(U_pi,3))

print("\nOptimal Policy (Policy Iteration)")
for row in policy_pi:
    print(row)


# Plot utility vs iterations

plt.plot(history)
plt.xlabel("Iterations")
plt.ylabel("Average Utility Estimate")
plt.title("Utility Estimates vs Iterations (Value Iteration)")
plt.show()