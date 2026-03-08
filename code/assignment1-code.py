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

start_state = (3, 2)


# HELPER FUNCTIONS
# check if (r,c) is inside grid and not a wall
def valid(r, c):
    if r < 0 or r >= rows or c < 0 or c >= cols:
        return False
    return grid[r][c] is not None

# return new coordinates after attempting action, stays in place if move is invalid
def move(r, c, action):
    dr, dc = action_vectors[action]
    nr, nc = r + dr, c + dc
    if valid(nr, nc):
        return nr, nc
    return r, c

# compute sum_{s'} P(s'|s,a) * U(s') for state (r,c) and action a.
def expected_utility(U, r, c, a):
    total = 0
    # Intended direction
    nr, nc = move(r, c, a)
    total += p_intended * U[nr, nc]
    # Two perpendicular directions
    for sa in side_actions[a]:
        nr, nc = move(r, c, sa)
        total += p_side * U[nr, nc]
    return total

# VALUE ITERATION
def value_iteration():
    
    # Initialize all U(s) = 0
    U = np.zeros((rows,cols))
    history_avg = []      # average utility over non-wall states
    history_start = []    # utility of start state
    
    # Iterate until convergence
    for iteration in range(5000):
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
        
        # Record metrics
        history_avg.append(np.mean(U_new[~np.isnan(reward)]))
        history_start.append(U_new[start_state])
        
        # Check for convergence
        delta = np.max(np.abs(U_new - U))
        if delta < 1e-4:
            break
            
        U = U_new
        
    return U, history_avg, history_start

# POLICY ITERATION
def policy_iteration():
    # Initialize random policy
    policy = np.random.choice(actions,(rows,cols))
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] is None:
                policy[r,c] = 'W'

    U = np.zeros((rows, cols))
    history_avg = []
    history_start = []
    stable = False

    while not stable:

        # POLICY EVALUATION
        while True:

            U_new = U.copy()

            for r in range(rows):
                for c in range(cols):

                    if grid[r][c] is None:
                        continue

                    a = policy[r,c]
                    U_new[r,c] = reward[r,c] + gamma * expected_utility(U,r,c,a) # U(s) = R(s) + gamma*[summation(P(s'|s,a)*U(s'))]

            delta = np.max(np.abs(U_new - U))
            U = U_new
            history_avg.append(np.mean(U[~np.isnan(reward)]))
            history_start.append(U[start_state])
            
            if delta < 1e-4:
                break

        # POLICY IMPROVEMENT
        stable = True
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

    return policy, U, history_avg, history_start

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


def plot_policy(policy, title):

    arrow_map = {
        'U': '↑',
        'D': '↓',
        'L': '←',
        'R': '→',
        'W': '■'
    }

    fig, ax = plt.subplots(figsize=(cols, rows))

    # Draw arrows / walls
    for r in range(rows):
        for c in range(cols):

            if grid[r][c] is None:
                symbol = '■'
            else:
                symbol = arrow_map[policy[r][c]]

            ax.text(
                c + 0.5,
                rows - r - 0.5,
                symbol,
                ha='center',
                va='center',
                fontsize=18
            )

    # Set limits
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)

    # Equal cell size
    ax.set_aspect('equal')

    # Create table-style grid
    ax.set_xticks(np.arange(0, cols + 1, 1))
    ax.set_yticks(np.arange(0, rows + 1, 1))
    ax.grid(True)

    # Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.set_title(title)

    plt.show()


print("Running Value Iteration...")
U_vi, vi_avg, vi_start = value_iteration()
policy_vi = extract_policy(U_vi)

print("Running Policy Iteration...")
policy_pi, U_pi, pi_avg, pi_start = policy_iteration()

# Display results
print("\nUtilities from Value Iteration:")
print(np.round(U_vi, 3))
print("\nOptimal Policy (Value Iteration):")
for row in policy_vi:
    print(row)

print("\nUtilities from Policy Iteration:")
print(np.round(U_pi, 3))
print("\nOptimal Policy (Policy Iteration):")
for row in policy_pi:
    print(row)

# Plot policies
plot_policy(policy_vi, "Optimal Policy (Value Iteration)")
plot_policy(policy_pi, "Optimal Policy (Policy Iteration)")

# Plot utility estimates for the start state
plt.figure()
plt.plot(vi_start, label='Value Iteration')
plt.plot(pi_start, label='Policy Iteration')
plt.xlabel("Iteration")
plt.ylabel("Utility of Start State")
plt.title("Utility Estimates vs Iterations")
plt.legend()
plt.show()

# Also plot average utility (optional)
plt.figure()
plt.plot(vi_avg, label='Value Iteration')
plt.plot(pi_avg, label='Policy Iteration')
plt.xlabel("Iteration")
plt.ylabel("Average Utility")
plt.title("Average Utility vs Iterations")
plt.legend()
plt.show()