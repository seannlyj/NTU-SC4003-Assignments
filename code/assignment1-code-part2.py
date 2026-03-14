import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.patches import Rectangle

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

# Grid representation:
# 1  = green (+1 reward)
# -1 = brown (-1 reward)
# 0  = white (-0.05 reward)
# None = wall
def generate_random_grid(rows, cols, wall_prob=0.2, green_count=2, brown_count=2, rng_seed=None):
    """Generate a random grid with walls, terminal rewards, and a valid start state."""
    rng = np.random.default_rng(rng_seed)

    generated_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    wall_mask = rng.random((rows, cols)) < wall_prob

    for r in range(rows):
        for c in range(cols):
            if wall_mask[r, c]:
                generated_grid[r][c] = None

    valid_cells = [(r, c) for r in range(rows) for c in range(cols) if generated_grid[r][c] is not None]
    needed_cells = green_count + brown_count + 1
    if len(valid_cells) < needed_cells:
        raise ValueError("Not enough non-wall cells. Reduce wall_prob or terminal counts.")

    rng.shuffle(valid_cells)
    terminal_cells = valid_cells[:green_count + brown_count]
    start = valid_cells[green_count + brown_count]

    for i, (r, c) in enumerate(terminal_cells):
        generated_grid[r][c] = 1 if i < green_count else -1

    return generated_grid, start


def build_reward_matrix(env_grid):
    reward_matrix = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            if env_grid[r][c] is None:
                reward_matrix[r, c] = np.nan
            elif env_grid[r][c] == 1:
                reward_matrix[r, c] = 1
            elif env_grid[r][c] == -1:
                reward_matrix[r, c] = -1
            else:
                reward_matrix[r, c] = -0.05
    return reward_matrix


rows = 10
cols = 10

grid, start_state = generate_random_grid(rows, cols, wall_prob=0.2, green_count=2, brown_count=2)
reward = build_reward_matrix(grid)


# HELPER FUNCTIONS
def run_experiment(env_grid):
    start_time = time.time()
    U_vi, hist_vi = value_iteration(tracked)
    vi_time = time.time() - start_time

    start_time = time.time()
    policy_pi, U_pi, hist_pi = policy_iteration(tracked)
    pi_time = time.time() - start_time

    print(f"Value iteration: {len(hist_vi[tracked[0]])} iterations, {vi_time:.2f} sec")
    print(f"Policy iteration: {len(hist_pi[tracked[0]])} outer loops, {pi_time:.2f} sec")

# check if (r,c) is inside grid and not a wall
def is_valid(r, c):
    if r < 0 or r >= rows or c < 0 or c >= cols:
        return False
    return grid[r][c] is not None

# return new coordinates after attempting action, stays in place if move is invalid
def move(r, c, action):
    dr, dc = action_vectors[action]
    nr, nc = r + dr, c + dc
    if is_valid(nr, nc):
        return nr, nc
    return r, c

# compute sum_{s'} P(s'|s,a) * U(s') for state (r,c) and action a.
def expected_utility(U, r, c, a):
    # Intended direction
    nr, nc = move(r, c, a)
    total = p_intended * U[nr, nc]
    # Two perpendicular directions
    for sa in side_actions[a]:
        nr, nc = move(r, c, sa)
        total += p_side * U[nr, nc]
    return total

# VALUE ITERATION
def value_iteration(track_states):
    
    # Initialize all U(s) = 0
    U = np.zeros((rows, cols))
    hist_track = {s: [] for s in track_states if is_valid(*s)} # dict mapping each tracked state to a list of utilities per iteration
    
    # Iterate until convergence
    for iteration in range(5000):
        U_new = U.copy()
        
        for r in range(rows):
            for c in range(cols):
                
                if not is_valid(r, c):
                    continue
                best = -1e9
                for a in actions:
                    val = expected_utility(U, r, c, a)      # expected utility of taking action a in state (r,c)
                    if val > best:                          # max(Summation(P(s'|s,a)*U(s')))
                        best = val
                U_new[r, c] = reward[r, c] + gamma * best   # U(s) = R(s) + gamma*max(Summation(P(s'|s,a)*U(s')))
        
        # Record utilities for tracked states
        for (r, c) in hist_track.keys():
            hist_track[(r, c)].append(U_new[r, c])
        
        # Check convergence
        delta = np.max(np.abs(U_new - U))
        if delta < 1e-4:
            break
        U = U_new
            
    return U, hist_track

# POLICY ITERATION
def policy_iteration(track_states):
    # Initialize random policy
    policy = np.random.choice(actions,(rows,cols))
    for r in range(rows):
        for c in range(cols):
            if not is_valid(r, c):
                policy[r, c] = 'W'

    U = np.zeros((rows, cols))
    outer_hist = {s: [] for s in track_states if is_valid(*s)}
    stable = False

    while not stable:

        # POLICY EVALUATION
        while True:

            U_new = U.copy()

            for r in range(rows):
                for c in range(cols):

                    if not is_valid(r, c):
                        continue

                    a = policy[r,c]
                    U_new[r, c] = reward[r, c] + gamma * expected_utility(U, r, c, a) # U(s) = R(s) + gamma*[summation(P(s'|s,a)*U(s'))]

            delta = np.max(np.abs(U_new - U))
            U = U_new
            if delta < 1e-4:
                break

        # Record utilities after this evaluation (outer loop)
        for (r, c) in outer_hist.keys():
            outer_hist[(r, c)].append(U[r, c])

        # POLICY IMPROVEMENT
        stable = True
        for r in range(rows):
            for c in range(cols):

                if not is_valid(r, c):
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

    return policy, U, outer_hist

def extract_policy(U):
    
    policy = [['' for _ in range(cols)] for _ in range(rows)]
    
    for r in range(rows):
        for c in range(cols):
            
            if not is_valid(r, c):
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
    """Display the policy as arrows in a grid."""
    arrow_map = {'U': '↑', 'D': '↓', 'L': '←', 'R': '→', 'W': '■'}
    fig, ax = plt.subplots(figsize=(cols, rows))
    for r in range(rows):
        for c in range(cols):
            cell_val = grid[r][c]
            if cell_val is None:
                face_color = 'lightgrey'
            elif cell_val == -1:
                face_color = 'orange'
            elif cell_val == 1:
                face_color = 'green'
            else:
                face_color = 'white'

            ax.add_patch(Rectangle((c, rows - r - 1), 1, 1, facecolor=face_color, edgecolor='none'))
            symbol = arrow_map[policy[r][c]] if policy[r][c] in arrow_map else '?'
            ax.text(c + 0.5, rows - r - 0.5, symbol, ha='center', va='center', fontsize=18)
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(0, cols+1, 1))
    ax.set_yticks(np.arange(0, rows+1, 1))
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title)
    plt.show()

def plot_convergence(hist_dict, title):
    """Plot utility estimates of several states over iterations."""
    plt.figure()
    for state, values in hist_dict.items():
        plt.plot(values, label=f'State {state}')
    plt.xlabel('Iteration')
    plt.ylabel('Utility')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Choose a few representative states to track (including start state)
tracked = [(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1), start_state]   # corners + start
tracked = list(dict.fromkeys(tracked))
tracked = [s for s in tracked if is_valid(*s)]

print(f"Running on random {rows}x{cols} maze...")
start = time.time()
U_vi, hist_vi = value_iteration(tracked)
vi_time = time.time() - start
print(f"Value iteration: {len(hist_vi[tracked[0]])} iterations, {vi_time:.2f} sec")

start = time.time()
policy_pi, U_pi, hist_pi = policy_iteration(tracked)
pi_time = time.time() - start
print(f"Policy iteration: {len(hist_pi[tracked[0]])} outer loops, {pi_time:.2f} sec")

print("Running Value Iteration...")
U_vi, hist_vi = value_iteration(tracked)
policy_vi = extract_policy(U_vi)

print("Running Policy Iteration...")
policy_pi, U_pi, hist_pi = policy_iteration(tracked)

# Print results
print("\n" + "="*50)
print("Value Iteration Results")
print("="*50)
print("\nUtilities of all states (rounded to 3 decimals):")
print(np.round(U_vi, 3))
print("\nOptimal policy (↑ ↓ ← →, ■ = wall):")
for row in policy_vi:
    print(' '.join(row))

print("\n" + "="*50)
print("Policy Iteration Results")
print("="*50)
print("\nUtilities of all states (rounded to 3 decimals):")
print(np.round(U_pi, 3))
print("\nOptimal policy (↑ ↓ ← →, ■ = wall):")
for row in policy_pi:
    print(' '.join(row))

# Plot policies
plot_policy(policy_vi, "Optimal Policy (Value Iteration)")
plot_policy(policy_pi, "Optimal Policy (Policy Iteration)")

# Plot convergence of utilities
plot_convergence(hist_vi, "Value Iteration – Utility Estimates")
plot_convergence(hist_pi, "Policy Iteration – Utility Estimates")