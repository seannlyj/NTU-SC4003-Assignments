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

# Generate a random grid with specified parameters, ensuring at least one valid start state and enough terminal states.
def generate_random_grid(rows, cols, wall_prob=0.5, green_count=3, brown_count=3, rng_seed=None):
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
    rows = len(env_grid)
    cols = len(env_grid[0])
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

# check if (r,c) is inside grid and not a wall
def is_valid(env_grid, r, c):
    rows = len(env_grid)
    cols = len(env_grid[0])
    if r < 0 or r >= rows or c < 0 or c >= cols:
        return False
    return env_grid[r][c] is not None

# return new coordinates after attempting action, stays in place if move is invalid
def move(env_grid, r, c, action):
    dr, dc = action_vectors[action]
    nr, nc = r + dr, c + dc
    if is_valid(env_grid, nr, nc):
        return nr, nc
    return r, c

# compute sum_{s'} P(s'|s,a) * U(s') for state (r,c) and action a.
def expected_utility(U, env_grid, r, c, a):
    # Intended direction
    nr, nc = move(env_grid, r, c, a)
    total = p_intended * U[nr, nc]
    # Two perpendicular directions
    for sa in side_actions[a]:
        nr, nc = move(env_grid, r, c, sa)
        total += p_side * U[nr, nc]
    return total

# VALUE ITERATION
def value_iteration(env_grid, reward_matrix, track_states, max_iterations=5000, tol=1e-4):
    rows = len(env_grid)
    cols = len(env_grid[0])
    
    # Initialize all U(s) = 0
    U = np.zeros((rows, cols))
    hist_track = {s: [] for s in track_states if is_valid(env_grid, *s)} # dict mapping each tracked state to a list of utilities per iteration
    avg_utility_history = []
    valid_cells = ~np.isnan(reward_matrix)
    
    # Iterate until convergence
    for iteration in range(max_iterations):
        U_new = U.copy()
        
        for r in range(rows):
            for c in range(cols):
                
                if not is_valid(env_grid, r, c):
                    continue
                best = -1e9
                for a in actions:
                    val = expected_utility(U, env_grid, r, c, a)      # expected utility of taking action a in state (r,c)
                    if val > best:                          # max(Summation(P(s'|s,a)*U(s')))
                        best = val
                U_new[r, c] = reward_matrix[r, c] + gamma * best   # U(s) = R(s) + gamma*max(Summation(P(s'|s,a)*U(s')))
        
        # Record utilities for tracked states
        for (r, c) in hist_track.keys():
            hist_track[(r, c)].append(U_new[r, c])
        avg_utility_history.append(np.mean(U_new[valid_cells]))
        
        # Check convergence
        delta = np.max(np.abs(U_new - U))
        if delta < tol:
            break
        U = U_new
            
    return U, hist_track, avg_utility_history

# POLICY ITERATION
def policy_iteration(env_grid, reward_matrix, track_states, tol=1e-4):
    rows = len(env_grid)
    cols = len(env_grid[0])

    # Initialize random policy
    policy = np.random.choice(actions, (rows, cols))
    for r in range(rows):
        for c in range(cols):
            if not is_valid(env_grid, r, c):
                policy[r, c] = 'W'

    U = np.zeros((rows, cols))
    outer_hist = {s: [] for s in track_states if is_valid(env_grid, *s)}
    avg_utility_history = []
    valid_cells = ~np.isnan(reward_matrix)
    stable = False

    while not stable:

        # POLICY EVALUATION
        while True:

            U_new = U.copy()

            for r in range(rows):
                for c in range(cols):

                    if not is_valid(env_grid, r, c):
                        continue

                    a = policy[r,c]
                    U_new[r, c] = reward_matrix[r, c] + gamma * expected_utility(U, env_grid, r, c, a) # U(s) = R(s) + gamma*[summation(P(s'|s,a)*U(s'))]

            delta = np.max(np.abs(U_new - U))
            U = U_new
            if delta < tol:
                break

        # Record utilities after this evaluation (outer loop)
        for (r, c) in outer_hist.keys():
            outer_hist[(r, c)].append(U[r, c])
        avg_utility_history.append(np.mean(U[valid_cells]))

        # POLICY IMPROVEMENT
        stable = True
        for r in range(rows):
            for c in range(cols):

                if not is_valid(env_grid, r, c):
                    continue

                old_action = policy[r,c]

                best_action = None
                best_val = -1e9

                for a in actions:
                    #Summation(P(s'|s,a)*U(s'))
                    val = expected_utility(U, env_grid, r, c, a)

                     # argmax(Summation(P(s'|s,a)*U(s')))
                    if val > best_val:
                        best_val = val
                        best_action = a

                policy[r,c] = best_action

                if best_action != old_action:
                    stable = False

    return policy, U, outer_hist, avg_utility_history

def extract_policy(U, env_grid):
    rows = len(env_grid)
    cols = len(env_grid[0])
    
    policy = [['' for _ in range(cols)] for _ in range(rows)]
    
    for r in range(rows):
        for c in range(cols):
            
            if not is_valid(env_grid, r, c):
                policy[r][c] = 'W'
                continue
                
            best_action = None
            best_val = -1e9
            
            for a in actions:
                val = expected_utility(U, env_grid, r, c, a)
                if val > best_val:
                    best_val = val
                    best_action = a
                    
            policy[r][c] = best_action
            
    return policy


# DIsplay policy as arrows in grid
def plot_policy(policy, env_grid, title):
    rows = len(env_grid)
    cols = len(env_grid[0])
    arrow_map = {'U': '↑', 'D': '↓', 'L': '←', 'R': '→', 'W': '■'}
    fig, ax = plt.subplots(figsize=(cols, rows))
    for r in range(rows):
        for c in range(cols):
            cell_val = env_grid[r][c]
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

# Plot utility estimates of tracked states over iterations
def plot_convergence(hist_dict, title):
    plt.figure()
    for state, values in hist_dict.items():
        plt.plot(values, label=f'State {state}')
    plt.xlabel('Iteration')
    plt.ylabel('Utility')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

maze_sizes = [30, 40, 50, 100]

comparison_vi_iters = []
comparison_pi_iters = []
comparison_vi_times = []
comparison_pi_times = []
comparison_vi_avg_utils_by_size = {}
comparison_pi_avg_utils_by_size = {}

for maze_size in maze_sizes:
    rows = maze_size
    cols = maze_size

    grid, start_state = generate_random_grid(
        rows,
        cols,
        wall_prob=0.2,
        green_count=20,
        brown_count=20,
        rng_seed=maze_size,
    )
    reward = build_reward_matrix(grid)

    # Randomly select 5 valid (non-wall) states to track
    valid_states = [(r, c) for r in range(rows) for c in range(cols) if is_valid(grid, r, c)]
    track_count = min(5, len(valid_states))
    state_rng = np.random.default_rng(maze_size + 1000)
    selected_indices = state_rng.choice(len(valid_states), size=track_count, replace=False)
    tracked = [valid_states[i] for i in selected_indices]

    print(f"\nRunning on random {rows}x{cols} maze...")

    print("Running Value Iteration...")
    start = time.time()
    U_vi, hist_vi, vi_avg_utility_history = value_iteration(grid, reward, tracked)
    vi_time = time.time() - start
    policy_vi = extract_policy(U_vi, grid)

    print("Running Policy Iteration...")
    start = time.time()
    policy_pi, U_pi, hist_pi, pi_avg_utility_history = policy_iteration(grid, reward, tracked)
    pi_time = time.time() - start

    vi_iters = max((len(values) for values in hist_vi.values()), default=0)
    pi_iters = max((len(values) for values in hist_pi.values()), default=0)

    comparison_vi_iters.append(vi_iters)
    comparison_pi_iters.append(pi_iters)
    comparison_vi_times.append(vi_time)
    comparison_pi_times.append(pi_time)
    comparison_vi_avg_utils_by_size[maze_size] = vi_avg_utility_history
    comparison_pi_avg_utils_by_size[maze_size] = pi_avg_utility_history

    print(f"Value iteration: {vi_iters} iterations, {vi_time:.2f} sec")
    print(f"Policy iteration: {pi_iters} outer loops, {pi_time:.2f} sec")

    # Print results
    print("\n" + "="*50)
    print(f"Value Iteration Results ({rows}x{cols})")
    print("="*50)
    print("\nUtilities of all states (rounded to 3 decimals):")
    print(np.round(U_vi, 3))
    print("\nOptimal policy (↑ ↓ ← →, ■ = wall):")
    for row in policy_vi:
        print(' '.join(row))

    print("\n" + "="*50)
    print(f"Policy Iteration Results ({rows}x{cols})")
    print("="*50)
    print("\nUtilities of all states (rounded to 3 decimals):")
    print(np.round(U_pi, 3))
    print("\nOptimal policy (↑ ↓ ← →, ■ = wall):")
    for row in policy_pi:
        print(' '.join(row))

    # Plot per-grid policies
    plot_policy(policy_vi, grid, f"Optimal Policy (Value Iteration) - {rows}x{cols}")
    plot_policy(policy_pi, grid, f"Optimal Policy (Policy Iteration) - {rows}x{cols}")

    # Plot per-grid utility convergence
    plot_convergence(hist_vi, f"Value Iteration – Utility Estimates ({rows}x{cols})")
    plot_convergence(hist_pi, f"Policy Iteration – Utility Estimates ({rows}x{cols})")


# Final comparison plots across maze sizes
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].plot(maze_sizes, comparison_vi_iters, marker='o', label='Value Iteration')
axes[0].plot(maze_sizes, comparison_pi_iters, marker='s', label='Policy Iteration')
axes[0].set_xlabel('Maze size (N for NxN)')
axes[0].set_ylabel('Iterations to converge')
axes[0].set_title('Iterations vs Maze Size')
axes[0].set_xticks(maze_sizes)
axes[0].grid(True)
axes[0].legend()

axes[1].plot(maze_sizes, comparison_vi_times, marker='o', label='Value Iteration')
axes[1].plot(maze_sizes, comparison_pi_times, marker='s', label='Policy Iteration')
axes[1].set_xlabel('Maze size (N for NxN)')
axes[1].set_ylabel('Runtime (seconds)')
axes[1].set_title('Runtime vs Maze Size')
axes[1].set_xticks(maze_sizes)
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.show()

# Final average-utility tracking plots (different colored lines by maze size)
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

for maze_size in maze_sizes:
    vi_series = comparison_vi_avg_utils_by_size[maze_size]
    axes[0].plot(np.arange(1, len(vi_series) + 1), vi_series, label=f'{maze_size}x{maze_size}')
    
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Average utility (valid cells)')
axes[0].set_title('Value Iteration: Avg Utility over Iterations')
axes[0].grid(True)
axes[0].legend()

for maze_size in maze_sizes:
    pi_series = comparison_pi_avg_utils_by_size[maze_size]
    axes[1].plot(np.arange(1, len(pi_series) + 1), pi_series, label=f'{maze_size}x{maze_size}')

axes[1].set_xlabel('Outer Policy Iteration Loop')
axes[1].set_ylabel('Average utility (valid cells)')
axes[1].set_title('Policy Iteration: Avg Utility over Iterations')
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.show()