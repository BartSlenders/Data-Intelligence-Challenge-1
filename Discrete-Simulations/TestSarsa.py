# Import our robot algorithm to use in this simulation:
from robot_configs.sarsa import robot_epoch
import pickle
from environment import Robot
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# Cleaned tile percentage at which the room is considered 'clean':
stopping_criteria = 100

# Keep track of some statistics:
efficiencies = []
n_moves = []
cleaned = []


"""
    Executes a run of Policy iteration. 
    @g: the gamma parameter
    @t: the theta parameter
    @c: the certainty used for policy iteration 
"""
def run(gamma=0.9, epsilon=0.1, alpha=0.5, episodes = 200, steps = 200, grid_file = 'house.grid'):
    deaths = 0
    # Open the grid file.
    # (You can create one yourself using the provided editor).
    with open(f'grid_configs/{grid_file}', 'rb') as f:
        grid = pickle.load(f)
    # Calculate the total visitable tiles:
    n_total_tiles = (grid.cells >= 0).sum()
    # Spawn the robot at (1,1) facing north with battery drainage enabled:
    robot = Robot(grid, (1, 1), orientation='n', battery_drain_p=1, battery_drain_lam=0.5)
    # Keep track of the number of robot decision epochs:
    n_epochs = 0
    while True:
        n_epochs += 1
        # Do a robot epoch (basically call the robot algorithm once):
        robot_epoch(robot, gamma=gamma, epsilon=epsilon, alpha=alpha, episodes = episodes, steps = steps)
        # Stop this simulation instance if robot died :( :
        if not robot.alive:
            deaths += 1
            break
        # Calculate some statistics:
        clean = (grid.cells == 0).sum()
        dirty = (grid.cells >= 1).sum()
        goal = (grid.cells == 2).sum()
        # Calculate the cleaned percentage:
        clean_percent = (clean / (dirty + clean)) * 100
        # See if the room can be considered clean, if so, stop the simulaiton instance:
        if clean_percent >= stopping_criteria and goal == 0:
            break
        # Calculate the effiency score:
        moves = [(x, y) for (x, y) in zip(robot.history[0], robot.history[1])]
        u_moves = set(moves)
        n_revisted_tiles = len(moves) - len(u_moves)
        efficiency = (100 * n_total_tiles) / (n_total_tiles + n_revisted_tiles)
    return clean_percent, efficiency

"""
    Plots violin plots representing the distribution of cleanliness and efficiency
     of 10 consecutive runs of Policy iteration. 
    @g: the gamma parameter
    @t: the theta parameter
    @c: the certainty used for policy iteration 
"""
def plotDistribution(gamma=0.9, epsilon=0.1, alpha=0.5, episodes = 200, steps = 200, runs=10):
    cleaned = []
    efficiencies = []
    for i in range(runs):
        cleaned_a, efficiency = run(gamma=gamma, epsilon=epsilon, alpha=alpha, episodes=episodes, steps=steps)
        cleaned.append(cleaned_a)
        efficiencies.append(efficiency)
    my_array = np.array([cleaned, efficiencies]).T
    df = pd.DataFrame(my_array, columns = ['cleaned', 'efficiencies'])
    ax = sns.violinplot(data=df)
    ax.set_ylabel("iteration policy performance (%)")
    ax.set_title("distribution of robot performance", size=18, weight='bold');
    plt.show()

"""
    Plots violin plots representing the distribution of cleanliness of 10 consecutive runs of Policy iteration on
     multiple grids. 
    @g: the gamma parameter
"""
def plotcleanness(gamma=0.9, epsilon=0.1, alpha=0.5, episodes = 200, steps = 200, runs=10):
    grid_files = ["death.grid", "house.grid", "example-random-house-3.grid", "snake.grid"]
    grid_files_names = ["death", "house", "random-house-3", "snake"]
    array = []
    for gf in grid_files:
        efficiencies = []
        for i in range(runs):
            cleaned_a, efficiency = run(gamma=gamma, epsilon=epsilon, alpha=alpha, episodes=episodes, steps=steps)
            efficiencies.append(efficiency)
        array.append(efficiencies)

    my_array = np.array(array).T
    df = pd.DataFrame(my_array, columns = grid_files_names)
    ax =sns.violinplot(data=df)

    ax.set_ylabel("efficiency (%)")
    ax.set_title("policy iteration - efficiency vs grid", size=18, weight='bold');
    plt.show()

"""
    Generates a csv file under the name "results.csv" containing the probabilities and efficiencies of multiple runs
    of policy iteration, together with the parameters used.
    @gamma: the gamma parameter
"""
def generate_results(gamma, epsilon, alpha, episodes, steps, runs_per_combination= 3):
    rows = []
    for g in gamma:
        for e in epsilon:
            for a in alpha:
                for episode in episodes:
                    for s in steps:
                        print('gamma:', g, '\tepsilon:', e, '\talpha:', a, '\tepisodes:', episode, '\tsteps:', s)
                        for i in range(runs_per_combination):
                            cleaned, efficiency = run(gamma=g, epsilon=e, alpha=a, episodes=episode, steps=s)
                            rows.append([g, e, a, episode, s, cleaned, efficiency])
                        print('\tcleaned:', cleaned, '\tefficiency:', efficiency)
    my_array = np.array(rows)
    df = pd.DataFrame(my_array, columns=['gamma', 'epsilon', 'alpha', 'episodes', 'steps', 'cleaned', 'efficiency'])
    df.to_csv("results.csv")

gamma = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.99])
# Epsilon from epsilon-greedy
epsilon = np.array([0.001, 0.05, 0.1, 0.3])
# Learning rate
alpha = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.99])
episodes = np.array([10, 50, 100, 200, 400])
steps = np.array([100, 200, 400, 800])

generate_results(gamma, epsilon, alpha, episodes, steps)