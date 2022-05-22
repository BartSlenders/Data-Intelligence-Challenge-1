# Import our robot algorithm to use in this simulation:
from robot_configs.qlearning import robot_epoch as ql
from robot_configs.sarsa import robot_epoch as sa

import pickle
from environment import Robot
import numpy as np
import pandas as pd

# Cleaned tile percentage at which the room is considered 'clean':
stopping_criteria = 100

# Keep track of some statistics:
efficiencies = []
n_moves = []
cleaned = []

"""
    Executes a run of SARSA or Q-learning. 
    @g: the gamma parameter
    @t: the theta parameter
    @c: the certainty used for policy iteration 
"""

def run(qlearning=True, sarsa = False, gamma=0.9, epsilon=0.1, alpha=0.5, episodes=200, steps=200, grid_file='house.grid'):
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
        if sarsa:
            sa(robot, gamma=gamma, epsilon=epsilon, alpha=alpha, episodes=episodes, steps=steps)
        elif qlearning:
            ql(robot, gamma=gamma, epsilon=epsilon, alpha=alpha, episodes=episodes, steps=steps)
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
    Generates a csv file under the name "results.csv" containing the probabilities and efficiencies of multiple runs
    of policy iteration, together with the parameters used.
    @gamma: the gamma parameter
"""


def generate_results(gamma, epsilon, alpha, episodes, steps, runs_per_combination=3):
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

gamma = np.array([0.5, 0.7, 0.9])
# Epsilon from epsilon-greedy
epsilon = np.array([0.05, 0.1, 0.2])
# Learning rate
alpha = np.array([0.3, 0.6, 0.9])
episodes = np.array([50, 150, 300])
steps = np.array([50, 150, 300])

generate_results(gamma, epsilon, alpha, episodes, steps)
