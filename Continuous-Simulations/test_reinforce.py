# Import our robot algorithm to use in this simulation:
from reinforce import robot_epoch as rl
from continuous import parse_config, Robot

import pickle
import numpy as np
import pandas as pd

# Cleaned tile percentage at which the room is considered 'clean':
stopping_criteria = 100


def run(gamma=0.9, alpha=0.5, episodes=200, steps=200):
    """
    Executes a run of Reinforce
    :param gamma: the discount factor
    :param alpha: learning rate
    :param episodes: nr of episodes
    :param steps: nr of steps
    """
    # Open the grid file.
    # (You can create one yourself using the provided editor).
    grid = parse_config('random_house_0.grid')
    grid.spawn_robots([Robot(id=1, battery_drain_p=1, battery_drain_lam=0.5)],
                      [(0, 0)])
    total_dirty = len(grid.goals)
    # Keep track of the number of robot decision epochs:
    n_epochs = 0
    while True:
        n_epochs += 1
        # Do a robot epoch (basically call the robot algorithm once):
        for robot in grid.robots:
            rl(robot=robot, gamma=gamma, alpha=alpha, episodes=episodes, steps=steps)

        # Calculate the cleaned percentage:
        goals_left = len(grid.goals)
        clean_percent = (total_dirty - goals_left / total_dirty) * 100
        # If the room is 'clean' or 100 iterations have passed, we stop:
        if goals_left <= 5 or n_epochs >= 100:
            break
    return clean_percent


def generate_results(gamma, alpha, episodes, steps, runs_per_combination=3):
    """
    Generates a csv file under the name "results.csv" containing the probabilities and efficiencies of multiple runs
    of sarsa/q-learning, together with the parameters used
    :param gamma: the discount factor
    :param alpha: learning rate
    :param episodes: nr of episodes
    :param steps: nr of steps
    :param runs_per_combination: nr of runs for each combination
    """
    rows = []
    for g in gamma:
        for a in alpha:
            for episode in episodes:
                for s in steps:
                    print('gamma:', g, '\talpha:', a, '\tepisodes:', episode, '\tsteps:', s)
                    for i in range(runs_per_combination):
                        cleaned = run(gamma=g, alpha=a, episodes=episode, steps=s)
                        rows.append([g, a, episode, s, cleaned])
                    print('\tcleaned:', cleaned)
    my_array = np.array(rows)
    df = pd.DataFrame(my_array, columns=['gamma', 'alpha', 'episodes', 'steps', 'cleaned'])
    df.to_csv("results.csv")

gamma = np.array([0.5, 0.7, 0.9])
alpha = np.array([0.3, 0.6, 0.9])
episodes = np.array([50, 150, 300])
steps = np.array([50, 150, 300])

generate_results(gamma, alpha, episodes, steps)
