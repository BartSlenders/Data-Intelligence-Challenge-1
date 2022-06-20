# Import our robot algorithm to use in this simulation:
from reinforce import robot_epoch as rl
from continuous import parse_config, Robot

import numpy as np
import pandas as pd

# Cleaned tile percentage at which the room is considered 'clean':
stopping_criteria = 100


def run(gamma, alpha, episodes, steps):
    """
    Executes a run of Reinforce
    :param gamma: discount factor
    :param alpha: learning rate for both networks
    :param episodes: number of episodes
    :param steps: number of steps
    """
    # Open the grid file.
    # (You can create one yourself using the provided editor).
    grid = parse_config('random_house_0.grid')
    grid.spawn_robots([Robot(id=1, battery_drain_p=1, battery_drain_lam=0.5)],
                      [(0, 0)])

    max_filthy = len(grid.filthy)
    max_goals = len(grid.goals)

    # Keep track of the number of robot decision epochs:
    while True:
        # Stop simulation if all robots died:
        if all([not robot.alive for robot in grid.robots]):
            break
        # Do a robot epoch (basically call the robot algorithm once):
        for robot in grid.robots:
            if robot.alive:
                rl(robot=robot, gamma=gamma, alpha=alpha, episodes=episodes, steps=steps)

        # Calculate the cleaned percentage:
        cleanpercent, _ = grid.evaluate(max_filthy, max_goals)
        # If the room is 'clean', we stop:
        if cleanpercent >= stopping_criteria:
            break

    return cleanpercent


def generate_results(gamma, alpha, episodes, steps, runs_per_combination=3):
    """
    Generates a csv file under the name "reinforce_results.csv" containing the probabilities and efficiencies of multiple runs
    of REINFORCE, together with the parameters used
    :param gamma: discount factor
    :param alpha: learning rate for both networks
    :param episodes: number of episodes
    :param steps: number of steps
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
                        df.to_csv("reinforce_results.csv")

# original
gamma = [0.9, 0.95, 0.99]
alpha = [0.001, 0.01, 0.1]
episodes = [20, 40, 60]
steps = [40, 60, 100]

generate_results(gamma, alpha, episodes, steps)
