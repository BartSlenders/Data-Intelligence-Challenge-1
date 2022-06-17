# Import our robot algorithm to use in this simulation:
from sac import robot_epoch as rl
from continuous import parse_config, Robot

import numpy as np
import pandas as pd

# Cleaned tile percentage at which the room is considered 'clean':
stopping_criteria = 100


def run(gamma, alpha, beta, batch_size, reward_scale, episodes, steps):
    """
    Executes a run of SAC
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
                rl(robot=robot, gamma=gamma, alpha=alpha, beta=beta, batch_size=batch_size, reward_scale=reward_scale,
                   episodes=episodes, steps=steps)

        # Calculate the cleaned percentage:
        cleanpercent, _ = grid.evaluate(max_filthy, max_goals)
        # If the room is 'clean', we stop:
        if cleanpercent >= stopping_criteria:
            break

    return cleanpercent


def generate_results(gamma, alpha, beta, batch_size, reward_scale, episodes, steps, runs_per_combination=3):
    """
    Generates a csv file under the name "results.csv" containing the probabilities and efficiencies of multiple runs
    of SAC, together with the parameters used
    :param gamma: the discount factor
    :param alpha: learning rate
    :param episodes: nr of episodes
    :param steps: nr of steps
    :param runs_per_combination: nr of runs for each combination
    """
    rows = []
    for g in gamma:
        for a in alpha:
            for b in beta:
                for batch in batch_size:
                    for reward in reward_scale:
                        for episode in episodes:
                            for s in steps:
                                print('gamma:', g, '\talpha:', a, '\tbeta:', b, '\tbatch_size:', batch,
                                      '\treward_scale:', reward, '\tepisodes:', episode, '\tsteps:', s)
                                for i in range(runs_per_combination):
                                    cleaned = run(gamma=g, alpha=a, beta=b, batch_size=batch, reward_scale=reward,
                                                  episodes=episode, steps=s)
                                    rows.append([g, a, b, batch, reward, episode, s, cleaned])
                                    print('\tcleaned:', cleaned)
    my_array = np.array(rows)
    df = pd.DataFrame(my_array, columns=['gamma', 'alpha', 'beta', 'batch_size', 'reward_scale', 'episodes', 'steps',
                                         'cleaned'])
    df.to_csv("sac_results.csv")


gamma = [0.9, 0.95, 0.99]
alpha = [0.001, 0.01]
beta = [0.001, 0.01]
batch_size = [10, 15]
reward_scale = [0.1, 1, 2]
episodes = [1, 7, 20]
steps = [20, 40, 90]

generate_results(gamma, alpha, beta, batch_size, reward_scale, episodes, steps)
