# Import our robot algorithm to use in this simulation:
from ppo import robot_epoch as rl
from continuous import parse_config, Robot

import numpy as np
import pandas as pd

# Cleaned tile percentage at which the room is considered 'clean':
stopping_criteria = 100


def run(gamma, epsilon, c1, c2, k_epoch, actor_lr, critic_lr, episodes, steps):
    """
    Executes a run of PPO
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
                rl(robot=robot, gamma=gamma, epsilon=epsilon, c1=c1, c2=c2, k_epoch=k_epoch, actor_lr=actor_lr,
                   critic_lr=critic_lr, episodes=episodes, steps=steps)

        # Calculate the cleaned percentage:
        cleanpercent, _ = grid.evaluate(max_filthy, max_goals)
        # If the room is 'clean', we stop:
        if cleanpercent >= stopping_criteria:
            break

    return cleanpercent


def generate_results(gamma, epsilon, c1, c2, k_epoch, actor_lr, critic_lr, episodes, steps, runs_per_combination=3):
    """
    Generates a csv file under the name "results.csv" containing the probabilities and efficiencies of multiple runs
    of PPO, together with the parameters used
    :param gamma: the discount factor
    :param alpha: learning rate
    :param episodes: nr of episodes
    :param steps: nr of steps
    :param runs_per_combination: nr of runs for each combination
    """
    rows = []
    for g in gamma:
        for e in epsilon:
            for c_1 in c1:
                for c_2 in c2:
                    for epoch in k_epoch:
                        for a_lr in actor_lr:
                            for c_lr in critic_lr:
                                for episode in episodes:
                                    for s in steps:
                                        print('gamma:', g, '\tepsilon:', e, '\tc1:', c_1, '\tc_2:', c_2,
                                              '\tepoch:', epoch, '\ta_lr:', a_lr, '\tc_lr:', c_lr,
                                              '\tepisodes:', episode, '\tsteps:', s)
                                        for i in range(runs_per_combination):
                                            cleaned = run(g, e, c_1, c_2, epoch, a_lr, c_lr, episode, s)
                                            rows.append([g, e, c_1, c_2, epoch, a_lr, c_lr, episode, s, cleaned])
                                        print('\tcleaned:', cleaned)
    my_array = np.array(rows)
    df = pd.DataFrame(my_array, columns=['gamma', 'epsilon', 'c1', 'c2', 'k_epoch', 'actor_lr', 'critic_lr', 'episodes',
                                         'steps', 'cleaned'])
    df.to_csv("ppo_results.csv")


gamma = [0.95, 0.99]
epsilon = [0.1, 0.2]
c1 = [0.5, 0.1]
c2 = [0.01, 0.1]
k_epoch = [20, 40]
actor_lr = [0.0003, 0.001]
critic_lr = [0.001, 0.01]
episodes = [10, 20, 40]
steps = [20, 40, 60]

generate_results(gamma, epsilon, c1, c2, k_epoch, actor_lr, critic_lr, episodes, steps)
