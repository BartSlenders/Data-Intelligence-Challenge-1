import numpy as np


# Choose a random, non-terminal state (white square) for the agent to begin this new episode.
# Choose an action (move up, right, down, or left) for the current state. Actions will be chosen using an epsilon greedy
# algorithm. This algorithm will usually choose the most promising action for the agent, but it will occasionally choose
# a less promising option in order to encourage the agent to explore the environment.
# Perform the chosen action, and transition to the next state (i.e., move to the next location).
# Receive the reward for moving to the new state, and calculate the temporal difference.
# Update the Q-value for the previous state and action pair.
# If the new (current) state is a terminal state, go to #1. Else, go to #2.

# define an epsilon greedy algorithm that will choose which action to take next (i.e., where to move next)
def get_next_action(current_row, current_col, epsilon, q_values):
    # if a randomly chosen value between 0 and 1 is less than epsilon,
    # then choose the most promising value from the Q-table for this state.
    if np.random.random() < epsilon:
        return np.argmax(q_values[current_row, current_col])
    else:  # choose a random action
        return np.random.randint(4)


def take_action(action, r, row, col):
    """
    Return the new position and the reward of taking action in the current position
    :param action: the action for the current position
    :param r: a reward 2D array
    :param row: the x coordinate of the state
    :param col: the y coordinate of the state
    :returns new_i, new_j, reward: coordinates of next state and its reward
    """
    # possible new coordinates direction = ['n', 'e', 's', 'w']
    new_coordinates = [(row, col - 1), (row , col + 1),
                       (row + 1, col), (row - 1, col)]
    new_i, new_j = new_coordinates[action]
    # get new reward
    reward = r[new_i, new_j]

    # check if we can move to the new location
    if reward < 0:
        # if we have a wall/obstacle don't update location
        # reward = r[i_position, j_position]
        return row, col, reward
    else:
        return new_i, new_j, reward


# # define a function that will get the next location based on the chosen action
# def get_next_location(current_row, current_col, action, actions, rows, cols):
#     """"This function returns the location of our bot after a specific move from a specific location"""
#     if actions[action] == 'n':
#         return current_row - 1, current_col
#     elif actions[action] == 'e':
#         return current_row, current_col + 1
#     elif actions[action] == 's':
#         return current_row + 1, current_col
#     elif actions[action] == 'w':
#         return current_row, current_col - 1


def robot_epoch(robot, gamma=0.5):
    inputgrid = robot.grid.cells
    rows = robot.grid.n_rows
    cols = robot.grid.n_cols
    filthy_tiles = 0
    r_values = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            current_tile = inputgrid[j, i]
            if current_tile < -2 or current_tile == 0:  # if the robot is on the tile or the tile is clean
                r_values[i, j] = -0.1  # we consider the tile clean
            elif current_tile == 3 or current_tile < 0:  # if the tile is a death tile, wall or obstacle
                r_values[i, j] = -100  # we consider it as a wall
            else:  # clean or goal tiles keep their original values of 1 and 3
                r_values[i, j] = current_tile
                filthy_tiles += 1

    q_values = np.zeros((rows, cols, 4))  # 4 actions
    # define training parameters
    epsilon = 0.7  # the ratio of time when we should take the best action (instead of a random action)
    learning_rate = 0.9  # alpha in the pseudocode
    max_steps = rows * cols / 2
    # run through 200 training episodes
    for episode in range(200):
        # get the starting location for this episode
        col, row = robot.pos
        count = 0
        r = np.copy(r_values)
        done_cleaning = False
        cleaned_tiles = 0
        # continue moving until all tiles are clean or a lot of steps have been taken without finding a clean tile
        while not done_cleaning:
            old_row, old_column = row, col  # store the old row and column indexes
            # choose which action to take (i.e., where to move next)
            action = get_next_action(row, col, epsilon, q_values)
            # perform the chosen action, and transition to the next state (i.e., move to the next location)
            row, col, reward = take_action(action, r, row, col)
            count += 1
            old_q_value = q_values[old_row, old_column, action]
            temporal_difference = reward + (gamma * np.max(q_values[row, col])) - old_q_value
            # update the Q-value for the previous state and action pair
            q_values[old_row, old_column, action] = old_q_value + (learning_rate * temporal_difference)

            # this is for termination only
            if reward == 1:
                count = 0
                r[row, col] = -0.1
                cleaned_tiles += 1
                if cleaned_tiles == filthy_tiles:
                    done_cleaning = True
            # this is here to prevent us from looping infinitely when there are unreachable filthy tiles
            elif count > max_steps:
                # print('reached max steps without any gain')
                break
    action = get_next_action(robot.pos[1], robot.pos[0], 1, q_values)
    direction = ['n', 'e', 's', 'w'][action]
    print('we want to move in direction', direction)
    while robot.orientation != direction:
        robot.rotate('r')
    print(q_values[robot.pos[1], robot.pos[0]])
    robot.move()

    # Q(st, at) ← Q(st, at) + α[rt + γ max_a Q(st+1, a) − Q(st, at)]
