import numpy as np


def init_rewards(rows, cols, inputgrid):
    """
    Return a 2D reward array for each cell based on the type of the cell
    :param rows: # of rows in inputgrid
    :param cols: # of columns in inputgrid
    :param inputgrid: a 2D array, where each cell holds the type of the corresponding tile
    :returns r: a reward 2D array
    """
    r = np.zeros((cols, rows))
    for i in range(cols):
        for j in range(rows):
            if inputgrid[i, j] < -2:  # if the robot is on the tile
                r[i, j] = 0  # we consider the tile clean
            elif inputgrid[i, j] == 3:  # if the tile is a death tile
                r[i, j] = -1  # we consider it as a wall
            else:
                r[i, j] = inputgrid[i, j]

    return r


def get_action(Q, epsilon, i_position, j_position):
    """
    Return an action for the current position using policy derived from Q (epsilon-greedy)
    :param Q: state-action value function
    :param epsilon: probability of taking a random action
    :param i_position: the x coordinate of the state
    :param j_position: the y coordinate of the state
    :returns action: an action for the current position using policy derived from Q (epsilon-greedy)
    """
    if np.random.uniform(0, 1) < epsilon:
        # take a random action
        action = np.random.randint(0, 4)
    else:
        # get the action with the highest value
        action = max(Q[i_position][j_position], key=Q[i_position][j_position].get)

    return action


def take_action(action, r, i_position, j_position):
    """
    Return the new position and the reward of taking action in the current position
    :param action: the action for the current position
    :param r: a reward 2D array
    :param i_position: the x coordinate of the state
    :param j_position: the y coordinate of the state
    :returns new_i, new_j, reward: coordinates of next state and its reward
    """
    # possible new coordinates
    new_coordinates = [(i_position, j_position - 1), (i_position + 1, j_position),
                       (i_position, j_position + 1), (i_position - 1, j_position)]
    new_i, new_j = new_coordinates[action]
    # get new reward
    reward = r[new_i][new_j]

    # check if we can move to the new location
    if reward < 0:
        # if we have a wall/obstacle don't update location
        return i_position, j_position, reward
    else:
        return new_i, new_j, reward


# def get_available_position(r):
#     positions = []
#     for i in range(len(r)):
#         for j in range(len(r[0])):
#             if r[i, j] >= 0:
#                 positions.append((i, j))
#     return positions


def robot_epoch(robot, gamma=0.9, epsilon=0.1, alpha=0.5, episodes = 200, steps = 200):
    """
    Execute SARSA algorithm to find the best move
    :param robot: main actor of type Robot
    :param gamma: discount factor
    :param epsilon: probability of taking a random action
    :param alpha: learning rate
    """
    inputgrid = robot.grid.cells
    rows = robot.grid.n_rows
    cols = robot.grid.n_cols
    r = init_rewards(rows, cols, inputgrid)
    # initialize the state-action value function with 0 for every state action pair
    Q = [[{0: 0, 1: 0, 2: 0, 3: 0} for _ in range(rows)] for _ in range(cols)]

    for _ in range(episodes):
        # initial state coordinates
        i_position, j_position = robot.pos
        action = get_action(Q, epsilon, i_position, j_position)
        for _ in range(steps):
            new_i, new_j, reward = take_action(action, r, i_position, j_position)
            new_action = get_action(Q, epsilon, new_i, new_j)
            Q[i_position][j_position][action] = (1 - alpha) * Q[i_position][j_position][action] \
                                                 + alpha * (reward + gamma * Q[new_i][new_j][new_action])

            i_position, j_position, action = new_i, new_j, new_action

    # obtain the best action from Q for the current state
    choice = max(Q[robot.pos[0]][robot.pos[1]], key=Q[robot.pos[0]][robot.pos[1]].get)
    direction = ['n', 'e', 's', 'w'][choice]
    while robot.orientation != direction:
        robot.rotate('r')
    robot.move()
