import random
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
        r[new_i][new_j] = 0
        return new_i, new_j, reward


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


def robot_epoch(robot, gamma=0.5, epsilon=0.1, episodes=900, steps=6):
    """
    Execute MC algorithm to find the best move
    :param robot: main actor of type Robot
    :param gamma: discount factor
    :param epsilon: probability of taking a random action
    :param episodes: number of episodes
    :param steps: number of steps
    """
    inputgrid = robot.grid.cells
    rows = robot.grid.n_rows
    cols = robot.grid.n_cols
    r = init_rewards(rows, cols, inputgrid)

    # initialize the state-action value function with 0 for every state action pair
    Q = [[{0: np.random.uniform(), 1: np.random.uniform(), 2: np.random.uniform(), 3: np.random.uniform()} for _ in range(rows)] for _ in range(cols)]
    # initialize the returns with empty list for every state action pair
    returns = [[{0: [], 1: [], 2: [], 3: []} for _ in range(rows)] for _ in range(cols)]
    x_pos, y_pos = robot.pos

    policy = np.full((cols, rows), 0)
    for i in range(1, cols-1):
        for j in range(1, rows-1):
            actions = []
            # we will check the surrounding tiles
            if r[i, j - 1] >= 0:
                actions.append(0)
            if r[i + 1, j] >= 0:
                actions.append(1)
            if r[i, j + 1] >= 0:
                actions.append(2)
            if r[i - 1, j] >= 0:
                actions.append(3)
            policy[i][j] = random.choice(actions)

    for _ in range(episodes):
        episode = []
        current_i, current_j = x_pos, y_pos
        r = init_rewards(rows, cols, inputgrid)
        # generate an episode following policy
        for t in range(steps):
            action = policy[current_i][current_j]
            new_i, new_j, reward = take_action(action, r, current_i, current_j)
            episode.append((current_i, current_j, action, reward))
            current_i, current_j = new_i, new_j

        G = 0
        state_action_pairs = [(element[0], element[1], element[2]) for element in episode]
        reversed_episode = episode[::-1]
        reversed_state_action_pairs = state_action_pairs[::-1]

        index = 1
        # loop for each step in episode, t = T-1,...,0
        for element in reversed_episode:
            G = G*gamma + element[3]
            # unless the pair S_t, A_t appears in S_0, A_0, ...,S_{t-1}, A_{t-1}
            if (element[0], element[1], element[2]) not in reversed_state_action_pairs[index:]:
                returns[element[0]][element[1]][element[2]].append(G)
                Q[element[0]][element[1]][element[2]] = np.mean(returns[element[0]][element[1]][element[2]])
                policy[element[0]][element[1]] = get_action(Q, epsilon, element[0], element[1])
            index += 1

    # obtain the best action from Q for the current state
    choice = policy[robot.pos[0]][robot.pos[1]]
    direction = ['n', 'e', 's', 'w'][choice]
    while robot.orientation != direction:
        robot.rotate('r')
    robot.move()
