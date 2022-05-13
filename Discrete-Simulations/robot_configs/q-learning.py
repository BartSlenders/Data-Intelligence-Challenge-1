import numpy as np

# Choose a random, non-terminal state (white square) for the agent to begin this new episode.
# Choose an action (move up, right, down, or left) for the current state. Actions will be chosen using an epsilon greedy algorithm. This algorithm will usually choose the most promising action for the agent, but it will occasionally choose a less promising option in order to encourage the agent to explore the environment.
# Perform the chosen action, and transition to the next state (i.e., move to the next location).
# Receive the reward for moving to the new state, and calculate the temporal difference.
# Update the Q-value for the previous state and action pair.
# If the new (current) state is a terminal state, go to #1. Else, go to #2.

#define an epsilon greedy algorithm that will choose which action to take next (i.e., where to move next)
def get_next_action(current_row, current_col, epsilon, q_values):
    #if a randomly chosen value between 0 and 1 is less than epsilon,
    #then choose the most promising value from the Q-table for this state.
    if np.random.random() < epsilon:
        return np.argmax(q_values[current_row, current_col])
    else: #choose a random action
       return np.random.randint(4)

#define a function that will get the next location based on the chosen action
def get_next_location(current_row, current_col, action, actions, rows, cols):
    new_row = current_row
    new_col = current_col
    if actions[action] == 'n' and current_row > 0:
        new_row -= 1
    elif actions[action] == 'e' and current_col < cols - 1:
        new_col += 1
    elif actions[action] == 's' and current_row < rows - 1:
        new_row += 1
    elif actions[action] == 'w' and current_col > 0:
        new_col -= 1
    return new_row, new_col

def robot_epoch(robot):
    inputgrid = robot.grid.cells
    rows = robot.grid.n_rows
    cols = robot.grid.n_cols
    i_position, j_position = robot.pos
    filthy_tiles = 0
    r_values = np.zeros((cols, rows))
    for i in range(cols):
        for j in range(rows):
            current_tile = inputgrid[i, j]
            if current_tile < -2:  # if the robot is on the tile
                r_values[i, j] = 0  # we consider the tile clean
            elif current_tile == 3:  # if the tile is a death tile
                r_values[i, j] = -1  # we consider it as a wall
            else:
                r_values[i, j] = current_tile
                if current_tile > 0:
                    filthy_tiles += 1

    q_values = np.zeros((rows, cols, 4))  # 4 actions
    actions = ['n', 'w', 'e', 's']
    # define training parameters
    epsilon = 0.7  # the percentage of time when we should take the best action (instead of a random action)
    gamma = 0.9  # discount factor for future rewards
    learning_rate = 0.9  # alpha in the pseudocode
    max_steps = rows*cols/2
    # run through 1000 training episodes
    for episode in range(1000):
        # get the starting location for this episode
        row, col = robot.pos
        # continue taking actions (i.e., moving) until we reach a terminal state
        # (i.e., until we reach the item packaging area or crash into an item storage location)
        count = 0
        r = np.copy(r_values)
        done_cleaning = False
        cleaned_tiles = 0
        while not done_cleaning:
            count += 1
            old_row, old_column = row, col  # store the old row and column indexes
            # choose which action to take (i.e., where to move next)
            action = get_next_action(row, col, epsilon, q_values)
            # perform the chosen action, and transition to the next state (i.e., move to the next location)
            row, column = get_next_location(row, col, action, actions, rows, cols)
            # receive the reward for moving to the new state, and calculate the temporal difference
            reward = r[row, col]
            if reward <= -1:  # this means we hit a wall, so we need to make a different move, shouldn't happen often
                continue
            old_q_value = q_values[old_row, old_column, action]
            temporal_difference = reward + (gamma * np.max(q_values[row, col])) - old_q_value
            # update the Q-value for the previous state and action pair
            new_q_value = old_q_value + (learning_rate * temporal_difference)
            q_values[old_row, old_column, action] = new_q_value
            if reward == 1:
                count = 0
                r[row, col] = 0
                cleaned_tiles += 1
                if cleaned_tiles == filthy_tiles:
                    done_cleaning = True
            elif count > max_steps:
                break
    action = get_next_action(j_position, i_position, epsilon, q_values)
    direction = actions[action]
    while robot.orientation != direction:
        robot.rotate('r')
    print('move')
    robot.move()


    # Q(st, at) ← Q(st, at) + α[rt + γ max_a Q(st+1, a) − Q(st, at)]
