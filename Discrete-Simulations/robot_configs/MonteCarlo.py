import numpy as np
import random
INIT = True


def robot_epoch(robot, gamma=0.8, theta=0.001, certainty=0.3):
    global INIT
    inputgrid = robot.grid.cells
    rows = robot.grid.n_rows
    cols = robot.grid.n_cols
    r = np.zeros((cols, rows))
    discount = 0.8
    for i in range(cols):
        for j in range(rows):
            if inputgrid[i, j] < -2:  # if the robot is on the tile
                r[i, j] = 0  # we consider the tile clean
            elif inputgrid[i, j] == 3:  # if the tile is a death tile
                r[i, j] = -1  # we consider it as a wall
            else:
                r[i, j] = inputgrid[i, j]

    # Initialise random policy
    policy = np.zeros((cols, rows))
    tiles = [rows][cols]
    for i in range(cols):
        for j in range(rows):
            if r[i, j] < 0:  # if the tile is a wall or obstacle, we don't assign an action
                continue
            else:  # we will check the surrounding tiles
                if r[i, j - 1] == 0 or 1:
                    tiles[i][j].append('n')
                if r[i - 1, j] == 0 or 1:
                    tiles[i][j].append('w')
                if r[i + 1, j] == 0 or 1:
                    tiles[i][j].append('e')
                if r[i, j + 1] == 0 or 1:
                    tiles[i][j].append('s')
                policy[i][j] = (random.choice(tiles[i][j]), 0)

    for x in range(3):
        # Select random action from state
        random_action = random.choice(tiles[robot.pos[0]][robot.pos[1]])
        stuck = False
        # Keep track of return value for each state-action combination
        Q_list = np.zeros(rows, cols, 4)
        i, j = robot.pos
        policy[i][j] = random_action
        states_seen = [(i, j)]
        while True:
            # Follow policy
            if policy[i][j] == 'n': j -= 1
            if policy[i][j] == 'e': i += 1
            if policy[i][j] == 's': j += 1
            if policy[i][j] == 'w': i -= 1
            next_state = (i, j)
            if next_state in states_seen:
                break
            states_seen.append(next_state)

        for state in states_seen:
            # Store return values in Q list
            Q_list[state[0]][state[1]][policy[i][j]].append(calc_return(state, discount))
            #TODO implement function for return value's using discount factor

    # Take averages of Q list and update policy greedily
    for i in range(cols):
        for j in range(rows):
            for a in ['n', 'e', 's', 'w']:
                if np.mean(Q_list[i][j][a]) > calc_return((i,j), discount):
                    policy[i][j] = a

    # Take the best action corresponding to the policy
    direction = policy[robot.pos[0]][robot.pos[1]]
    while robot.orientation != direction:
        robot.rotate('r')

    robot.move()
