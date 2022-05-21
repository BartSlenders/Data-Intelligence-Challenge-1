import numpy as np
import random


def calc_return(states, gamma, rows, cols, r):
    updated_returns = np.zeros((cols, rows))
    for state in states:
        remaining_states = states[states.index(state):]
        remaining_states_value = 0
        for remaining_state in remaining_states:
            remaining_states_value += r[remaining_state[0]][remaining_state[1]]
        return_value = r[state[0]][state[1]] + gamma * remaining_states_value
        updated_returns[state[0]][state[1]] = return_value
    return updated_returns


def robot_epoch(robot, iterations_per_evaluation=100, gamma=0.8, epsilon=0.1, epochs=100):
    inputgrid = robot.grid.cells
    rows = robot.grid.n_rows
    cols = robot.grid.n_cols
    r = np.zeros((cols, rows))
    for i in range(cols):
        for j in range(rows):
            if inputgrid[i, j] < -2:  # if the robot is on the tile
                r[i, j] = 0  # we consider the tile clean
            elif inputgrid[i, j] == 3:  # if the tile is a death tile
                r[i, j] = -1  # we consider it as a wall
            else:
                r[i, j] = inputgrid[i, j]

    # Initialise random policy
    policy = np.full((cols, rows), 'x')
    Q = np.zeros((cols, rows))
    actions_per_state = np.full((cols, rows), None)
    for i in range(1,cols-1):
        for j in range(1,rows-1):
            actions = []
            # we will check the surrounding tiles
            if r[i, j - 1] == 0 or r[i, j - 1] == 1:
                actions.append('n')
            if r[i + 1, j] == 0 or r[i + 1, j] == 1:
                actions.append('e')
            if r[i, j + 1] == 0 or r[i, j + 1] == 1:
                actions.append('s')
            if r[i - 1, j] == 0 or r[i - 1, j] == 1:
                actions.append('w')
            policy[i][j] = random.choice(actions)
            actions_per_state[i, j] = actions

    # Keep track of return value for each state-action combination
    Q_list = np.full((cols, rows), {'n': 0, 'e': 0, 's': 0, 'w': 0})
    for evaluations in range(epochs):
        for x in range(iterations_per_evaluation):
            stuck = False
            states = []
            # Select random action from state
            for p in range(1, cols-1):
                for q in range(1, rows-1):
                    if r[p][q]>=0:
                        states.append((p,q))
            i, j = random.choice(states)
            random_action = random.choice(actions_per_state[i, j])
            policy[i][j] = random_action
            states_seen = [(i, j)]
            while not stuck:
                # Follow policy
                if policy[i][j] == 'n':
                    j -= 1
                if policy[i][j] == 'e':
                    i += 1
                if policy[i][j] == 's':
                    j += 1
                if policy[i][j] == 'w':
                    i -= 1
                next_state = (i, j)
                states_seen.append(next_state)
                if next_state in states_seen:
                    stuck = True

            # Store return values in Q list
            update_Q = calc_return(states_seen, gamma, rows, cols, r)
            for state in states_seen:
                Q_list[state[0], state[1]][policy[state[0]][state[1]]] = update_Q[state[0]][state[1]]

        # Take averages of Q list and update policy greedily
        for i in range(cols):
            for j in range(rows):
                for a in ['n', 'e', 's', 'w']:
                    if np.mean(Q_list[i, j][a]) > Q[i][j]:
                        policy[i][j] = a
                        Q[i][j] = np.mean(Q_list[i, j][a])

    # Take the best action corresponding to the epsilon-greedy policy
    if np.random.uniform(0, 1) < epsilon:
        # choose a random action
        direction = random.choice(['n', 'e', 's', 'w'])
    else:
        # take the action according to the policy
        direction = policy[robot.pos[0]][robot.pos[1]]

    while robot.orientation != direction:
        robot.rotate('r')

    robot.move()
