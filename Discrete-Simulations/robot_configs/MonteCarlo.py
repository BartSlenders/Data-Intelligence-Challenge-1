import numpy as np
import random


def calc_return(states, discount, rows, cols, r):
    updated_returns = np.zeros((rows, cols))
    for state in states:
        remaining_states = states[states.index(state):]
        remaining_states_value = 0
        for remaining_state in remaining_states:
            remaining_states_value += r[remaining_state[0]][remaining_state[1]]
        return_value = r[state[0]][state[1]] + discount * remaining_states_value
        updated_returns[state[0]][state[1]] = return_value
    return updated_returns


def robot_epoch(robot, iterations_per_evaluation=3, discount=0.8, epsilon=0.3, epochs=10):
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
    actions_per_state = np.full((cols, rows), [])
    for i in range(cols):
        for j in range(rows):
            actions = []
            if r[i, j] < 0:  # if the tile is a wall or obstacle, we don't assign an action
                continue
            else:  # we will check the surrounding tiles
                if r[i, j - 1] == 0 or 1:
                    actions.append('n')
                if r[i - 1, j] == 0 or 1:
                    actions.append('w')
                if r[i + 1, j] == 0 or 1:
                    actions.append('e')
                if r[i, j + 1] == 0 or 1:
                    actions.append('s')
                policy[i][j] = (random.choice(actions))
                actions_per_state[i, j] = actions

    for evaluations in range(epochs):
        for x in range(iterations_per_evaluation):
            # Select random action from state
            random_action = random.choice(actions_per_state[robot.pos[0], robot.pos[1]])
            stuck = False
            # Keep track of return value for each state-action combination
            Q_list = np.full((cols, rows), {'n': 0, 'e': 0, 's': 0, 'w': 0})
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

            # Store return values in Q list
            update_Q = calc_return(states_seen, discount, rows, cols, r)
            for reward in update_Q:
                Q_list[i, j][policy[i][j]] = reward[i][j]

        # Take averages of Q list and update policy greedily
        for i in range(cols):
            for j in range(rows):
                for a in ['n', 'e', 's', 'w']:
                    if np.mean(Q_list[i, j][a]) > Q[i][j]:
                        policy[i][j] = a
                        Q[i][j] = np.mean(Q_list[i, j][a])

        # Take the best action corresponding to the epsilon-greedy policy
        if np.random.uniform(0, 1) < epsilon:
            # take the action according to the policy
            direction = policy[robot.pos[0]][robot.pos[1]]
        else:
            # choose a random action
            direction = np.random(['n', 'e', 's', 'w'])

        while robot.orientation != direction:
            robot.rotate('r')

        robot.move()
