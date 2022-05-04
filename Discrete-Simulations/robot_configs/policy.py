import numpy as np


def robot_epoch(robot, gamma=0.9, theta=0.001, certainty=0.8):
    inputgrid = robot.grid.cells
    rows = robot.grid.n_rows
    cols = robot.grid.n_cols
    i_position, j_position = robot.pos
    r = np.zeros((cols, rows))
    for i in range(cols):
        for j in range(rows):
            if inputgrid[i, j] < -2:  # if the robot is on the tile
                r[i, j] = 0  # we consider the tile clean
            elif inputgrid[i, j] == 3:  # if the tile is a death tile
                r[i, j] = -1  # we consider it as a wall
            else:
                r[i, j] = inputgrid[i, j]

    # 1 up, 2 right, 3 down, 4 left
    policy = np.random.randint(1, 5, size=(cols, rows))
    v_values = np.zeros((cols, rows))

    delta = 999999
    iteration = 0

    policy_stable = False
    while not policy_stable:

        while delta >= theta:
            iteration = 1
            delta = 0
            for i in range(cols):
                for j in range(rows):
                    if r[i, j] < 0:  # if the tile is a wall or obstacle, we are never on it, so we don't update it
                        continue

                    v = v_values[i, j]
                    choice = policy[i, j]
                    if choice == 1:
                        v_values[i, j] = certainty * (v_values[i, j - 1] * gamma ** iteration + r[i, j]) + (
                                1 - certainty) / 3 * \
                                         (r[i, j] * 3 + gamma ** iteration *
                                          (v_values[i, j + 1] + v_values[i - 1, j] + v_values[i + 1, j]))

                    elif choice == 2:
                        v_values[i, j] = certainty * (v_values[i + 1, j] * gamma ** iteration + r[i, j]) + \
                                         (1 - certainty) / 3 * \
                                         (r[i, j] * 3 + gamma ** iteration * (
                                                 v_values[i, j + 1] + v_values[i - 1, j] + v_values[i, j - 1]))

                    elif choice == 3:
                        v_values[i, j] = certainty * (v_values[i, j + 1] * gamma ** iteration + r[i, j]) + \
                                         (1 - certainty) / 3 * \
                                         (r[i, j] * 3 + gamma ** iteration * (
                                                 v_values[i, j - 1] + v_values[i - 1, j] + v_values[i + 1, j]))

                    elif choice == 4:
                        v_values[i, j] = certainty * (v_values[i - 1, j] * gamma ** iteration + r[i, j]) + \
                                         (1 - certainty) / 3 * \
                                         (r[i, j] * 3 + gamma ** iteration * (
                                                 v_values[i, j + 1] + v_values[i + 1, j] + v_values[i, j - 1]))

                    delta = max(delta, abs(v - v_values[i, j]))

        # print(v_values)
        policy_stable = True
        for i in range(cols):
            for j in range(rows):
                if r[i, j] < 0:  # if the tile is a wall or obstacle, we are never on it, so we don't update it
                    continue

                action = policy[i, j]
                actions = []
                up = certainty * (v_values[i, j - 1] * gamma ** iteration + r[i, j]) + (1 - certainty) / 3 * \
                     (r[i, j] * 3 + gamma ** iteration *
                      (v_values[i, j + 1] + v_values[i - 1, j] + v_values[i + 1, j]))
                actions.append(up)

                right = certainty * (v_values[i + 1, j] * gamma ** iteration + r[i, j]) + \
                        (1 - certainty) / 3 * \
                        (r[i, j] * 3 + gamma ** iteration * (
                                v_values[i, j + 1] + v_values[i - 1, j] + v_values[i, j - 1]))
                actions.append(right)

                down = certainty * (v_values[i, j + 1] * gamma ** iteration + r[i, j]) + \
                       (1 - certainty) / 3 * \
                       (r[i, j] * 3 + gamma ** iteration * (
                               v_values[i, j - 1] + v_values[i - 1, j] + v_values[i + 1, j]))
                actions.append(down)

                left = certainty * (v_values[i - 1, j] * gamma ** iteration + r[i, j]) + \
                       (1 - certainty) / 3 * \
                       (r[i, j] * 3 + gamma ** iteration * (
                               v_values[i, j + 1] + v_values[i + 1, j] + v_values[i, j - 1]))
                actions.append(left)

                max_action = np.argmax(actions) + 1

                policy[i, j] = max_action

                if action != max_action:
                    policy_stable = False

                # if policy_stable == True:
                #     print(i, j, actions, v_values[i,j+1], v_values[i+1,j])

    direction = ['n', 'e', 's', 'w'][policy[i_position, j_position] - 1]
    # print(direction)
    while robot.orientation != direction:
        robot.rotate('r')
    robot.move()
