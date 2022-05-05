import numpy as np
import random


def robot_epoch(robot, gamma=0.9, theta=0.01, certainty=0.8):
    inputgrid = robot.grid.cells
    rows = robot.grid.n_rows
    cols = robot.grid.n_cols
    i_position, j_position = robot.pos
    v_values = np.zeros((cols, rows))
    for i in range(cols):
        for j in range(rows):
            if inputgrid[i, j] < -2:  # if the robot is on the tile
                v_values[i, j] = 0  # we consider the tile clean
            elif inputgrid[i, j] == 3:  # if the tile is a death tile
                v_values[i, j] = -1  # we consider it as a wall
            else:
                v_values[i, j] = inputgrid[i, j]
    # 0 up, 1 right, 2 down, 3 left
    policy = np.random.randint(0, 4, size=(cols, rows))

    value_V = 0

    policy_stable = False
    while not policy_stable:
        iteration = 0
        delta = 999999
        while delta >= theta:
            iteration += 1
            for i in range(cols):
                for j in range(rows):
                    if v_values[i, j] < 0:  # if the tile is a wall or obstacle, we are never on it, so we don't update it
                        continue
                    tiles = [v_values[i, j - 1], v_values[i + 1, j], v_values[i, j + 1], v_values[i - 1, j]]
                    tiles = [i if i > 0 else 0 for i in tiles]
                    choice = policy[i, j]
                    correctmove = tiles[choice]
                    wrongmove = (sum(tiles) - correctmove) / 3
                    return_value = correctmove * certainty + wrongmove * (1 - certainty)
                    V_value = return_value * gamma ** iteration
                    v_values[i, j] = v_values[i, j] + V_value  # function for value iteration

            prevtotalvalueV = value_V
            value_V = sum([sum(i) for i in v_values]) / (rows * cols)
            delta = abs(value_V - prevtotalvalueV)

        # print(v_values)
        policy_stable = True
        for i in range(cols):
            for j in range(rows):
                if r[i, j] < 0:  # if the tile is a wall or obstacle, we are never on it, so we don't update it
                    continue

                tiles = [v_values[i, j - 1], v_values[i + 1, j], v_values[i, j + 1], v_values[i - 1, j]]
                tiles = [i if i > 0 else 0 for i in tiles]
                action = policy[i, j]
                max_action = np.argmax(tiles)
                policy[i, j] = max_action

                if action != max_action:
                    policy_stable = False

    direction = ['n', 'e', 's', 'w'][policy[i_position, j_position]]
    while robot.orientation != direction:
        robot.rotate('r')
    if random.randint(1, 5) != 1:
        robot.move()
    else:
        for i in range(random.randint(1, 3)):
            robot.rotate('r')
        robot.move()
