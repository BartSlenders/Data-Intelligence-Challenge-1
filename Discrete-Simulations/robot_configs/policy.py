import numpy as np


def robot_epoch(robot, gamma=0.95, theta=0.001, certainty=0.255):
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

    # 0 up, 1 right, 2 down, 3 left
    policy = np.random.randint(0, 4, size=(cols, rows))
    V = np.zeros((cols, rows))

    delta = theta

    policy_stable = False
    while not policy_stable:

        while delta >= theta:
            delta = 0
            for i in range(cols):
                for j in range(rows):
                    if r[i, j] < 0:  # if the tile is a wall or obstacle, we are never on it, so we don't update it
                        continue

                    old_v = V[i, j]
                    choice = policy[i, j]

                    tiles = [V[i, j - 1], V[i + 1, j], V[i, j + 1], V[i - 1, j]]
                    correct_move = tiles[choice]
                    wrong_move = (sum(tiles) - correct_move) / 3
                    V[i, j] = certainty * (correct_move * gamma + r[i, j]) + \
                        (1 - certainty) * (wrong_move * gamma + r[i, j])

                    delta = max(delta, abs(old_v - V[i, j]))

        policy_stable = True
        for i in range(cols):
            for j in range(rows):
                if r[i, j] < 0:  # if the tile is a wall or obstacle, we are never on it, so we don't update it
                    continue

                tiles = [V[i, j - 1], V[i + 1, j], V[i, j + 1], V[i - 1, j]]
                actions = [certainty * (move * gamma + r[i, j]) +
                           (1 - certainty) * ((sum(tiles) - move)/3 * gamma + r[i, j]) for move in tiles]

                best_action = np.argmax(actions)
                prev_action = policy[i, j]
                policy[i, j] = best_action

                if prev_action != best_action:
                    policy_stable = False

    direction = ['n', 'e', 's', 'w'][policy[i_position, j_position]]
    while robot.orientation != direction:
        robot.rotate('r')
    robot.move()
