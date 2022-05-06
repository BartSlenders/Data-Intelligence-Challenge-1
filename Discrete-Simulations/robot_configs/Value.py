import numpy as np
import random


def robot_epoch(robot, gamma=0.3, theta=3, certainty=0.8):
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
    V = np.zeros((cols, rows))


    delta = 9999
    count = 0
    position_bestmove = -99
    while delta > theta or position_bestmove <= 0:  # if V doesn't change, we stop
        delta = 0
        # we start by looping over all tiles
        for i in range(cols):
            for j in range(rows):
                if r[i, j] < 0:  # if the tile is a wall or obstacle, we are never on it, so we don't update it
                    continue
                else:  # we will check the surrounding tiles
                    old_v = V[i, j]
                    tiles = [V[i, j - 1], V[i - 1, j], V[i + 1, j], V[i, j + 1]]
                    best_move = max(tiles)
                    wrong_move = (sum(tiles) - best_move) / 3
                    V[i, j] = certainty * (best_move * gamma + r[i, j]) + \
                        (1 - certainty) * (wrong_move * gamma + r[i, j])
                    if i == i_position and j == j_position:
                        position_bestmove = best_move
                    delta = max(delta, abs(old_v - V[i, j]))
        if count > 1000:
            print("We can't find a dirty tile")
            break
        count += 1
    # print(count)
    tiles = [V[i_position, j_position - 1], V[i_position - 1, j_position],
             V[i_position + 1, j_position], V[i_position, j_position + 1]]
    best_move = -99
    for i in range(4):
        if tiles[i] > best_move:
            best_move = tiles[i]
            position = i  # remember, format of tiles is up, left, right, down
    direction = ['n', 'w', 'e', 's'][position]

    while robot.orientation != direction:
        robot.rotate('r')
    if random.randint(1, 5) != 1:
        robot.move()
    else:
        for i in range(random.randint(1, 3)):
            robot.rotate('r')
        robot.move()
