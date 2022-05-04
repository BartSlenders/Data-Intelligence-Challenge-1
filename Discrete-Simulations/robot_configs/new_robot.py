import numpy as np
import random

def robot_epoch(robot, gamma = 0.3, theta = 3):
    inputgrid = robot.grid.cells
    rows = robot.grid.n_rows
    cols = robot.grid.n_cols
    i_position, j_position = robot.pos
    V = np.zeros((cols,rows))
    for i in range(cols):
        for j in range(rows):
            # value =
            # grid[i,j] = inputgrid[i,j] == -1
            if inputgrid[i,j] < -2:  # if the robot is on the tile
                V[i,j] = 0  # we consider the tile clean
            elif inputgrid[i,j] == 3:  # if the tile is a death tile
                V[i,j] = -1  # we consider it as a wall
            else:
                V[i,j] = inputgrid[i,j]
    correctprob = 0.8
    wrongprob = 1 - correctprob
    # initialize V
    delta = -999
    prev_delta = -9999
    count = 0
    position_bestmove = -99
    while delta - prev_delta > theta or position_bestmove <= 0:  # if V doesn't change, we stop
        prev_delta = delta
        count += 1
        # we start by looping over all tiles
        for i in range(cols):
            for j in range(rows):
                current_tile = V[i,j]
                if current_tile < 0:  # if the tile is a wall or obstacle, we are never on it, so we don't update it
                    continue
                else:  # we will check the surrounding tiles, format of tiles is up, left, right, down
                    tiles = [V[i,j-1], V[i-1,j], V[i+1, j], V[i,j+1]]
                    tiles = [i if i > 0 else 0 for i in tiles]
                    bestmove = max(tiles)
                    wrongmove = (sum(tiles)-bestmove)/3
                    return_value = bestmove * correctprob + wrongmove * wrongprob
                    V_value = return_value * gamma**count
                    V[i,j] = current_tile + V_value# function for value iteration
                    if i == i_position and j == j_position:
                        position_bestmove = bestmove
        # at the end of an iteration we update V
        delta = sum([sum(i)for i in V])/(rows*cols)
        if count > 1000:
            print("We can't find a dirty tile")
            break
    # print(count)
    tiles = [V[i_position, j_position - 1], V[i_position - 1, j_position],
             V[i_position + 1, j_position], V[i_position, j_position + 1]]
    bestmove = -99
    print(tiles)
    for i in range(4):
        if tiles[i] > bestmove:
            bestmove = tiles[i]
            position = i  # remember, format of tiles is up, left, right, down
    # print(i)
    direction = ['n', 'w', 'e', 's'][position]
    # print(robot.orientation)

    while robot.orientation != direction:
        robot.rotate('r')
    if random.randint(1,5) != 1:
        robot.move()
    else:
        for i in range(random.randint(1,3)):
            robot.rotate('r')
        robot.move()




