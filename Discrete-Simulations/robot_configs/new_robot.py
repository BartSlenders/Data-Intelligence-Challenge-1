import numpy as np

def robot_epoch(robot):
    inputgrid = robot.grid.cells
    rows = robot.grid.n_rows
    cols = robot.grid.n_cols
    i_position, j_position = robot.pos
    grid = np.zeros((cols,rows))
    # print(robot.grid.__dict__)
    for i in range(cols):
        for j in range(rows):
            # value =
            # grid[i,j] = inputgrid[i,j] == -1
            if inputgrid[i,j] < -2:  # if the robot is on the tile
                grid[i,j] = 0  # we consider the tile clean
            elif inputgrid[i,j] == 3:  # if the tile is a death tile
                grid[i,j] = -1  # we consider it as a wall
            else:
                grid[i,j] = inputgrid[i,j]
    # initialize V
    V = -999
    prev_V = -9999
    count = 0
    gamma = 0.3 # gamma is always below 1, and is to calculate return values
    theta = 3
    position_bestmove = -99
    while V - prev_V > theta or position_bestmove <= 0:  # if V doesn't change, we stop
        prev_V = V
        count += 1
        # we start by looping over all tiles
        for i in range(cols):
            for j in range(rows):
                current_tile = grid[i,j]
                if current_tile < 0:  # if the tile is a wall or obstacle, we are never on it, so we don't update it
                    continue
                else:  # we will check the surrounding tiles, format of tiles is up, left, right, down
                    tiles = [grid[i,j-1], grid[i-1,j], grid[i+1, j], grid[i,j+1]]
                    bestmove = max(tiles)
                    return_value = bestmove * gamma**count
                    grid[i,j] = current_tile + return_value  # function for value iteration
                    if i == i_position and j == j_position:
                        position_bestmove = bestmove
        # at the end of an iteration we update V
        V = sum([sum(i)for i in grid])/(rows*cols)
        if count > 1000:
            print("We can't find a dirty tile")
            break
    # print(count)
    tiles = [grid[i_position, j_position - 1], grid[i_position - 1, j_position],
             grid[i_position + 1, j_position], grid[i_position, j_position + 1]]
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
    # print('beep')
    # print(grid)
    robot.move()






    # if 1.0 in list(possible_tiles.values()) or 2.0 in list(possible_tiles.values()):
    #     # If we can reach a goal tile this move:
    #     if 2.0 in list(possible_tiles.values()):
    #         move = list(possible_tiles.keys())[list(possible_tiles.values()).index(2.0)]
    #     # If we can reach a dirty tile this move:
    #     elif 1.0 in list(possible_tiles.values()):
    #         # Find the move that makes us reach the dirty tile:
    #         move = list(possible_tiles.keys())[list(possible_tiles.values()).index(1.0)]
    #     else:
    #         assert False
    #     # Find out how we should orient ourselves:
    #     new_orient = list(robot.dirs.keys())[list(robot.dirs.values()).index(move)]
    #     # Orient ourselves towards the dirty tile:
    #     while new_orient != robot.orientation:
    #         # If we don't have the wanted orientation, rotate clockwise until we do:
    #         # print('Rotating right once.')
    #         robot.rotate('r')
    #     # Move:
    #     robot.move()
    # # If we cannot reach a dirty tile:
    # else:
    #     # If we can no longer move:
    #     while not robot.move():
    #         # Check if we died to avoid endless looping:
    #         if not robot.alive:
    #             break
    #         # Decide randomly how often we want to rotate:
    #         times = random.randrange(1, 4)
    #         # Decide randomly in which direction we rotate:
    #         if random.randrange(0, 2) == 0:
    #             # print(f'Rotating right, {times} times.')
    #             for k in range(times):
    #                 robot.rotate('r')
    #         else:
    #             # print(f'Rotating left, {times} times.')
    #             for k in range(times):
    #                 robot.rotate('l')
    # #print('Historic coordinates:', [(x, y) for (x, y) in zip(robot.history[0], robot.history[1])])
