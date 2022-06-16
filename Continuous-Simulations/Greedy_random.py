from continuous import parse_config, Robot, SimGrid
import time
import matplotlib.pyplot as plt
import random
from copy import deepcopy

plt.ion()

grid = parse_config('random_house_0.grid', divideby=2)
grid.spawn_robots([Robot(id=1, battery_drain_p=0.2, battery_drain_lam=10)],
                   [(2, 14)])

Potential_moves = [(0.85,0), (0,0.85), (-0.85,0), (0,-0.85),
                   (0.46, 0.46), (-0.46, 0.46), (0.46, -0.46), (-0.46, -0.46),
                    (1.05,0), (0,1.05), (-1.05,0), (0,-1.05),
                   (1.06,1.06),(-1.06,1.06),(1.06,-1.06),(-1.06,-1.06)]

while True:
    grid.plot_grid()
    # Stop simulation if all robots died:
    if all([not robot.alive for robot in grid.robots]):
        break
    robot = grid.robots[0]
    # To avoid deadlocks, only try to move alive robots:
    if robot.alive:
        test = SimGrid(grid)
        reward = -10
        for move in Potential_moves:
            value = test.reward(move)
            if value>reward:
                reward = value
                bestmove = [move]
                if value > 1:
                    print(reward)
                    break
            elif value == reward:
                bestmove.append(move)
        move = random.sample(bestmove, 1)[0]
        print(move)
        move = (move[0]/10, move[1]/10)
        robot.direction_vector = move
        robot.move()

cleanpercent, batteryleft = grid.evaluate()
print('the floor is', cleanpercent, 'percent clean')
print('there is', batteryleft, 'of the battery left')
grid.plot_grid()
time.sleep(3)
