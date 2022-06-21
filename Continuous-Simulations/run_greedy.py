from continuous import parse_config, Robot, SimGrid
import time
import matplotlib.pyplot as plt
import random
plt.ion()

grid = parse_config('random_house_0.grid', divideby=1)
grid.spawn_robots([Robot(id=1, battery_drain_p=1, battery_drain_lam=0.5)],
                   [(0, 0)])

max_filthy = len(grid.filthy)
max_goals = len(grid.goals)

Potential_moves = [(0.85,0), (0,0.85), (-0.85,0), (0,-0.85),
                   (0.46, 0.46), (-0.46, 0.46), (0.46, -0.46), (-0.46, -0.46),
                    (1.05,0), (0,1.05), (-1.05,0), (0,-1.05),
                   (1.06,1.06),(-1.06,1.06),(1.06,-1.06),(-1.06,-1.06)]

cleanpercent = 0
while True:
    grid.plot_grid('Greedy', str(round(cleanpercent, 2)))
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
                    break
            elif value == reward:
                bestmove.append(move)
        move = random.sample(bestmove, 1)[0]
        #print(move)
        move = (move[0], move[1])
        robot.direction_vector = move
        robot.move()
        cleanpercent, batteryleft = grid.evaluate(max_filthy, max_goals)

cleanpercent, batteryleft = grid.evaluate(max_filthy, max_goals)
print('the floor is', cleanpercent, 'percent clean')
print('there is', batteryleft, 'of the battery left')
grid.plot_grid('Greedy', str(round(cleanpercent, 2)))
time.sleep(3)
