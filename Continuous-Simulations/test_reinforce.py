from continuous import parse_config, Robot
import time
import matplotlib.pyplot as plt
from reinforce import robot_epoch

plt.ion()

grid = parse_config('random_house_0.grid')
grid.spawn_robots([Robot(id=1, battery_drain_p=0.2, battery_drain_lam=10)],
                  [(0, 0), (1, 6), (2, 14)])

while True:
    grid.plot_grid()
    # Stop simulation if all robots died:
    if all([not robot.alive for robot in grid.robots]):
        break
    for robot in grid.robots:
        # To avoid deadlocks, only try to move alive robots:
        if robot.alive:
            robot_epoch(robot=robot)
cleanpercent, batteryleft = grid.evaluate()
print('the floor is', cleanpercent, 'percent clean')
print('there is', batteryleft, 'of the battery left')
grid.plot_grid()
time.sleep(3)
