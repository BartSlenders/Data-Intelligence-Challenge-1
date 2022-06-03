from continuous import parse_config, Robot
import time
import matplotlib.pyplot as plt

plt.ion()

grid = parse_config('example.grid')
grid.spawn_robots([Robot(id=1, battery_drain_p=0.1, battery_drain_lam=10),
                   Robot(id=2, battery_drain_p=0.1, battery_drain_lam=10)],
                  [(0, 0), (1, 2)])

while True:
    grid.plot_grid()
    # Stop simulation if all robots died:
    if all([not robot.alive for robot in grid.robots]):
        break
    for robot in grid.robots:
        # To avoid deadlocks, only try to move alive robots:
        if robot.alive:
            if not robot.move(p_random=0.05):
                robot.direction_vector = (0.1, 0.1)
grid.plot_grid()
time.sleep(3)
