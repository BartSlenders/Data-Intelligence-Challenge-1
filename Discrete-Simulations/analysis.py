import seaborn as sns

# Import our robot algorithm to use in this simulation:
from robot_configs.MonteCarlo import robot_epoch as mc
from robot_configs.qlearning import robot_epoch as ql
from robot_configs.sarsa import robot_epoch as sa
from robot_configs.policy import robot_epoch as pc
from robot_configs.Value import robot_epoch as va
import pickle
from environment import Robot
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd




def runMC(gamma=0.7, epsilon=0.05, episodes=500, steps=3, grid_file='house.grid'):
    # Cleaned tile percentage at which the room is considered 'clean':
    if grid_file == "example-random-house-3.grid":
        stopping_criteria = 82.5
    elif grid_file == "death.grid":
        stopping_criteria = 87.5
    else:
        stopping_criteria = 100
    deaths = 0
    # Open the grid file.
    # (You can create one yourself using the provided editor).
    with open(f'grid_configs/{grid_file}', 'rb') as f:
        grid = pickle.load(f)
    # Calculate the total visitable tiles:
    n_total_tiles = (grid.cells >= 0).sum()
    # Spawn the robot at (1,1) facing north with battery drainage enabled:
    robot = Robot(grid, (1, 1), orientation='n', battery_drain_p=1, battery_drain_lam=0.5)
    # Keep track of the number of robot decision epochs:
    n_epochs = 0
    while True:
        n_epochs += 1
        # Do a robot epoch (basically call the robot algorithm once):
        mc(robot, gamma=gamma, epsilon=epsilon, episodes=episodes, steps=steps)
        # Stop this simulation instance if robot died :( :
        if not robot.alive:
            deaths += 1
            break
        # Calculate some statistics:
        clean = (grid.cells == 0).sum()
        dirty = (grid.cells >= 1).sum()
        goal = (grid.cells == 2).sum()
        # Calculate the cleaned percentage:
        clean_percent = (clean / (dirty + clean)) * 100
        # See if the room can be considered clean, if so, stop the simulaiton instance:
        if clean_percent >= stopping_criteria and goal == 0:
            break
        # Calculate the effiency score:
        moves = [(x, y) for (x, y) in zip(robot.history[0], robot.history[1])]
        u_moves = set(moves)
        n_revisted_tiles = len(moves) - len(u_moves)
        efficiency = (100 * n_total_tiles) / (n_total_tiles + n_revisted_tiles)
    return clean_percent, efficiency


def runQL(gamma=0.5, epsilon=0.2, alpha=0.6, episodes=300, steps=300, grid_file='house.grid'):
    # Cleaned tile percentage at which the room is considered 'clean':
    if grid_file == "example-random-house-3.grid":
        stopping_criteria = 82.5
    elif grid_file == "death.grid":
        stopping_criteria = 87.5
    else:
        stopping_criteria = 100
    deaths = 0
    # Open the grid file.
    # (You can create one yourself using the provided editor).
    with open(f'grid_configs/{grid_file}', 'rb') as f:
        grid = pickle.load(f)
    # Calculate the total visitable tiles:
    n_total_tiles = (grid.cells >= 0).sum()
    # Spawn the robot at (1,1) facing north with battery drainage enabled:
    robot = Robot(grid, (1, 1), orientation='n', battery_drain_p=1, battery_drain_lam=0.5)
    # Keep track of the number of robot decision epochs:
    n_epochs = 0
    while True:
        n_epochs += 1
        # Do a robot epoch (basically call the robot algorithm once):
        ql(robot, gamma=gamma, epsilon=epsilon, alpha=alpha, episodes=episodes, steps=steps)
        # Stop this simulation instance if robot died :( :
        if not robot.alive:
            deaths += 1
            break
        # Calculate some statistics:
        clean = (grid.cells == 0).sum()
        dirty = (grid.cells >= 1).sum()
        goal = (grid.cells == 2).sum()
        # Calculate the cleaned percentage:
        clean_percent = (clean / (dirty + clean)) * 100
        # See if the room can be considered clean, if so, stop the simulaiton instance:
        if clean_percent >= stopping_criteria and goal == 0:
            break
        # Calculate the effiency score:
        moves = [(x, y) for (x, y) in zip(robot.history[0], robot.history[1])]
        u_moves = set(moves)
        n_revisted_tiles = len(moves) - len(u_moves)
        efficiency = (100 * n_total_tiles) / (n_total_tiles + n_revisted_tiles)
    return clean_percent, efficiency


def runSA(gamma=0.5, epsilon=0.05, alpha=0.6, episodes=300, steps=300, grid_file='house.grid'):
    # Cleaned tile percentage at which the room is considered 'clean':
    if grid_file == "example-random-house-3.grid":
        stopping_criteria = 82.5
    elif grid_file == "death.grid":
        stopping_criteria = 87.5
    else:
        stopping_criteria = 100
    deaths = 0
    # Open the grid file.
    # (You can create one yourself using the provided editor).
    with open(f'grid_configs/{grid_file}', 'rb') as f:
        grid = pickle.load(f)
    # Calculate the total visitable tiles:
    n_total_tiles = (grid.cells >= 0).sum()
    # Spawn the robot at (1,1) facing north with battery drainage enabled:
    robot = Robot(grid, (1, 1), orientation='n', battery_drain_p=1, battery_drain_lam=0.5)
    # Keep track of the number of robot decision epochs:
    n_epochs = 0
    while True:
        n_epochs += 1
        # Do a robot epoch (basically call the robot algorithm once):
        sa(robot, gamma=gamma, epsilon=epsilon, alpha=alpha, episodes=episodes, steps=steps)
        # Stop this simulation instance if robot died :( :
        if not robot.alive:
            deaths += 1
            break
        # Calculate some statistics:
        clean = (grid.cells == 0).sum()
        dirty = (grid.cells >= 1).sum()
        goal = (grid.cells == 2).sum()
        # Calculate the cleaned percentage:
        clean_percent = (clean / (dirty + clean)) * 100
        # See if the room can be considered clean, if so, stop the simulaiton instance:
        if clean_percent >= stopping_criteria and goal == 0:
            break
        # Calculate the effiency score:
        moves = [(x, y) for (x, y) in zip(robot.history[0], robot.history[1])]
        u_moves = set(moves)
        n_revisted_tiles = len(moves) - len(u_moves)
        efficiency = (100 * n_total_tiles) / (n_total_tiles + n_revisted_tiles)
    return clean_percent, efficiency

def runPC(g=0.9,t=0.001,c=0.3, grid_file = 'house.grid'):
    # Cleaned tile percentage at which the room is considered 'clean':
    if grid_file == "example-random-house-3.grid":
        stopping_criteria = 82.5
    elif grid_file == "death.grid":
        stopping_criteria = 87.5
    else:
        stopping_criteria = 100
    deaths = 0
    # Open the grid file.
    # (You can create one yourself using the provided editor).
    with open(f'grid_configs/{grid_file}', 'rb') as f:
        grid = pickle.load(f)
    # Calculate the total visitable tiles:
    n_total_tiles = (grid.cells >= 0).sum()
    # Spawn the robot at (1,1) facing north with battery drainage enabled:
    robot = Robot(grid, (1, 1), orientation='n', battery_drain_p=1, battery_drain_lam=0.5)
    # Keep track of the number of robot decision epochs:
    n_epochs = 0
    while True:
        n_epochs += 1
        # Do a robot epoch (basically call the robot algorithm once):
        pc(robot, gamma=g, theta=t, certainty=c)
        # Stop this simulation instance if robot died :( :
        if not robot.alive:
            deaths += 1
            break
        # Calculate some statistics:
        clean = (grid.cells == 0).sum()
        dirty = (grid.cells >= 1).sum()
        goal = (grid.cells == 2).sum()
        # Calculate the cleaned percentage:
        clean_percent = (clean / (dirty + clean)) * 100
        # See if the room can be considered clean, if so, stop the simulaiton instance:
        if clean_percent >= stopping_criteria and goal == 0:
            break
        # Calculate the effiency score:
        moves = [(x, y) for (x, y) in zip(robot.history[0], robot.history[1])]
        u_moves = set(moves)
        n_revisted_tiles = len(moves) - len(u_moves)
        efficiency = (100 * n_total_tiles) / (n_total_tiles + n_revisted_tiles)
    return clean_percent, efficiency

def runVA(g=0.8, t=0.001, c=0.3, grid_file = 'house.grid'):
    # Cleaned tile percentage at which the room is considered 'clean':
    if grid_file == "example-random-house-3.grid":
        stopping_criteria = 82.5
    elif grid_file == "death.grid":
        stopping_criteria = 87.5
    else:
        stopping_criteria = 100
    deaths = 0
    # Open the grid file.
    # (You can create one yourself using the provided editor).
    with open(f'grid_configs/{grid_file}', 'rb') as f:
        grid = pickle.load(f)
    # Calculate the total visitable tiles:
    n_total_tiles = (grid.cells >= 0).sum()
    # Spawn the robot at (1,1) facing north with battery drainage enabled:
    robot = Robot(grid, (1, 1), orientation='n', battery_drain_p=1, battery_drain_lam=0.5)
    # Keep track of the number of robot decision epochs:
    n_epochs = 0
    while True:
        n_epochs += 1
        # Do a robot epoch (basically call the robot algorithm once):
        va(robot, gamma=g, theta=t, certainty=c)
        # Stop this simulation instance if robot died :( :
        if not robot.alive:
            deaths += 1
            break
        # Calculate some statistics:
        clean = (grid.cells == 0).sum()
        dirty = (grid.cells >= 1).sum()
        goal = (grid.cells == 2).sum()
        # Calculate the cleaned percentage:
        clean_percent = (clean / (dirty + clean)) * 100
        # See if the room can be considered clean, if so, stop the simulaiton instance:
        if clean_percent >= stopping_criteria and goal == 0:
            break
        # Calculate the effiency score:
        moves = [(x, y) for (x, y) in zip(robot.history[0], robot.history[1])]
        u_moves = set(moves)
        n_revisted_tiles = len(moves) - len(u_moves)
        efficiency = (100 * n_total_tiles) / (n_total_tiles + n_revisted_tiles)
    return clean_percent, efficiency


def generate_results(grids, runs_per_combination=10):
    rows = []
    for g in grids:
        for i in range(runs_per_combination):
            # Monte-Carlo
            cleaned, efficiency = runMC(grid_file=g)
            rows.append([g, "Monte-Carlo", "cleanliness", cleaned.astype('float')])
            print(g, "Monte-Carlo", "cleanliness", cleaned.astype('float'))
            rows.append([g, "Monte-Carlo", "efficiency", efficiency.astype('float')])
            print(g, "Monte-Carlo", "efficiency", efficiency.astype('float'))
            # Sarsa
            cleaned, efficiency = runSA(grid_file=g)
            rows.append([g, "SARSA", "cleanliness", cleaned])
            print(g, "SARSA", "cleanliness", cleaned)
            rows.append([g, "SARSA", "efficiency", efficiency])
            print(g, "SARSA", "efficiency", efficiency)
            # Q-learning
            cleaned, efficiency = runQL(grid_file=g)
            rows.append([g, "Q-learning", "cleanliness", cleaned])
            print(g, "Q-learning", "cleanliness", cleaned)
            rows.append([g, "Q-learning", "efficiency", efficiency])
            print(g, "Q-learning", "efficiency", efficiency)
            # Policy Iteration
            cleaned, efficiency = runPC(grid_file=g)
            rows.append([g, "Policy iteration", "cleanliness", cleaned.astype('float')])
            print(g, "Policy Iteration", "cleanliness", cleaned.astype('float'))
            rows.append([g, "Policy iteration", "efficiency", efficiency.astype('float')])
            print(g, "Policy Iteration", "efficiency", efficiency.astype('float'))
            # Value Iteration
            cleaned, efficiency = runVA(grid_file=g)
            rows.append([g, "Value iteration", "cleanliness", cleaned.astype('float')])
            print(g, "Value Iteration", "cleanliness", cleaned.astype('float'))
            rows.append([g, "Value iteration", "efficiency", efficiency.astype('float')])
            print(g, "Value Iteration", "efficiency", efficiency.astype('float'))
    return rows



grid_files = ["death.grid", "house.grid", "example-random-house-3.grid", "snake.grid"]
rows = generate_results(grid_files)
my_array = np.array(rows)

df = pd.DataFrame(my_array, columns=['grid', 'algorithm', 'measurement', 'performance'])
df['grid'] = df.grid.astype('category')
df['algorithm'] = df.algorithm.astype('category')
df['measurement'] = df.measurement.astype('category')
df['performance'] = df.performance.astype('float64')

df.to_csv("violin3.csv")

g = sns.catplot(x="algorithm", y="performance",
                hue="measurement", col="grid",
                data=df, kind="violin", split=True,
                height=4, aspect=.7);
plt.show()