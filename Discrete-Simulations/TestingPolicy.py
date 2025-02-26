# Import our robot algorithm to use in this simulation:
from robot_configs.policy import robot_epoch
import pickle
from environment import Robot
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# Cleaned tile percentage at which the room is considered 'clean':
stopping_criteria = 100

# Keep track of some statistics:
efficiencies = []
n_moves = []
cleaned = []


"""
    Executes a run of Policy iteration. 
    @g: the gamma parameter
    @t: the theta parameter
    @c: the certainty used for policy iteration 
"""
def run(g=0.9,t=0.1,c=0.9, grid_file = 'house.grid'):
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
        robot_epoch(robot, gamma=g, theta=t, certainty=c)
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

"""
    Plots violin plots representing the distribution of cleanliness and efficiency
     of 10 consecutive runs of Policy iteration. 
    @g: the gamma parameter
    @t: the theta parameter
    @c: the certainty used for policy iteration 
"""
def plotDistribution(g=0.99, t=0.001, c=0.3):
    cleaned = []
    efficiencies = []
    for i in range(10):
        cleaned_a, efficiency = run(g=g, t=t, c=c)
        cleaned.append(cleaned_a)
        efficiencies.append(efficiency)
    my_array = np.array([cleaned, efficiencies]).T
    df = pd.DataFrame(my_array, columns = ['cleaned','efficiencies'])
    ax =sns.violinplot(data=df)
    ax.set_ylabel("iteration policy performance (%)")
    ax.set_title("distribution of robot performance", size=18, weight='bold');
    plt.show()

"""
    Plots violin plots representing the distribution of cleanliness of 10 consecutive runs of Policy iteration on
     multiple grids. 
    @g: the gamma parameter
    @t: the theta parameter
    @c: the certainty used for policy iteration 
"""
def plotcleanness(g=0.9, t=0.001, c=0.3):
    grid_files = ["death.grid", "house.grid", "example-random-house-3.grid", "snake.grid"]
    grid_files_names = ["death", "house", "random-house-3", "snake"]
    array = []
    for gf in grid_files:
        efficiencies = []
        for i in range(10):
            cleaned_a, efficiency = run(g=g, t=t, c=c, grid_file=gf)
            efficiencies.append(efficiency)
        array.append(efficiencies)

    my_array = np.array(array).T
    df = pd.DataFrame(my_array, columns = grid_files_names)
    ax =sns.violinplot(data=df)

    ax.set_ylabel("efficiency (%)")
    ax.set_title("policy iteration - efficiency vs grid", size=18, weight='bold');
    plt.show()

"""
    Generates a csv file under the name "results.csv" containing the probabilities and efficiencies of multiple runs
    of policy iteration, together with the parameters used.
    @gamma: the gamma parameter
    @theta: the theta parameter
    @ccertainty: the certainty used for policy iteration 
"""
def generate_results(gamma, theta, certainty):
    rows = []
    for g in gamma:
        for t in theta:
            for c in certainty:
                for i in range(20):
                    cleaned, efficiency = run(g=g,t=t,c=c)
                    rows.append([g, t, c, cleaned, efficiency])
    my_array = np.array(rows)
    df = pd.DataFrame(my_array, columns=['gamma', 'theta', 'certainty', 'cleaned', 'efficiency'])
    df.to_csv("results9.csv")


theta = np.array([0.001])
gamma = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.99])
certainty = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


generate_results(gamma, theta, certainty)
plotDistribution()
plotcleanness()