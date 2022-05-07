# Import our robot algorithm to use in this simulation:
#from robot_configs.greedy_random_robot import robot_epoch
from robot_configs.policy import robot_epoch
import pickle
from environment import Robot
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

grid_file = 'house.grid'
# Cleaned tile percentage at which the room is considered 'clean':
stopping_criteria = 100

# Keep track of some statistics:
efficiencies = []
n_moves = []

cleaned = []

# hyper-parameters, 1000 combinations
theta = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10])
gamma = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
certainty = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

def run(g=0.9,t=0.1,c=0.9):
    deaths = 0
    # Open the grid file.
    # (You can create one yourself using the provided editor).
    with open(f'grid_configs/{grid_file}', 'rb') as f:
        grid = pickle.load(f)
    # Calculate the total visitable tiles:
    n_total_tiles = (grid.cells >= 0).sum()
    # Spawn the robot at (1,1) facing north with battery drainage enabled:
    robot = Robot(grid, (1, 1), orientation='n', battery_drain_p=0.5, battery_drain_lam=2)
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

# # Make some plots:
# cleaned,efficiencies = run()
# plt.hist(cleaned)
# plt.title('Percentage of tiles cleaned.')
# plt.xlabel('% cleaned')
# plt.ylabel('count')
# plt.show()
#
# plt.hist(efficiencies)
# plt.title('Efficiency of robot.')
# plt.xlabel('Efficiency %')
# plt.ylabel('count')
# plt.show()
def plotDistribution(g=0.95, t=0.001, c=0.255):
    cleaned = []
    efficiencies = []
    for i in range(10):
        cleaned_a, efficiency = run(g=g, t=t, c=c)
        cleaned.append(cleaned_a)
        efficiencies.append(efficiency)
    my_array = np.array([cleaned, efficiencies]).T
    df = pd.DataFrame(my_array, columns = ['cleaned','efficiencies'])
    ax =sns.violinplot(data=df)
    ax.set_ylabel("performance (%)")
    ax.set_title("distribution of robot performance", size=18, weight='bold');
    plt.show()

def generate_results(gamma, theta, certainty):
    rows = []
    for g in gamma:
        for t in theta:
            for c in certainty:
                for i in range(100):
                    cleaned, efficiency = run(g=g,t=t,c=c)
                    rows.append([g, t, c, cleaned, efficiency])
    my_array = np.array(rows)
    df = pd.DataFrame(my_array, columns=['gamma', 'theta', 'certainty', 'cleaned', 'efficiency'])
    df.to_csv("results6V.csv")

theta = np.array([0.01, 0.05, 0.1, 0.5, 1, 5])
gamma = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.99])
certainty = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1])

theta = np.array([0.001])
gamma = np.array([0.9, 0.925, 0.95, 0.99])
certainty = np.array([0.255, 0.28, 0.3, 0.35])

theta = np.array([0.001])
gamma = np.array([0.92, 0.925, 0.93, 0.94, 0.95, 0.96])
certainty = np.array([0.2501, 0.2505, 0.251, 0.255, 0.265, 0.28])

# generate_results(gamma, theta, certainty)
plotDistribution()