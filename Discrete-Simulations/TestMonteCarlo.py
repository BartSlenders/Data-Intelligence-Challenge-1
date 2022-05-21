# Import our robot algorithm to use in this simulation:
from robot_configs.MonteCarlo import robot_epoch
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
def run(ipe=5, gamma=0.8, epsilon=0.1, steps = 200, grid_file ='house.grid'):
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
        robot_epoch(robot, iterations_per_evaluation=ipe, gamma=gamma, epsilon=epsilon, epochs=steps)
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
def plotDistribution(robot, ipe=3, discount=0.8, epsilon=0.95, epochs=100):
    cleaned = []
    efficiencies = []
    for i in range(runs):
        cleaned_a, efficiency = run(iterations_per_evaluation=ipe, gamma=discount, epsilon=epsilon, epochs=epochs)
        cleaned.append(cleaned_a)
        efficiencies.append(efficiency)
    my_array = np.array([cleaned, efficiencies]).T
    df = pd.DataFrame(my_array, columns = ['cleaned', 'efficiencies'])
    ax = sns.violinplot(data=df)
    ax.set_ylabel("iteration policy performance (%)")
    ax.set_title("distribution of robot performance", size=18, weight='bold');
    plt.show()

"""
    Plots violin plots representing the distribution of cleanliness of 10 consecutive runs of Policy iteration on
     multiple grids. 
    @g: the gamma parameter
"""
def plotcleanness(robot, ipe=3, discount=0.8, epsilon=0.95, epochs=100):
    grid_files = ["death.grid", "house.grid", "example-random-house-3.grid", "snake.grid"]
    grid_files_names = ["death", "house", "random-house-3", "snake"]
    array = []
    for gf in grid_files:
        efficiencies = []
        for i in range(runs):
            cleaned_a, efficiency = run(iterations_per_evaluation=ipe, gamma=discount, epsilon=epsilon, epochs=epochs)
            efficiencies.append(efficiency)
        array.append(efficiencies)

    my_array = np.array(array).T
    df = pd.DataFrame(my_array, columns = grid_files_names)
    ax =sns.violinplot(data=df)

    ax.set_ylabel("efficiency (%)")
    ax.set_title("Monte Carlo method - efficiency vs grid", size=18, weight='bold');
    plt.show()

"""
    Generates a csv file under the name "results.csv" containing the probabilities and efficiencies of multiple runs
    of policy iteration, together with the parameters used.
    @gamma: the gamma parameter
"""
def generate_results(ipe, discount, epsilon, steps, runs_per_combination=3):
    rows = []

    for d in discount:
        for e in epsilon:
            for i in ipe:
                for s in steps:
                    print('ipe:', i, '\tepochs:', s, 'gamma:', d, '\tepsilon:', e)
                    for i in range(runs_per_combination):
                        cleaned, efficiency = run(ipe=i, gamma=d, epsilon=e, steps=s)
                        rows.append([i, s, d, e, cleaned, efficiency])
                    print('\tcleaned:', cleaned, '\tefficiency:', efficiency)
    my_array = np.array(rows)
    df = pd.DataFrame(my_array, columns=['ipe', 'epochs', 'gamma', 'epsilon', 'cleaned', 'efficiency'])
    df.to_csv("results.csv")


iterations_per_evaluation = np.array([1, 5, 10])
gammas = np.array([1, 0.8, 0.6, 0.4])
epsilon = np.array([0.05, 0.1, 0.3])
epochs = np.array([100, 200])

# gammas = np.array([0.5, 0.7, 0.9])
# epsilon = np.array([0.05, 0.1, 0.2])
# iterations_per_evaluation = np.array([50, 150, 300])
# epochs = np.array([50, 150, 300])

generate_results(iterations_per_evaluation, gammas, epsilon, epochs)
