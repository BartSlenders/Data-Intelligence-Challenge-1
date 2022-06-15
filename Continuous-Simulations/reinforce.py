import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from continuous import Square

device = "cuda" if torch.cuda.is_available() else "cpu"


class PolicyNetwork(nn.Module):
    # state_space=4? if x_start,y_start,x_end,y_end
    def __init__(self, state_space, action_space):
        super(PolicyNetwork, self).__init__()
        # TODO: experiment with different hidden state sizes (instead of 128)
        self.input_layer = nn.Linear(state_space, 128)
        self.output_layer = nn.Linear(128, action_space)

    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)
        actions = self.output_layer(x)
        action_probs = F.softmax(actions, dim=1)

        return action_probs

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        # get action probabilities
        action_probs = self.forward(state)
        state.detach()

        # sample an action using the probability distribution
        m = Categorical(action_probs)
        action = m.sample()

        # return action
        return action.item(), m.log_prob(action)


class ValueNetwork(nn.Module):
    def __init__(self, observation_space):
        super(ValueNetwork, self).__init__()
        # TODO: experiment with different hidden state sizes (instead of 128)
        self.input_layer = nn.Linear(observation_space, 128)
        self.output_layer = nn.Linear(128, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)
        state_value = self.output_layer(x)

        return state_value


def check_intersections(bounding_box, filthy, goals, obstacles, grid):
    blocked = any([ob.intersect(bounding_box) for ob in obstacles]) or \
                          not (bounding_box.x1 >= 0 and bounding_box.x2 <= grid.width and bounding_box.y1 >= 0 and
                               bounding_box.y2 <= grid.height)

    if blocked:
        return None, None, blocked

    new_filthy = copy.deepcopy(filthy)
    for i, filth in enumerate(filthy):
        if filth is not None:
            if filth.intersect(bounding_box):
                new_filthy.remove(new_filthy[i])

    new_goals = copy.deepcopy(goals)
    for i, goal in enumerate(goals):
        if goal is not None:
            if goal.intersect(bounding_box):
                new_goals.remove(new_goals[i])

    return new_filthy, new_goals, blocked


# TODO: experiment with gamma, alpha, episodes, steps
def robot_epoch(robot, gamma=0.95, alpha=0.001, episodes=25, steps=50):

    # TODO: experiment with action max/min value
    low = -0.2
    high = 0.2

    # The actions are the 4 straight directions and 20 random directions.
    actions = []
    # TODO: experiment with more or less than 20 actions
    for _ in range(40):
        # add random actions
        actions.append((np.random.uniform(low=low, high=high), np.random.uniform(low=low, high=high)))

    # TODO: should match the high/low
    # add up,down,left,right
    actions.append((0, 0.2))
    actions.append((0, -0.2))
    actions.append((0.2, 0))
    actions.append((-0.2, 0))

    policy_network = PolicyNetwork(4, len(actions)).to(device)
    value_network = ValueNetwork(4).to(device)

    policy_optimizer = optim.Adam(policy_network.parameters(), lr=alpha)
    value_optimizer = optim.Adam(value_network.parameters(), lr=alpha)

    x_pos, y_pos = robot.pos

    for _ in range(episodes):
        episode = []
        state = np.array([x_pos, y_pos, x_pos+robot.size, y_pos+robot.size])
        # NEW:
        prior_filthy = copy.deepcopy(robot.grid.filthy)
        prior_goals = copy.deepcopy(robot.grid.goals)

        # generate an episode following policy
        for t in range(steps):
            action, log_prob = policy_network.select_action(state)
            new_x_pos = state[0] + actions[action][0]
            new_y_pos = state[1] + actions[action][1]
            new_state = np.array([new_x_pos, new_y_pos, new_x_pos+robot.size, new_y_pos+robot.size])

            # calculate reward
            # check if the new position is possible: not out of bounds and does not intersect obstacle,
            # if so do not update position
            new_filthy, new_goals, is_blocked = check_intersections(Square(new_x_pos, new_x_pos + robot.size, new_y_pos,
                                                                           new_y_pos + robot.size),
                                                                    prior_filthy, prior_goals, robot.grid.obstacles,
                                                                    robot.grid)

            if is_blocked:
                # TODO: experiment with different reward
                reward = -2
                episode.append([state, action, reward, log_prob])
            else:
                factor_filthy = len(prior_filthy) - len(new_filthy)
                factor_goals = len(prior_goals) - len(new_goals)
                # TODO: experiment with different factors instead of 1 and 2
                reward = 1*factor_filthy + 2*factor_goals
                reward = 1*factor_filthy + 3*factor_goals

                episode.append([new_state, action, reward, log_prob])

                state = new_state
                prior_filthy = new_filthy
                prior_goals = new_goals


        states = [step[0] for step in episode]
        rewards = [step[2] for step in episode]
        log_probs = [step[3] for step in episode]

        G = []
        total_r = 0

        for reward in rewards[::-1]:
            total_r = reward + total_r * gamma
            G.append(total_r)

        G = G[::-1]

        #whitening rewards
        G = torch.tensor(G).to(device)
        if G.std().item() != 0:
            G = (G - G.mean())/G.std()

        value_estimates = []
        for state in states:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            value_estimates.append(value_network(state))

        value_estimates = torch.stack(value_estimates).squeeze()

        # Train value network
        value_loss = F.mse_loss(value_estimates, G)
        #print("value loss "+str(value_loss))

        #Backpropagate
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        # Train policy network
        deltas = [gt - estimate for gt, estimate in zip(G, value_estimates)]
        deltas = torch.tensor(deltas).to(device)

        policy_loss = []

        #calculate loss to be backpropagated
        for d, lp in zip(deltas, log_probs):
            #add negative sign since we are performing gradient ascent
            policy_loss.append(-d * lp)

        #Backpropagation
        policy_optimizer.zero_grad()
        sum(policy_loss).backward()
        #print("policy loss "+str(sum(policy_loss)))
        policy_optimizer.step()

    # obtain the best action from Q for the current state
    state = np.array([x_pos, y_pos, x_pos+robot.size, y_pos+robot.size])
    # print('state '+str(state))
    action, _ = policy_network.select_action(state)
    print('action '+str(actions[action]))
    robot.direction_vector = actions[action]
    robot.move()
