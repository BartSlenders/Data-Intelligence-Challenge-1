import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from continuous import SimGridComplex as SimGrid

device = "cuda" if torch.cuda.is_available() else "cpu"


class PolicyNetwork(nn.Module):
    """
    Class implementing a Neural Network approximator for the policy
    """
    def __init__(self, state_space, action_space):
        """
        Initializes the network's layers
        :param state_space: the dimensionality of state (x,y)
        :param action_space: the number of actions
        """
        super(PolicyNetwork, self).__init__()
        self.input_layer = nn.Linear(state_space, 128)
        self.output_layer = nn.Linear(128, action_space)

    def forward(self, state):
        """
        Perform a forward pass through the network
        :param state: a state (x,y)
        :returns action_probabilities: the probability for each action
        """
        state = self.input_layer(state)
        state = F.relu(state)
        actions = self.output_layer(state)
        action_probabilities = F.softmax(actions, dim=1)

        return action_probabilities

    def get_action(self, state):
        """
        Sample an action and its log probability from the network
        :param state: a state (x,y)
        :returns action_id, log_prob: an action and its log probability
        """
        state = torch.Tensor(state).float().unsqueeze(0).to(device)
        action_probs = self.forward(state)
        state.detach()
        categorical_distribution = Categorical(action_probs)
        action = categorical_distribution.sample()
        action_id = action.item()
        log_prob = categorical_distribution.log_prob(action)

        return action_id, log_prob


class ValueNetwork(nn.Module):
    """
    Class implementing a Neural Network approximator for the value function baseline
    """
    def __init__(self, state_space):
        """
        Initializes the network's layers
        :param state_space: the dimensionality of state (x,y)
        """
        super(ValueNetwork, self).__init__()
        self.input_layer = nn.Linear(state_space, 128)
        self.output_layer = nn.Linear(128, 1)

    def forward(self, state):
        """
        Perform a forward pass through the network
        :param state: a state (x,y)
        :returns state_value: the evaluation of state
        """
        state = self.input_layer(state)
        state = F.relu(state)
        state_value = self.output_layer(state)

        return state_value


def robot_epoch(robot, gamma=0.9, alpha=0.1, episodes=20, steps=100):
    """
    Execute Reinforce with baseline algorithm to find the best move
    :param robot: main actor of type Robot
    :param gamma: discount factor
    :param alpha: learning rate for both networks
    :param episodes: number of episodes
    :param steps: number of steps
    """
    # the discretized action space
    actions = [(0.85, 0), (0, 0.85), (-0.85, 0), (0, -0.85),
               (1.05, 0), (0, 1.05), (-1.05, 0), (0, -1.05),
               (0.46, 0.46), (-0.46, 0.46), (0.46, -0.46), (-0.46, -0.46),
               (1.06, 1.06), (-1.06, 1.06), (1.06, -1.06), (-1.06, -1.06)]

    policy_network = PolicyNetwork(2, len(actions)).to(device)
    value_network = ValueNetwork(2).to(device)

    policy_optimizer = optim.Adam(policy_network.parameters(), lr=alpha)
    value_optimizer = optim.Adam(value_network.parameters(), lr=alpha)

    x_pos, y_pos = robot.pos

    for _ in range(episodes):
        episode = []
        state = np.array([x_pos, y_pos])
        prior_filthy = copy.deepcopy(robot.grid.filthy)
        prior_goals = copy.deepcopy(robot.grid.goals)

        for t in range(steps):
            action_id, log_prob = policy_network.get_action(state)
            new_x_pos = state[0] + actions[action_id][0]
            new_y_pos = state[1] + actions[action_id][1]
            new_state = np.array([new_x_pos, new_y_pos])

            # create simulation grid to get the result of a move
            test = SimGrid(new_state, robot.grid, prior_filthy, prior_goals)
            reward, is_blocked, done, new_filthy, new_goals = test.reward(actions[action_id])

            if is_blocked:
                # do not update the state to the new state
                episode.append([state, action_id, reward, log_prob])
            else:
                episode.append([new_state, action_id, reward, log_prob])
                state = new_state
                prior_filthy = new_filthy
                prior_goals = new_goals

            if done:
                break

        # obtain elements of the episode
        states = []
        rewards = []
        log_probs = []

        for step in episode:
            states.append(step[0])
            rewards.append(step[2])
            log_probs.append(step[3])

        # compute the expected return for episode
        G = []
        reward_sum = 0
        for reward in rewards[::-1]:
            reward_sum = reward + reward_sum * gamma
            G.append(reward_sum)

        # normalize
        G = torch.tensor(G[::-1]).to(device)
        if G.std().item() != 0:
            G = (G - G.mean()) / G.std()

        # compute the value estimate baselines for each state in the episode
        value_estimates = []
        for state in states:
            state = torch.Tensor(state).float().unsqueeze(0).to(device)
            value_estimates.append(value_network(state))

        value_estimates = torch.stack(value_estimates).squeeze()

        # Train value network
        value_loss = F.mse_loss(value_estimates, G)

        # backpropagate loss
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        # Train policy network
        gradients = [g_t - estimate for g_t, estimate in zip(G, value_estimates)]
        gradients = torch.tensor(gradients).to(device)

        policy_loss = []

        # calculate policy loss
        for gradient, log_prob in zip(gradients, log_probs):
            # add minus for gradient ascent
            policy_loss.append(-gradient * log_prob)

        # backpropagate loss
        policy_optimizer.zero_grad()
        sum(policy_loss).backward()
        policy_optimizer.step()

    # obtain the best action from the policy network
    state = np.array([x_pos, y_pos])
    action_id, _ = policy_network.get_action(state)
    robot.direction_vector = actions[action_id]
    robot.move()