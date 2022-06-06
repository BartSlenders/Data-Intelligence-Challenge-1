import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


device = "cuda" if torch.cuda.is_available() else "cpu"


class PolicyNetwork(nn.Module):
    # state_space=4? if x_start,y_start,x_end,y_end
    def __init__(self, state_space, action_space):
        super(PolicyNetwork, self).__init__()

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

        self.input_layer = nn.Linear(observation_space, 128)
        self.output_layer = nn.Linear(128, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)
        state_value = self.output_layer(x)

        return state_value


def robot_epoch(robot, gamma=0.95, episodes=10, steps=10):
    low=-0.1
    high=0.1

    actions = []
    for _ in range(20):
        actions.append((np.random.uniform(low=low, high=high), np.random.uniform(low=low, high=high)))

    policy_network = PolicyNetwork(4, len(actions)).to(device)
    value_network = ValueNetwork(4).to(device)

    policy_optimizer = optim.Adam(policy_network.parameters(), lr=1e-2)
    value_optimizer = optim.Adam(value_network.parameters(), lr=1e-2)

    x_pos, y_pos = robot.pos

    for _ in range(episodes):
        episode = []
        state = np.array([x_pos, y_pos, x_pos+robot.size, y_pos+robot.size])

        # generate an episode following policy
        for t in range(steps):
            action, log_prob = policy_network.select_action(state)

            new_x_pos = state[0] + actions[action][0]
            new_y_pos = state[1] + actions[action][1]
            new_state = np.array([new_x_pos, new_y_pos, new_x_pos+robot.size, new_y_pos+robot.size])

            # to change when we have function for intersection
            reward = np.random.randint(-1, 2)

            episode.append([state, action, reward, log_prob])

            state = new_state

        states = [step[0] for step in episode]
        rewards = [step[2] for step in episode]
        log_probs = [step[3] for step in episode]

        G = []
        total_r = 0

        for reward in rewards[::-1]:
            G.append(reward + total_r * gamma)

        G = G[::-1]

        #whitening rewards
        G = torch.tensor(G).to(device)
        G = (G - G.mean())/G.std()

        value_estimates = []
        for state in states:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            value_estimates.append(value_network(state))

        value_estimates = torch.stack(value_estimates).squeeze()

        # Train value network
        value_loss = F.mse_loss(value_estimates, G)

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
        policy_optimizer.step()

    # obtain the best action from Q for the current state
    state = np.array([x_pos, y_pos, x_pos+robot.size, y_pos+robot.size])
    print('state '+str(state))
    action, _ = policy_network.select_action(state)
    print('action '+str(actions[action]))
    robot.direction_vector = action
    robot.move()
