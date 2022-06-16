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
    # state_space=4: x_start,y_start,x_end,y_end
    def __init__(self, state_space, action_space):
        super(PolicyNetwork, self).__init__()
        # TODO: experiment with different hidden state sizes (instead of 128)
        self.input_layer = nn.Linear(state_space, 128)
        self.output_layer = nn.Linear(128, action_space)

    def forward(self, state):
        state = self.input_layer(state)
        state = F.relu(state)
        actions = self.output_layer(state)
        action_probabilities = F.softmax(actions, dim=1)

        return action_probabilities

    # sample an action from the output of a network, use log_prob to construct an equivalent loss function
    def get_action(self, state):
        state = torch.Tensor(state).float().unsqueeze(0).to(device)
        action_probs = self.forward(state)
        state.detach()
        categorical_distribution = Categorical(action_probs)
        action = categorical_distribution.sample()

        return action.item(), categorical_distribution.log_prob(action)


class ValueNetwork(nn.Module):
    def __init__(self, state_space):
        super(ValueNetwork, self).__init__()
        # TODO: experiment with different hidden state sizes (instead of 128)
        self.input_layer = nn.Linear(state_space, 128)
        self.output_layer = nn.Linear(128, 1)

    def forward(self, state):
        state = self.input_layer(state)
        state = F.relu(state)
        state_value = self.output_layer(state)

        return state_value


def check_intersections(bounding_box, filthy, goals, obstacles, grid):
    blocked = any([ob.intersect(bounding_box) for ob in obstacles]) or \
                          not (bounding_box.x1 >= 0 and bounding_box.x2 <= grid.width and bounding_box.y1 >= 0 and
                               bounding_box.y2 <= grid.height)

    if blocked:
        return filthy, goals, blocked, False

    new_filthy = copy.deepcopy(filthy)
    for i, filth in enumerate(filthy):
        if filth is not None:
            if filth.intersect(bounding_box):
                new_filthy[i] = None

    new_filthy = [i for i in new_filthy if i]

    new_goals = copy.deepcopy(goals)
    for i, goal in enumerate(goals):
        if goal is not None:
            if goal.intersect(bounding_box):
                new_goals[i] = None

    new_goals = [i for i in new_goals if i]

    if len(new_filthy) == 0 and len(new_goals) == 0:
        done = True
    else:
        done = False

    return new_filthy, new_goals, blocked, done


# TODO: experiment with gamma, alpha, episodes, steps
def robot_epoch(robot, gamma=0.95, alpha=0.001, episodes=20, steps=40):
    # TODO: experiment with action max/min value
    low = -0.5
    high = 0.5

    # The actions are the 4 straight directions and 20 random directions.
    # actions = []
    # # TODO: experiment with more or less than 20 actions
    # for _ in range(40):
    #     # add random actions
    #     actions.append((np.random.uniform(low=low, high=high), np.random.uniform(low=low, high=high)))
    #
    # # TODO: should match the high/low
    # # add up,down,left,right
    # actions.append((0, 0.2))
    # actions.append((0, -0.2))
    # actions.append((0.2, 0))
    # actions.append((-0.2, 0))
    # actions.append((0, 0.1))
    # actions.append((0, -0.1))
    # actions.append((0.1, 0))
    # actions.append((-0.1, 0))

    actions = [(0.85, 0), (0, 0.85), (-0.85, 0), (0, -0.85),
               (1.05, 0), (0, 1.05), (-1.05, 0), (0, -1.05),
               (0.46, 0.46), (-0.46, 0.46), (0.46, -0.46), (-0.46, -0.46),
               (1.06, 1.06), (-1.06, 1.06), (1.06, -1.06), (-1.06, -1.06)]

    policy_network = PolicyNetwork(4, len(actions)).to(device)
    value_network = ValueNetwork(4).to(device)

    policy_optimizer = optim.Adam(policy_network.parameters(), lr=alpha)
    value_optimizer = optim.Adam(value_network.parameters(), lr=alpha)

    x_pos, y_pos = robot.pos

    for _ in range(episodes):
        episode = []
        state = np.array([x_pos, y_pos, x_pos + robot.size, y_pos + robot.size])
        # NEW:
        prior_filthy = copy.deepcopy(robot.grid.filthy)
        prior_goals = copy.deepcopy(robot.grid.goals)

        # generate an episode following policy
        for t in range(steps):
            action_id, log_prob = policy_network.get_action(state)
            new_x_pos = state[0] + actions[action_id][0]
            new_y_pos = state[1] + actions[action_id][1]
            new_state = np.array([new_x_pos, new_y_pos, new_x_pos + robot.size, new_y_pos + robot.size])

            # calculate reward
            # check if the new position is possible: not out of bounds and does not intersect obstacle,
            # if so do not update position
            new_filthy, new_goals, is_blocked, done = check_intersections(Square(new_x_pos, new_x_pos + robot.size,
                                                                                 new_y_pos, new_y_pos + robot.size),
                                                                    prior_filthy, prior_goals, robot.grid.obstacles,
                                                                    robot.grid)

            if is_blocked:
                # TODO: experiment with different reward
                reward = -2
                episode.append([state, action_id, reward, log_prob])
            else:
                factor_filthy = len(prior_filthy) - len(new_filthy)
                factor_goals = len(prior_goals) - len(new_goals)
                # TODO: experiment with different factors instead of 1 and 2
                reward = 1 * factor_filthy + 3 * factor_goals

                episode.append([new_state, action_id, reward, log_prob])

                state = new_state
                prior_filthy = new_filthy
                prior_goals = new_goals

            #print('filthy: ', len(new_filthy), 'state: ', state, 'action: ', actions[action_id])

            if done:
                break

        states = []
        rewards = []
        log_probs = []

        for step in episode:
            states.append(step[0])
            rewards.append(step[2])
            log_probs.append(step[3])

        G = []
        reward_sum = 0

        # iterate rewards backwards
        for reward in rewards[::-1]:
            reward_sum = reward + reward_sum * gamma
            G.append(reward_sum)

        G = G[::-1]

        # whitening rewards
        G = torch.tensor(G).to(device)
        if G.std().item() != 0:
            G = (G - G.mean()) / G.std()

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
        deltas = [g_t - estimate for g_t, estimate in zip(G, value_estimates)]
        deltas = torch.tensor(deltas).to(device)

        policy_loss = []

        # calculate policy loss
        for delta, log_prob in zip(deltas, log_probs):
            # add minus for gradient ascent
            policy_loss.append(-delta * log_prob)

        # backpropagate loss
        policy_optimizer.zero_grad()
        sum(policy_loss).backward()
        policy_optimizer.step()

    # obtain the best action from Q for the current state
    state = np.array([x_pos, y_pos, x_pos + robot.size, y_pos + robot.size])
    # print('state '+str(state))
    action_id, _ = policy_network.get_action(state)
    print('action ' + str(actions[action_id]))
    robot.direction_vector = actions[action_id]
    robot.move()
