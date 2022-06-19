import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from continuous import SimGridComplex as SimGrid

device = "cuda" if torch.cuda.is_available() else "cpu"


class ActorCriticNetwork(nn.Module):
    """
    Class implementing a shared network architecture between an actor and a critic network
    """
    def __init__(self, state_space, action_space):
        """
        Initializes the actor and critic networks' layers
        :param state_space: the dimensionality of state (x,y)
        :param action_space: the number of actions
        """
        super(ActorCriticNetwork, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_space, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_space),
            nn.Softmax(dim=1))

        self.critic = nn.Sequential(
            nn.Linear(state_space, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1))

    def forward(self):
        raise NotImplementedError

    def get_action(self, state):
        """
        Sample an action and its log probability from the actor network
        :param state: a state (x,y)
        :returns action_id, log_prob: an action and its log probability
        """
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_probs = self.actor(state)
        categorical_distribution = Categorical(action_probs)
        action = categorical_distribution.sample()
        action_id = action.item()
        log_prob = categorical_distribution.log_prob(action)

        return action_id, log_prob

    def evaluate_action(self, states, actions):
        """
        Sample an action and its log probability from the actor network
        :param states: the states in the batch
        :param actions: the actions in the batch
        :returns action_id, log_prob: an action and its log probability
        """
        # convert state to float tensor, add 1 dimension, allocate tensor on device
        states_tensor = torch.stack([torch.from_numpy(state).float().unsqueeze(0) for state in states]).squeeze(1)

        # use network to predict action probabilities
        action_probs = self.actor(states_tensor)

        # get probability distribution
        categorical_distribution = Categorical(action_probs)

        # return log_prob and entropy
        return categorical_distribution.log_prob(torch.Tensor(actions)), categorical_distribution.entropy()


def robot_epoch(robot, gamma=0.99, epsilon=0.2, c1=0.5, c2=0.01, k_epoch=40, actor_lr=0.0003, critic_lr=0.001,
                batch_size=5, episodes=20, steps=40):
    actions = [(0.85, 0), (0, 0.85), (-0.85, 0), (0, -0.85),
               (1.05, 0), (0, 1.05), (-1.05, 0), (0, -1.05),
               (0.46, 0.46), (-0.46, 0.46), (0.46, -0.46), (-0.46, -0.46),
               (1.06, 1.06), (-1.06, 1.06), (1.06, -1.06), (-1.06, -1.06)]

    actor_critic = ActorCriticNetwork(state_space=2, action_space=len(actions))

    optimizer = torch.optim.Adam([
        {'params': actor_critic.actor.parameters(), 'lr': actor_lr},
        {'params': actor_critic.critic.parameters(), 'lr': critic_lr}])

    x_pos, y_pos = robot.pos

    counter = 0

    for _ in range(1, episodes+1):
        batch = []
        state = np.array([x_pos, y_pos])
        prior_filthy = copy.deepcopy(robot.grid.filthy)
        prior_goals = copy.deepcopy(robot.grid.goals)

        # generate an episode following policy
        for t in range(steps):
            counter += 1
            action_id, log_prob = actor_critic.get_action(state)
            new_x_pos = state[0] + actions[action_id][0]
            new_y_pos = state[1] + actions[action_id][1]
            new_state = np.array([new_x_pos, new_y_pos])

            # calculate reward
            # check if the new position is possible: not out of bounds and does not intersect obstacle,
            # if so do not update position
            test = SimGrid(new_state, robot.grid, prior_filthy, prior_goals)
            reward, is_blocked, done, new_filthy, new_goals = test.reward(actions[action_id])

            if is_blocked:
                batch.append([state, action_id, reward, log_prob, done])
            else:
                batch.append([new_state, action_id, reward, log_prob, done])

                state = new_state
                prior_filthy = new_filthy
                prior_goals = new_goals

            #print('filthy: ', len(new_filthy), 'state: ', state, 'action: ', actions[action_id], 'blocked: ', is_blocked)

            # learn
            if counter % batch_size == 0:
                # create lists to store batch items
                states = []
                action_ids = []
                rewards = []
                old_log_probs = []
                terminals = []

                # get batch items
                for step in batch:
                    states.append(step[0])
                    action_ids.append(step[1])
                    rewards.append(step[2])
                    old_log_probs.append(step[3])
                    terminals.append(step[4])

                G = []
                reward_sum = 0

                # iterate rewards, dones backwards
                for reward, done in zip(rewards[::-1], terminals[::-1]):
                    reward_sum = reward + reward_sum * gamma
                    # no future rewards if nothing left to clean
                    if done:
                        reward_sum = reward

                    G.append(reward_sum)

                G = G[::-1]
                G = torch.tensor(G).to(device)

                if G.std().item() != 0:
                    G = (G - G.mean()) / G.std()

                # perform k-epoch update
                for epoch in range(k_epoch):
                    # get ratio
                    new_log_probs, entropies = actor_critic.evaluate_action(states, action_ids)
                    ratios = torch.exp(new_log_probs - torch.Tensor(old_log_probs))

                    # compute advantages
                    states_tensor = []
                    for state in states:
                        states_tensor.append(torch.from_numpy(state).float().unsqueeze(0))
                    states_tensor = torch.stack(states_tensor).squeeze(1)
                    value_estimates = actor_critic.critic(states_tensor).squeeze(1).detach()
                    advantages = G - value_estimates

                    # clip surrogate objective
                    surrogate1 = torch.clamp(ratios, min=1 - epsilon, max=1 + epsilon) * advantages
                    surrogate2 = ratios * advantages

                    # compute loss
                    # add minus for gradient ascent
                    loss = -torch.min(surrogate1, surrogate2) + c1 * F.mse_loss(G, value_estimates) - c2 * entropies

                    # backpropagate loss
                    optimizer.zero_grad()
                    loss.mean().backward()
                    optimizer.step()

                # clear batch buffer
                batch = []

            if done:
                break

    # obtain the best action from the ppo actor_critic network
    state = np.array([x_pos, y_pos])
    # print('state '+str(state))
    action_id, _ = actor_critic.get_action(state)
    # print('action ' + str(actions[action_id]))
    robot.direction_vector = actions[action_id]
    robot.move()