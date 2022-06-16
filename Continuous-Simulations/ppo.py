import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from continuous import Square

device = "cuda" if torch.cuda.is_available() else "cpu"


class ActorCriticNetwork(nn.Module):

    def __init__(self, obs_space, action_space):
        '''
        Args:
        - obs_space (int): observation space
        - action_space (int): action space

        '''
        super(ActorCriticNetwork, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(obs_space, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_space),
            nn.Softmax(dim=1))

        self.critic = nn.Sequential(
            nn.Linear(obs_space, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1))

    def forward(self):
        ''' Not implemented since we call the individual actor and critc networks for forward pass
        '''
        raise NotImplementedError

    def select_action(self, state):
        ''' Selects an action given current state
        Args:
        - network (Torch NN): network to process state
        - state (Array): Array of action space in an environment

        Return:
        - (int): action that is selected
        - (float): log probability of selecting that action given state and network
        '''

        # convert state to float tensor, add 1 dimension, allocate tensor on device
        state = torch.from_numpy(state).float().unsqueeze(0)

        # use network to predict action probabilities
        action_probs = self.actor(state)

        # sample an action using the probability distribution
        m = Categorical(action_probs)
        action = m.sample()

        # return action
        return action.item(), m.log_prob(action)

    def evaluate_action(self, states, actions):
        ''' Get log probability and entropy of an action taken in given state
        Args:
        - states (Array): array of states to be evaluated
        - actions (Array): array of actions to be evaluated

        '''

        # convert state to float tensor, add 1 dimension, allocate tensor on device
        states_tensor = torch.stack([torch.from_numpy(state).float().unsqueeze(0) for state in states]).squeeze(1)

        # use network to predict action probabilities
        action_probs = self.actor(states_tensor)

        # get probability distribution
        m = Categorical(action_probs)

        # return log_prob and entropy
        return m.log_prob(torch.Tensor(actions)), m.entropy()


class PPO_policy():

    def __init__(self, gamma, epsilon, beta, theta, c1, c2, k_epoch, actor_lr, critic_lr, state_space, action_space):

    def process_rewards(self, rewards, terminals):
        ''' Converts our rewards history into cumulative discounted rewards
        Args:
        - rewards (Array): array of rewards

        Returns:
        - G (Array): array of cumulative discounted rewards
        '''
        # Calculate Gt (cumulative discounted rewards)
        G = []

        # track cumulative reward
        total_r = 0

        # iterate rewards from Gt to G0
        for r, done in zip(reversed(rewards), reversed(terminals)):

            # Base case: G(T) = r(T)
            # Recursive: G(t) = r(t) + G(t+1)^DISCOUNT
            total_r = r + total_r * self.γ

            # no future rewards if current step is terminal
            if done:
                total_r = r

            # add to front of G
            G.insert(0, total_r)

        # whitening rewards
        G = torch.tensor(G)
        G = (G - G.mean()) / G.std()

        return G

    def clipped_update(self):
        ''' Update policy using clipped surrogate objective
        '''
        # get items from trajectory
        states = [sample[0] for sample in self.batch]
        actions = [sample[1] for sample in self.batch]
        rewards = [sample[2] for sample in self.batch]
        old_lps = [sample[3] for sample in self.batch]
        terminals = [sample[4] for sample in self.batch]

        # calculate cumulative discounted rewards
        Gt = self.process_rewards(rewards, terminals)

        # perform k-epoch update
        for epoch in range(self.k_epoch):
            # get ratio
            new_lps, entropies = self.actor_critic.evaluate_action(states, actions)

            ratios = torch.exp(new_lps - torch.Tensor(old_lps))

            # compute advantages
            states_tensor = torch.stack([torch.from_numpy(state).float().unsqueeze(0) for state in states]).squeeze(1)
            vals = self.actor_critic.critic(states_tensor).squeeze(1).detach()
            advantages = Gt - vals

            # clip surrogate objective
            surrogate1 = torch.clamp(ratios, min=1 - self.ϵ, max=1 + self.ϵ) * advantages
            surrogate2 = ratios * advantages

            # loss, flip signs since this is gradient descent
            loss = -torch.min(surrogate1, surrogate2) + self.c1 * F.mse_loss(Gt, vals) - self.c2 * entropies

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # clear batch buffer
        self.batch = []

def robot_epoch(robot, gamma=0.99, epsilon=0.2, beta=1, theta=0.01, c1=0.5, c2=0.01, k_epoch=40, actor_lr = 0.0003,
                critic_lr = 0.001, episodes=20, steps=40):
    actions = [(0.1,0), (0,0.1), (-0.1,0), (0,-0.1),
                    (0.2,0), (0,0.2), (-0.2,0), (0,-0.2),
                   (0.1,0.1),(-0.1,0.1),(0.1,-0.1),(-0.1,-0.1),
                   (0.2,0.2),(-0.2,0.2),(0.2,-0.2),(-0.2,-0.2)]

    actor_critic = ActorCriticNetwork(state_space=4, action_space=2)

    optimizer = torch.optim.Adam([
        {'params': actor_critic.actor.parameters(), 'lr': actor_lr},
        {'params': actor_critic.critic.parameters(), 'lr': critic_lr}])

    x_pos, y_pos = robot.pos

    counter = 0

    for _ in range(episodes):
        episode = []
        state = np.array([x_pos, y_pos, x_pos + robot.size, y_pos + robot.size])
        # NEW:
        prior_filthy = copy.deepcopy(robot.grid.filthy)
        prior_goals = copy.deepcopy(robot.grid.goals)

        # generate an episode following policy
        for t in range(steps):
            counter += 1
            action_id, log_prob = actor_critic.get_action(state)
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
                reward = -2
                episode.append([state, action_id, reward, log_prob])
            else:
                factor_filthy = len(prior_filthy) - len(new_filthy)
                factor_goals = len(prior_goals) - len(new_goals)
                reward = 1 * factor_filthy + 3 * factor_goals

                episode.append([new_state, action_id, reward, log_prob])

                state = new_state
                prior_filthy = new_filthy
                prior_goals = new_goals

            if counter % batch_size

            states = []
            rewards = []
            log_probs = []

            for step in episode:
                states.append(step[0])
                rewards.append(step[2])
                log_probs.append(step[3])

            G = []
            reward_sum = 0

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

            if done:
                break

    # obtain the best action from Q for the current state
    state = np.array([x_pos, y_pos, x_pos + robot.size, y_pos + robot.size])
    # print('state '+str(state))
    action_id, _ = policy_network.get_action(state)
    print('action ' + str(actions[action_id]))
    robot.direction_vector = actions[action_id]
    robot.move()