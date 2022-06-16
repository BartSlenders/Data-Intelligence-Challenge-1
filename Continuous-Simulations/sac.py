import os

import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
import copy
from continuous import Square


class ReplayBuffer:
    def __init__(self, max_memory_size, state_space, action_space):
        self.memory_size = max_memory_size
        self.memory_counter = 0
        self.state_memory = np.zeros((self.memory_size, state_space))
        self.new_state_memory = np.zeros((self.memory_size, state_space))
        self.action_memory = np.zeros((self.memory_size, action_space))
        self.reward_memory = np.zeros(self.memory_size)
        self.terminal_memory = np.zeros(self.memory_size, dtype=np.bool)

    def store_transition(self, state, action, reward, new_state, done):
        index = self.memory_counter % self.memory_size

        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.memory_counter += 1

    def sample_buffer(self, batch_size):
        max_memory = min(self.memory_counter, self.memory_size)
        batch = np.random.choice(max_memory, batch_size)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, new_states, dones


class CriticNetwork(nn.Module):
    def __init__(self, beta, state_space, action_space, fc1_dims=256, fc2_dims=256, name='critic',
                 checkpoint_dir='tmp/sac'):
        super(CriticNetwork, self).__init__()

        self.fc1 = nn.Linear(state_space+action_space, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q_layer = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.checkpoint_file = os.path.join(checkpoint_dir, name + '_sac')

        self.to(self.device)

    def forward(self, state, action):
        x = self.fc1(T.cat([state, action], dim=1))
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        q_estimate = self.q_layer(x)

        return q_estimate

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ValueNetwork(nn.Module):
    def __init__(self, beta, state_space, fc1_dims=256, fc2_dims=256, name='value', checkpoint_dir='tmp/sac'):
        super(ValueNetwork, self).__init__()

        self.fc1 = nn.Linear(state_space, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.value_layer = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.checkpoint_file = os.path.join(checkpoint_dir, name + '_sac')

        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        value_estimate = self.value_layer(x)

        return value_estimate

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, state_space, action_space, max_action_value, fc1_dims=256, fc2_dims=256, name='actor',
                 checkpoint_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()

        self.fc1 = nn.Linear(state_space, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mean = nn.Linear(fc2_dims, action_space)
        self.std = nn.Linear(fc2_dims, action_space)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.checkpoint_file = os.path.join(checkpoint_dir, name+'_sac')
        self.reparameterization_noise = 1e-6
        self.max_action_value = max_action_value
        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        mean = self.mean(x)
        std = self.std(x)
        # fix width of distribution in range [1e-6, 1]
        std = T.clamp(std, min=self.reparameterization_noise, max=1)

        return mean, std

    def sample_normal(self, state, reparameterize=True):
        mean, std = self.forward(state)
        probabilities = Normal(mean, std)

        if reparameterize:
            # add noise for exploration
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        # https: // arxiv.org / pdf / 1812.05905.pdf appendix section c
        # use invertable squashing function to bound actions to a finite interval and
        # multiply by max_action_value to fix action in the appropriate range
        action = T.tanh(actions)*T.tensor(self.max_action_value).to(self.device)
        # get log of the probabilities for the computation of the loss
        log_probs = probabilities.log_prob(actions)
        # add noise to prevent undefined log of 0
        log_probs -= T.log(1 - action.pow(2) + self.reparameterization_noise)
        log_probs = log_probs.sum(1, keepdim=True) # If you want to use this change state -> [state] in select_action

        # log_probs = np.sum(log_probs.detach().numpy(), axis=1)

        # return action, torch.tensor(log_probs).float()
        return action, log_probs

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent:
    def __init__(self, state_space, action_space, max_action, alpha, beta, gamma, max_size, tau, batch_size,
                 reward_scale):
        self.gamma = gamma
        self.tau = tau
        self.buffer = ReplayBuffer(max_size, state_space, action_space)
        self.batch_size = batch_size
        self.action_space = action_space

        self.actor_network = ActorNetwork(alpha, state_space, action_space, max_action)
        self.critic_network_1 = CriticNetwork(beta, state_space, action_space, name='critic_1')
        self.critic_network_2 = CriticNetwork(beta, state_space, action_space, name='critic_2')
        self.value_network = ValueNetwork(beta, state_space, name='value')
        self.target_value_network = ValueNetwork(beta, state_space, name='target_value')

        self.scale = reward_scale
        self.update_target_network_parameters(tau=1)

    def select_action(self, state):
        state = T.Tensor([state]).to(self.actor_network.device)#[state]
        actions, _ = self.actor_network.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]

    def store_in_buffer(self, state, action, reward, new_state, done):
        self.buffer.store_transition(state, action, reward, new_state, done)

    def update_target_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value_network.named_parameters()
        value_params = self.value_network.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau * value_state_dict[name].clone() + \
                                     (1 - tau) * target_value_state_dict[name].clone()

        self.target_value_network.load_state_dict(value_state_dict)

    def save_models(self):
        self.actor_network.save_checkpoint()
        self.value_network.save_checkpoint()
        self.target_value_network.save_checkpoint()
        self.critic_network_1.save_checkpoint()
        self.critic_network_2.save_checkpoint()

    def load_models(self):
        self.actor_network.load_checkpoint()
        self.value_network.load_checkpoint()
        self.target_value_network.load_checkpoint()
        self.critic_network_1.load_checkpoint()
        self.critic_network_2.load_checkpoint()

    def learn(self):

        def get_critic_value_and_log_probs():
            actions, log_probs = self.actor_network.sample_normal(state, reparameterize=False)
            log_probs = log_probs.view(-1)
            q1_new_policy = self.critic_network_1(state, actions)
            q2_new_policy = self.critic_network_2(state, actions)
            critic_value = T.min(q1_new_policy, q2_new_policy)
            critic_value = critic_value.view(-1)

            return critic_value, log_probs

        if self.buffer.memory_counter < self.batch_size:
            return

        state, action, reward, new_state, done = \
            self.buffer.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.actor_network.device)
        done = T.tensor(done).to(self.actor_network.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.actor_network.device)
        state = T.tensor(state, dtype=T.float).to(self.actor_network.device)
        action = T.tensor(action, dtype=T.float).to(self.actor_network.device)

        value = self.value_network(state).view(-1)
        target_value = self.target_value_network(new_state).view(-1)
        target_value[done] = 0.0

        critic_value, log_probs = get_critic_value_and_log_probs()

        self.value_network.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value_network.optimizer.step()

        critic_value, log_probs = get_critic_value_and_log_probs()

        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor_network.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_network.optimizer.step()

        self.critic_network_1.optimizer.zero_grad()
        self.critic_network_2.optimizer.zero_grad()
        q_estimate = self.scale * reward + self.gamma * target_value
        q1_old_policy = self.critic_network_1(state, action).view(-1)
        q2_old_policy = self.critic_network_2(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_estimate)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_estimate)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_network_1.optimizer.step()
        self.critic_network_2.optimizer.step()

        self.update_target_network_parameters()


def check_intersections(bounding_box, filthy, goals, obstacles, grid):
    blocked = any([ob.intersect(bounding_box) for ob in obstacles]) or \
                          not (bounding_box.x1 >= 0 and bounding_box.x2 <= grid.width and bounding_box.y1 >= 0 and
                               bounding_box.y2 <= grid.height)

    if blocked:
        return None, None, blocked, None

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
# reward_scale is the most important parameter - entropy comes from it, encourages exploration when it is decreased,
# encourages exploitation when increased
def robot_epoch(robot, episodes=7, steps=200, state_space=4, action_space=2, max_action=1, alpha=0.001, beta=0.001,
                gamma=0.99, max_size=1000000, tau=0.005, batch_size=256, reward_scale=0.1):
    agent = Agent(state_space, action_space, max_action, alpha, beta, gamma, max_size, tau, batch_size, reward_scale)

    best_score = -9999
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()

    x_pos, y_pos = robot.pos

    for _ in range(episodes):
        state = np.array([x_pos, y_pos, x_pos+robot.size, y_pos+robot.size])
        score = 0
        # NEW:
        prior_filthy = copy.deepcopy(robot.grid.filthy)
        prior_goals = copy.deepcopy(robot.grid.goals)

        # generate an episode following policy
        for t in range(steps):
            action = agent.select_action(state)
            new_x_pos = state[0] + action[0]
            new_y_pos = state[1] + action[1]
            new_state = np.array([new_x_pos, new_y_pos, new_x_pos+robot.size, new_y_pos+robot.size])

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
                score += reward
                agent.store_in_buffer(state, action, reward, state, done)
            else:
                factor_filthy = len(prior_filthy) - len(new_filthy)
                factor_goals = len(prior_goals) - len(new_goals)
                # TODO: experiment with different factors instead of 1 and 2
                reward = 1*factor_filthy + 3*factor_goals

                score += reward

                agent.store_in_buffer(state, action, reward, new_state, done)

                state = new_state
                prior_filthy = new_filthy
                prior_goals = new_goals

            if not load_checkpoint:
                agent.learn()

            if done:
                break

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

    # obtain the best action from Q for the current state
    state = np.array([x_pos, y_pos, x_pos+robot.size, y_pos+robot.size])
    action = agent.select_action(state)
    robot.direction_vector = (action[0], action[1])
    print(robot.direction_vector)
    robot.move()