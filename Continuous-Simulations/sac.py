import os

import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
import copy
from continuous import SimGridComplex as SimGrid


class ReplayBuffer:
    """
    Class implementing memory buffer to store contents a batch size of episode elements
    """
    def __init__(self, max_memory_size, state_space, action_space):
        """
        Initialize the buffer
        :param max_memory_size: maximum number of elements in batch
        :param state_space: the dimensionality of state (x,y)
        :param action_space: the dimensionality of action displacement vector (x,y)
        """
        self.memory_size = max_memory_size
        self.memory_counter = 0
        self.state_memory = np.zeros((self.memory_size, state_space))
        self.new_state_memory = np.zeros((self.memory_size, state_space))
        self.action_memory = np.zeros((self.memory_size, action_space))
        self.reward_memory = np.zeros(self.memory_size)
        self.terminal_memory = np.zeros(self.memory_size, dtype=np.bool)

    def store_transition(self, state, action, reward, new_state, done):
        """
        Store step of an episode in buffer
        :param state: a state (x,y)
        :param action: an action displacement vector (x,y)
        :param reward: reward for taking the action in the state
        :param new_state: the next state (x1,y1)
        :param done: boolean indicating if agent is done cleaning
        """
        index = self.memory_counter % self.memory_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.memory_counter += 1

    def sample_buffer(self, batch_size):
        """
        Sample a batch_size number of steps of an episode from buffer
        :param batch_size: the batch size
        """
        max_memory = min(self.memory_counter, self.memory_size)
        batch = np.random.choice(max_memory, batch_size)
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, new_states, dones


class CriticNetwork(nn.Module):
    """
    Class implementing a Neural Network approximator for the critic
    """
    def __init__(self, beta, state_space, action_space, fc1_dims=256, fc2_dims=256):
        """
        Initializes the network's layers
        :param state_space: the dimensionality of state (x,y)
        :param action_space: the dimensionality of action displacement vector (x,y)
        :param fc1_dims: the number of output features of the 1st linear layer
        :param fc2_dims: the number of input features of the 1st linear layer
        """
        super(CriticNetwork, self).__init__()

        self.fc1 = nn.Linear(state_space+action_space, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q_layer = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        """
        Perform a forward pass through the network
        :param state: a state (x,y)
        :param action: an action displacement vector (x,y)
        :returns q_estimate: evaluation of state-action pair
        """
        x = self.fc1(T.cat([state, action], dim=1))
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        q_estimate = self.q_layer(x)

        return q_estimate


class ValueNetwork(nn.Module):
    """
    Class implementing a Neural Network approximator for the state value function
    """
    def __init__(self, beta, state_space, fc1_dims=256, fc2_dims=256):
        """
        Initializes the network's layers
        :param state_space: the dimensionality of state (x,y)
        :param fc1_dims: the number of output features of the 1st linear layer and input features of 2nd layer
        :param fc2_dims: the number of output features of the 2nd linear layer and input features of 3rd layer
        """
        super(ValueNetwork, self).__init__()

        self.fc1 = nn.Linear(state_space, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.value_layer = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        """
        Perform a forward pass through the network
        :param state: a state (x,y)
        :returns value_estimate: the evaluation of state
        """
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        value_estimate = self.value_layer(x)

        return value_estimate


class ActorNetwork(nn.Module):
    """
    Class implementing a Neural Network approximator for the actor
    """
    def __init__(self, alpha, state_space, action_space, max_action_value, fc1_dims=256, fc2_dims=256):
        """
        Initializes the network's layers
        :param state_space: the dimensionality of state (x,y)
        :param action_space: the dimensionality of action displacement vector (x,y)
        :param max_action_value: the maximum value of an action displacement vector coordinate
        :param fc1_dims: the number of output features of the 1st linear layer and input features of 2nd layer
        :param fc2_dims: the number of output features of the 2nd linear layer and input features of 3rd and 4th layer
        """
        super(ActorNetwork, self).__init__()

        self.fc1 = nn.Linear(state_space, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mean = nn.Linear(fc2_dims, action_space)
        self.std = nn.Linear(fc2_dims, action_space)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.reparameterization_noise = 1e-6
        self.max_action_value = max_action_value
        self.to(self.device)

    def forward(self, state):
        """
        Perform a forward pass through the network
        :param state: a state (x,y)
        :returns mean, std: the mean and standard deviation used to generate a Normal distribution
        """
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
        """
        Sample an action and its log probability from a Normal distribution
        :param state: a state (x,y)
        :param reparameterize: a boolean indicating if parametrization noise should be applied to encourage exploration
        :returns mean, std: the mean and standard deviation used to generate a Normal distribution
        """
        mean, std = self.forward(state)
        probabilities = Normal(mean, std)

        if reparameterize:
            # add noise for exploration
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        # use invertable squashing function to bound actions to a finite interval and
        # multiply by max_action_value to fix action coordinates in the appropriate range
        action = T.tanh(actions)*T.tensor(self.max_action_value).to(self.device)
        # get log of the probabilities for the computation of the loss
        log_probs = probabilities.log_prob(actions)
        # add noise to prevent undefined log of 0
        log_probs -= T.log(1 - action.pow(2) + self.reparameterization_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs


class Agent:
    """
    Class implementing the SAC agent
    """
    def __init__(self, state_space, action_space, max_action, alpha, beta, gamma, max_size, tau, batch_size,
                 reward_scale):
        """
        Initializes the agent's neural network approximators
        :param state_space: the dimensionality of state (x,y)
        :param action_space: the dimensionality of action displacement vector (x,y)
        :param max_action: the maximum value of an action displacement vector coordinate
        :param alpha: learning rate of network
        :param beta: learning rate of network
        :param gamma: discount factor
        :param max_size: size of buffer
        :param tau: controls the soft copy of the value network
        :param batch_size: maximum number of elements in batch
        :param reward_scale: encourages exploration when it is decreased, encourages exploitation when increased
        """
        self.gamma = gamma
        self.tau = tau
        self.buffer = ReplayBuffer(max_size, state_space, action_space)
        self.batch_size = batch_size
        self.action_space = action_space

        self.actor_network = ActorNetwork(alpha, state_space, action_space, max_action)
        self.critic_network_1 = CriticNetwork(beta, state_space, action_space)
        self.critic_network_2 = CriticNetwork(beta, state_space, action_space)
        self.value_network = ValueNetwork(beta, state_space)
        self.target_value_network = ValueNetwork(beta, state_space)

        self.scale = reward_scale
        self.update_target_network_parameters(tau=1)

    def select_action(self, state):
        """
        Sample an action by passing state through actor_network
        :param state: a state (x,y)
        :returns action: the sampled action
        """
        state = T.Tensor(state).unsqueeze(0).to(self.actor_network.device)
        action, _ = self.actor_network.sample_normal(state, reparameterize=False)

        return action.cpu().detach().numpy()[0]

    def store_in_buffer(self, state, action, reward, new_state, done):
        """
         Store a step in buffer
         :param state: a state (x,y)
         :param action: an action displacement vector (x,y)
         :param reward: the reward for taking an action in state
         :param new_state: the new state
         :param done: boolean indicating whether cleaning/episode was done
         """
        self.buffer.store_transition(state, action, reward, new_state, done)

    def update_target_network_parameters(self, tau=None):
        """
        Update target value network parameters
        :param tau: if not None make an exact copy of the value network for the target value network,
                    else do a soft copy based on tau's value
        """
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

    def learn(self):
        """
        Update the parameters of the neural network approximators
        """
        def get_critic_value_and_log_probs():
            """
            Get min evaluation of the 2 critic networks for the state-action pairs and
            the log_probability of the sampled action from the actor_network given a state
            """
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
        # if episode is finished set target network to 0
        target_value[done] = 0.0

        critic_value, log_probs = get_critic_value_and_log_probs()

        # update value_network parameters by backpropagation
        self.value_network.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value_network.optimizer.step()

        critic_value, log_probs = get_critic_value_and_log_probs()

        # update actor_network parameters by backpropagation
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

        # update the critic networks' parameters by backpropagation
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_network_1.optimizer.step()
        self.critic_network_2.optimizer.step()

        self.update_target_network_parameters()


def robot_epoch(robot, gamma=0.9, alpha=0.001, state_space=2, action_space=2, max_action=1, beta=0.001,
                 max_size=1000000, tau=0.005, batch_size=5, reward_scale=2, episodes=3, steps=10):
    """
    Execute SAC algorithm to find the best move
    :param robot: main actor of type Robot
    :param gamma: discount factor
    :param alpha: learning rate actor network
    :param state_space: the dimensionality of state (x,y)
    :param action_space: the dimensionality of action displacement vector (x,y)
    :param max_action: the maximum value of an action displacement vector coordinate
    :param beta: learning rate for critic and value network
    :param max_size: the max size of the buffer
    :param tau: controls the soft copy of the value network
    :param batch_size: maximum number of elements in batch
    :param reward_scale: encourages exploration when it is decreased, encourages exploitation when increased
    :param episodes: number of episodes
    :param steps: number of steps
    """
    agent = Agent(state_space, action_space, max_action, alpha, beta, gamma, max_size, tau, batch_size, reward_scale)

    x_pos, y_pos = robot.pos

    for _ in range(episodes):
        state = np.array([x_pos, y_pos])
        score = 0
        prior_filthy = copy.deepcopy(robot.grid.filthy)
        prior_goals = copy.deepcopy(robot.grid.goals)

        # generate an episode following policy
        for t in range(steps):
            action = agent.select_action(state)
            new_x_pos = state[0] + action[0]
            new_y_pos = state[1] + action[1]
            new_state = np.array([new_x_pos, new_y_pos])

            # create simulation grid to get the result of a move
            test = SimGrid(new_state, robot.grid, prior_filthy, prior_goals)
            reward, is_blocked, done, new_filthy, new_goals = test.reward(action)
            score += reward

            if is_blocked:
                # do not update the state to the new state
                agent.store_in_buffer(state, action, reward, state, done)
            else:
                agent.store_in_buffer(state, action, reward, new_state, done)
                state = new_state
                prior_filthy = new_filthy
                prior_goals = new_goals

            agent.learn()

            if done:
                break

    # obtain the best action from the sac agent
    state = np.array([x_pos, y_pos])
    action = agent.select_action(state)
    robot.direction_vector = (action[0], action[1])
    robot.move()