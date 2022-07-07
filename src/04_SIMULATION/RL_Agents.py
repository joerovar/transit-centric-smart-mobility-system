import numpy as np
import torch as T
from RL_Networks import DeepQNetwork, DuelingDeepQNetwork
# from replay_memory import ReplayBuffer
from copy import deepcopy


class Agent:
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-5,
                 replace=500, algo=None, chkpt_dir='tmp/dqn', fc_dims=256):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        self.batch_size = batch_size
        self.replace_target_cnt = replace
        self.algo = algo
        self.chkpt_dir = chkpt_dir
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
        self.fc_dims = fc_dims

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def choose_action(self, observation):
        raise NotImplementedError

    def q_values(self, observation):
        raise NotImplementedError

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min

    def sample_memory(self):
        state, action, reward, new_state, done = \
                                                self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.Q_eval.device)
        rewards = T.tensor(reward).to(self.Q_eval.device)
        dones = T.tensor(done).to(self.Q_eval.device)
        actions = T.tensor(action).to(self.Q_eval.device)
        states_ = T.tensor(new_state).to(self.Q_eval.device)

        return states, actions, rewards, states_, dones

    def learn(self):
        raise NotImplementedError

    def save_models(self):
        self.Q_eval.save_checkpoint()
        self.Q_next.save_checkpoint()

    def load_models(self):
        self.Q_eval.load_checkpoint()
        self.Q_next.load_checkpoint()


class DQNAgent(Agent):
    def __init__(self, *args, **kwargs):
        super(DQNAgent, self).__init__(*args, **kwargs)
        # elf, lr, input_dims, fc1_dims, fc2_dims, n_actions, chkpt_dir, name
        self.Q_eval = DeepQNetwork(self.lr, self.input_dims,
                                   fc1_dims=self.fc_dims, fc2_dims=self.fc_dims,
                                   n_actions=self.n_actions,
                                   name=self.algo+'_q_eval',
                                   chkpt_dir=self.chkpt_dir)
        self.Q_next = DeepQNetwork(self.lr, self.input_dims,
                                   n_actions=self.n_actions,
                                   fc1_dims=self.fc_dims, fc2_dims=self.fc_dims,
                                   name=self.algo + '_q_next',
                                   chkpt_dir=self.chkpt_dir)

    def choose_action(self, observation, mask_idx=None):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            if mask_idx is not None:
                actions[0][mask_idx] = float('-inf')
            action = T.argmax(actions).item()
        else:
            action_space = np.array(deepcopy(self.action_space))
            if mask_idx is not None:
                msk_arr = np.zeros(len(action_space), dtype=bool)
                msk_arr[mask_idx] = True
                action_space = action_space[~msk_arr]
            action = np.random.choice(action_space)
        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        self.Q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()

        indices = np.arange(self.batch_size)
        q_pred = self.Q_eval.forward(states)[indices, actions]
        q_next = self.Q_next.forward(states_).max(dim=1)[0]
        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next

        loss = self.Q_eval.loss(q_target, q_pred).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()


class DDQNAgent(Agent):
    def __init__(self, *args, **kwargs):
        super(DDQNAgent, self).__init__(*args, **kwargs)
        self.Q_eval = DeepQNetwork(self.lr, self.input_dims, n_actions=self.n_actions,
                                   fc1_dims=self.fc_dims, fc2_dims=self.fc_dims,
                                   name=self.algo+'_q_eval',
                                   chkpt_dir=self.chkpt_dir)
        self.Q_next = DeepQNetwork(self.lr, self.input_dims, n_actions=self.n_actions,
                                   fc1_dims=self.fc_dims, fc2_dims=self.fc_dims,
                                   name=self.algo + '_q_next',
                                   chkpt_dir=self.chkpt_dir)

    def choose_action(self, observation, mask_idx=None):
        if np.random.random() >= self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            if mask_idx is not None:
                actions[0][mask_idx] = float('-inf')
            action = T.argmax(actions).item()
        else:
            action_space = np.array(deepcopy(self.action_space))
            if mask_idx is not None:
                msk_arr = np.zeros(len(action_space), dtype=bool)
                msk_arr[mask_idx] = True
                action_space = action_space[~msk_arr]
            action = np.random.choice(action_space)

        return action

    def q_values(self, observation):
        state = T.tensor([observation]).to(self.Q_eval.device)
        q_values = self.Q_eval.forward(state)
        return q_values

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        self.Q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()

        indices = np.arange(self.batch_size)
        q_pred = self.Q_eval.forward(states)[indices, actions]
        q_next = self.Q_next.forward(states_)
        q_eval = self.Q_eval.forward(states_)
        max_actions = T.argmax(q_eval, dim=1)
        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next[indices, max_actions]

        loss = self.Q_eval.loss(q_target, q_pred).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()


class DuelingDQNAgent(Agent):
    def __init__(self, *args, **kwargs):
        super(DuelingDQNAgent, self).__init__(*args, **kwargs)

        self.Q_eval = DuelingDeepQNetwork(self.lr, self.input_dims,n_actions=self.n_actions,
                                          fc1_dims=self.fc_dims, fc2_dims=self.fc_dims,
                                          name=self.algo + '_q_eval',
                                          chkpt_dir=self.chkpt_dir)
        self.Q_next = DuelingDeepQNetwork(self.lr, self.input_dims,n_actions=self.n_actions,
                                          fc1_dims=self.fc_dims, fc2_dims=self.fc_dims,
                                          name=self.algo + '_q_next',
                                          chkpt_dir=self.chkpt_dir)

    def choose_action(self, observation, mask_idx=None):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            _, advantage = self.Q_eval.forward(state)
            if mask_idx is not None:
                for i in mask_idx:
                    advantage[0][i] = float('-inf')
            action = T.argmax(advantage).item()
        else:
            action_space = np.array(deepcopy(self.action_space))
            if mask_idx is not None:
                msk_arr = np.zeros(len(action_space), dtype=bool)
                for i in mask_idx:
                    msk_arr[i] = True
                action_space = action_space[~msk_arr]
            action = np.random.choice(action_space)

        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        self.Q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()

        indices = np.arange(self.batch_size)

        V_s, A_s = self.Q_eval.forward(states)
        V_s_, A_s_ = self.Q_eval.forward(states_)

        q_pred = T.add(V_s,
                       (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = T.add(V_s_,
                       (A_s_ - A_s_.mean(dim=1, keepdim=True))).max(dim=1)[0]
        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next

        loss = self.Q_eval.loss(q_target, q_pred).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()


class DuelingDDQNAgent(Agent):
    def __init__(self, *args, **kwargs):
        super(DuelingDDQNAgent, self).__init__(*args,**kwargs)

        self.Q_eval = DuelingDeepQNetwork(self.lr, self.input_dims,n_actions=self.n_actions,
                                          fc1_dims=self.fc_dims, fc2_dims=self.fc_dims,
                                          name=self.algo+'_q_eval',
                                          chkpt_dir=self.chkpt_dir)
        self.Q_next = DuelingDeepQNetwork(self.lr, self.input_dims,n_actions=self.n_actions,
                                          fc1_dims=self.fc_dims, fc2_dims=self.fc_dims,
                                          name=self.algo + '_q_next',
                                          chkpt_dir=self.chkpt_dir)

    def choose_action(self, observation, mask_idx=None):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            _, advantage = self.Q_eval.forward(state)
            if mask_idx is not None:
                advantage[0][mask_idx] = float('-inf')
            action = T.argmax(advantage).item()
        else:
            action_space = np.array(deepcopy(self.action_space))
            if mask_idx is not None:
                msk_arr = np.zeros(len(action_space), dtype=bool)
                msk_arr[mask_idx] = True
                action_space = action_space[~msk_arr]
            action = np.random.choice(action_space)
        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        self.Q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()

        indices = np.arange(self.batch_size)

        V_s, A_s = self.Q_eval.forward(states)
        V_s_, A_s_ = self.Q_eval.forward(states_)
        V_s_eval, A_s_eval = self.Q_eval.forward(states_)
        q_pred = T.add(V_s,
                       (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = T.add(V_s_,
                       (A_s_ - A_s_.mean(dim=1, keepdim=True)))
        q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))
        max_actions = T.argmax(q_eval, dim=1)
        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next[indices, max_actions]

        loss = self.Q_eval.loss(q_target, q_pred).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                         dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


