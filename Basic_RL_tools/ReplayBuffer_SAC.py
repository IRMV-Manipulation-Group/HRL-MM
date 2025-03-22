import numpy as np
import os

class Simple_ReplayBuffer():
    # state, action, reward, next_state, done, gamma
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.pos = 0
        # Indicate whether overflow the capacity
        self.overflow_flag = 0
        self.state_matrix = np.zeros((capacity, state_dim))
        self.action_matrix = np.zeros((capacity, action_dim))
        self.reward_matrix = np.zeros(capacity)
        self.next_state_matrix = np.zeros((capacity,state_dim))
        self.done_matrix = np.zeros(capacity, dtype=bool)
        self.gamma_matrix = np.zeros(capacity)
            
    def push(self, state, action, reward, next_state, done, gamma):
        self.state_matrix[self.pos] = state
        self.action_matrix[self.pos] = action
        self.reward_matrix[self.pos] = reward
        self.next_state_matrix[self.pos] = next_state
        self.done_matrix[self.pos] = done
        self.gamma_matrix[self.pos] = gamma
        if self.pos + 1 == self.capacity:
            self.overflow_flag = 1
        self.pos = (self.pos + 1) % self.capacity
    
    def pushes(self, states, actions, rewards, next_states, dones, gammas):
        # Notice: the data length must be smaller than the capacity
        data_len = states.shape[0]
        if self.pos+data_len < self.capacity:
            self.state_matrix[self.pos:self.pos+data_len,:] = states
            self.action_matrix[self.pos:self.pos+data_len,:] = actions
            self.reward_matrix[self.pos:self.pos+data_len] = rewards
            self.next_state_matrix[self.pos:self.pos+data_len,:] = next_states
            self.done_matrix[self.pos:self.pos+data_len] = dones
            self.gamma_matrix[self.pos:self.pos+data_len] = gammas

        else:
            self.state_matrix[self.pos:self.capacity,:] = states[:self.capacity-self.pos,:]
            self.action_matrix[self.pos:self.capacity,:] = actions[:self.capacity-self.pos,:]
            self.reward_matrix[self.pos:self.capacity] = rewards[:self.capacity-self.pos]
            self.next_state_matrix[self.pos:self.capacity,:] = next_states[:self.capacity-self.pos,:]
            self.done_matrix[self.pos:self.capacity] = dones[:self.capacity-self.pos]
            self.gamma_matrix[self.pos:self.capacity] = gammas[:self.capacity-self.pos]
            self.overflow_flag = 1
            # ----------------
            self.state_matrix[0:data_len-self.capacity+self.pos,:] = states[self.capacity-self.pos:,:]
            self.action_matrix[0:data_len-self.capacity+self.pos,:] = actions[self.capacity-self.pos:,:]
            self.reward_matrix[0:data_len-self.capacity+self.pos] = rewards[self.capacity-self.pos:]
            self.next_state_matrix[0:data_len-self.capacity+self.pos,:] = next_states[self.capacity-self.pos:,:]
            self.done_matrix[0:data_len-self.capacity+self.pos] = dones[self.capacity-self.pos:]
            self.gamma_matrix[0:data_len-self.capacity+self.pos] = gammas[self.capacity-self.pos:]

        self.pos = (self.pos + data_len) % self.capacity

    
    def sample(self, batch_size):
        N = self.capacity if self.overflow_flag else self.pos

        indices = np.random.choice(N, batch_size)
        sample_state_matrix = self.state_matrix[indices,:]
        sample_action_matrix = self.action_matrix[indices,:]
        sample_reward_matrix = self.reward_matrix[indices]
        sample_next_state_matrix = self.next_state_matrix[indices,:]
        sample_done_matrix = self.done_matrix[indices]
        sample_gamma_matrix = self.gamma_matrix[indices]
        
        return sample_state_matrix, sample_action_matrix, sample_reward_matrix.reshape(-1, 1), \
                sample_next_state_matrix, sample_done_matrix.reshape(-1,1), sample_gamma_matrix.reshape(-1,1)
    
    def save(self, directory, mode):
        np.save(os.path.join(directory, mode, 'state_matrix.npy'), self.state_matrix)
        np.save(os.path.join(directory, mode, 'action_matrix.npy'), self.action_matrix)
        np.save(os.path.join(directory, mode, 'reward_matrix.npy'), self.reward_matrix)
        np.save(os.path.join(directory, mode, 'next_state_matrix.npy'), self.next_state_matrix)
        np.save(os.path.join(directory, mode, 'done_matrix.npy'), self.done_matrix)
        np.save(os.path.join(directory, mode, 'gamma_matrix.npy'), self.gamma_matrix)
        np.save(os.path.join(directory, mode, 'overflow_flag.npy'), self.overflow_flag)
        np.save(os.path.join(directory, mode, 'pos.npy'), self.pos)

    def load(self, directory, mode):
        self.state_matrix = np.load(os.path.join(directory, mode, 'state_matrix.npy'))
        self.action_matrix = np.load(os.path.join(directory, mode, 'action_matrix.npy'))
        self.reward_matrix = np.load(os.path.join(directory, mode, 'reward_matrix.npy'))
        self.next_state_matrix = np.load(os.path.join(directory, mode, 'next_state_matrix.npy'))
        self.done_matrix = np.load(os.path.join(directory, mode, 'done_matrix.npy'))
        self.gamma_matrix = np.load(os.path.join(directory, mode, 'gamma_matrix.npy'))
        self.overflow_flag = np.load(os.path.join(directory, mode, 'overflow_flag.npy'))
        self.pos = np.load(os.path.join(directory, mode, 'pos.npy'))
    
    def sample_near(self, batch_size, near_number):
        if near_number <= self.pos:
            L = range(self.pos-near_number, self.pos) 
        else:
            if self.overflow_flag:
                L = np.concatenate((range(self.pos), range(self.capacity-(near_number - self.pos) ,self.capacity)))
            else:
                L = range(self.pos)
        # N = self.capacity if self.overflow_flag else self.pos
        indices = np.random.choice(L, batch_size)
        sample_state_matrix = self.state_matrix[indices,:]
        sample_action_matrix = self.action_matrix[indices,:]
        sample_reward_matrix = self.reward_matrix[indices]
        sample_next_state_matrix = self.next_state_matrix[indices,:]
        sample_done_matrix = self.done_matrix[indices]
        sample_gamma_matrix = self.gamma_matrix[indices]
        
        return sample_state_matrix, sample_action_matrix, sample_reward_matrix.reshape(-1, 1), \
                sample_next_state_matrix, sample_done_matrix.reshape(-1,1), sample_gamma_matrix.reshape(-1,1)
    
class Additional_ReplayBuffer(Simple_ReplayBuffer):
    # state, action, reward, next_state, done, gamma
    def __init__(self, capacity, state_dim, action_dim, extra_dim):
        super().__init__(capacity, state_dim, action_dim)
        self.extra_matrix = np.zeros((capacity, extra_dim))
        self.next_extra_matrix = np.zeros((capacity, extra_dim))

         
    def push(self, state, extra, action, reward, next_state, next_extra, done, gamma):
        super().push(state, action, reward, next_state, done, gamma)
        self.extra_matrix[self.pos] = extra
        self.next_extra_matrix[self.pos] = next_extra

    
    def pushes(self, states, extras, actions, rewards, next_states, next_extras, dones, gammas):
        super().pushes(states, actions, rewards, next_states, dones, gammas)
        # Notice: the data length must be smaller than the capacity
        data_len = states.shape[0]

        if self.pos+data_len < self.capacity:
            self.extra_matrix[self.pos:self.pos+data_len,:] = extras
            self.next_extra_matrix[self.pos:self.pos+data_len,:] = next_extras


        else:
            self.extra_matrix[self.pos:self.capacity,:] = extras[:self.capacity-self.pos,:]
            self.next_extra_matrix[self.pos:self.capacity,:] = next_extras[:self.capacity-self.pos,:]
            # ----------------
            self.extra_matrix[0:data_len-self.capacity+self.pos,:] = extras[self.capacity-self.pos:,:]
            self.next_extra_matrix[0:data_len-self.capacity+self.pos,:] = next_extras[self.capacity-self.pos:,:]
    

    
    def sample(self, batch_size):
        N = self.capacity if self.overflow_flag else self.pos

        indices = np.random.choice(N, batch_size)

        sample_state_matrix = self.state_matrix[indices,:]
        sample_extra_matrix = self.extra_matrix[indices,:]

        sample_action_matrix = self.action_matrix[indices,:]
        sample_reward_matrix = self.reward_matrix[indices]
        sample_next_state_matrix = self.next_state_matrix[indices,:]        
        sample_next_extra_matrix = self.next_extra_matrix[indices,:]

        sample_done_matrix = self.done_matrix[indices]
        sample_gamma_matrix = self.gamma_matrix[indices]
        
        return sample_state_matrix, sample_extra_matrix, sample_action_matrix, sample_reward_matrix.reshape(-1, 1), \
                sample_next_state_matrix, sample_next_extra_matrix, sample_done_matrix.reshape(-1,1), sample_gamma_matrix.reshape(-1,1)
    
    def save(self, directory, mode):
        super().save(directory, mode)
        np.save(os.path.join(directory, mode, 'extra_matrix.npy'), self.extra_matrix)
        np.save(os.path.join(directory, mode, 'next_extra_matrix.npy'), self.next_extra_matrix)


    def load(self, directory, mode):
        super().load(directory, mode)
        self.extra_matrix = np.load(os.path.join(directory, mode, 'extra_matrix.npy'))
        self.next_extra_matrix = np.load(os.path.join(directory, mode, 'next_extra_matrix.npy'))
    