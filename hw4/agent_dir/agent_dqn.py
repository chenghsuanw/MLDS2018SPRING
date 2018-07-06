from agent_dir.agent import Agent

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import random
from collections import deque

BATCH_SIZE = 32
GAMMA = 0.99                 
MEMORY_CAPACITY = 500000

INITIAL_EPSILON = 1
FINAL_EPSILON = 0.02

TARGET_REPLACE_ITER = 1000
EVAL_REPLACE_ITER = 4

LR = 1.5e-4
N_EPISODES = 100000
MAX_N_STEP = 10000

OBS = 0
EXP = 80000

cuda = True if torch.cuda.is_available() else False
print('using_cuda:', cuda)


class Net(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(in_features=7*7*64, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_actions)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Dueling_DQN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(Dueling_DQN, self).__init__()
        self.num_actions = num_actions
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.fc1_adv = nn.Linear(in_features=7*7*64, out_features=512)
        self.fc1_val = nn.Linear(in_features=7*7*64, out_features=512)

        self.fc2_adv = nn.Linear(in_features=512, out_features=num_actions)
        self.fc2_val = nn.Linear(in_features=512, out_features=1)

        self.relu = nn.ReLU()


    def forward(self, x):
        batch_size = x.size(0)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        adv = self.relu(self.fc1_adv(x))
        val = self.relu(self.fc1_val(x))

        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(x.size(0), self.num_actions)
        
        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)
        return x
    

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN,self).__init__(env)
        
        self.env = env
        self.args = args
        self.memory_counter = 0
        self.replayMemory = deque()
        self.n_actions = env.action_space.n
        self.timestep = 0
        self.epsilon = INITIAL_EPSILON
        
        print(self.env.action_space.n)                        # 4
        print(self.env.observation_space.shape)               # (84, 84, 4)

        in_channels = self.env.observation_space.shape[-1]
        # self.eval_net, self.target_net = Net(in_channels, self.n_actions), Net(in_channels, self.n_actions)
        self.eval_net, self.target_net = Dueling_DQN(in_channels, self.n_actions), Dueling_DQN(in_channels, self.n_actions)

        if cuda:
            self.eval_net = self.eval_net.cuda()
            self.target_net = self.target_net.cuda()

        if args.test_dqn:
            state_dict = torch.load('./model/Dualing_DQN_1064000.pkl')
            self.eval_net.load_state_dict(state_dict)
            self.target_net.load_state_dict(state_dict)
            print('loading trained model')

        # state_dict = torch.load('./model_11/Dualing_DQN_1064000.pkl')
        # self.eval_net.load_state_dict(state_dict)
        # self.target_net.load_state_dict(state_dict)

        # self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.optimizer = torch.optim.RMSprop(self.eval_net.parameters(), lr=0.00025, eps=0.01)
        self.loss_func = nn.MSELoss()
        
        

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass


    def train(self):
        """
        Implement your training algorithm here
        """
        
        learning_history = []
        
        for e in range(N_EPISODES):
            current_state = self.env.reset()
            total_reward = 0
            step_cnt = 0
            
            for s in range(MAX_N_STEP):
                action = self.make_action(current_state, test=False)
                next_state, reward, done, _ = self.env.step(action)
                self.store_transition(current_state, action, reward, next_state, done)
                current_state = next_state
                
                total_reward += reward
                step_cnt += 1
                
                if done:
                    learning_history.append((e, step_cnt, total_reward, self.timestep))
                    break

            if e % 50 == 0:
                np.save('./result/dqn_learning_history_12.npy', np.array(learning_history))
                print('episode: {}, total_reward: {}, step_cnt: {}, epsilon: {}, memory: {}'.format(e, total_reward, step_cnt, self.epsilon, len(self.replayMemory)))

                
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        observation = Variable(torch.cuda.FloatTensor(observation.reshape((1,84,84,4))).transpose(1, 3).transpose(2, 3))
        # Q_value = np.random.random(self.n_actions)
        Q_value = self.eval_net.forward(observation).data
        
        threshold = 0.01 if test else self.epsilon

        if np.random.random() > threshold:
            action = np.argmax(Q_value)      
        else:
            action = np.random.randint(0, self.n_actions)
        
        if self.epsilon > FINAL_EPSILON and self.timestep > OBS:
            self.epsilon -= (self.epsilon - FINAL_EPSILON)/EXP
        
        return action
    
                                         
    def store_transition(self, s, a, r, s_, done):
        one_hot_action = np.zeros(self.n_actions)
        one_hot_action[a] = 1
        self.replayMemory.append((s, a, r, s_, done))
                                         
        if len(self.replayMemory) > MEMORY_CAPACITY:
            self.replayMemory.popleft()
        if len(self.replayMemory) > BATCH_SIZE:
            if self.timestep % 4 == 0:
                self.train_Q_network()
        self.timestep += 1
    
                                         
    def train_Q_network(self):
        if self.timestep % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        fTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        lTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

        minibatch = random.sample(self.replayMemory, BATCH_SIZE)
        b_s = fTensor(np.asarray([data[0] for data in minibatch])).transpose(1, 3).transpose(2, 3)
        b_a = lTensor(np.asarray([data[1] for data in minibatch]))
        b_r = fTensor(np.asarray([data[2] for data in minibatch]))
        b_s_ = fTensor(np.asarray([data[3] for data in minibatch])).transpose(1, 3).transpose(2, 3)
        b_d = fTensor(np.asarray([1 if data[4] == 0 else 1 for data in minibatch ]))

        q_eval = self.eval_net(Variable(b_s)).gather(1, Variable(b_a.view(-1, 1)))
        q_next = self.target_net(Variable(b_s_)).detach()

        # print(b_d)
        # print((GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)).data)
        # print((GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1) * Variable(b_d.view(-1, 1))).data)
        q_target = Variable(b_r.view(-1, 1)) + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1) * Variable(b_d.view(-1, 1))

        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.timestep % 2000 == 0:
            torch.save(self.target_net.state_dict(), './model_12/Dualing_DQN_{}.pkl'.format(self.timestep))