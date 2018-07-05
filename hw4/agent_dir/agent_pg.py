from agent_dir.agent import Agent
import numpy as np
import os.path
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
'''This is the inference version of code '''
def prepro(I):
    """ from openai """
    I = I[35:195]  
    I = I[::2, ::2, 0]  
    I[I == 144] = 0  
    I[I == 109] = 0  
    I[I != 0] = 1  
    return I.astype(np.float).ravel()


class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)
        self.env = env

        self.sess = tf.InteractiveSession()

        
        # +1 for up, -1 for down
        self.sampled_actions = tf.placeholder(tf.float32, [None, 1])
        self.advantage = tf.placeholder(
            tf.float32, [None, 1])
        self.observations = tf.placeholder(tf.float32, [None, 6400])
        h = tf.layers.dense(
            self.observations,
            units=256,
            activation=tf.nn.relu)

        self.up_probability = tf.layers.dense(
            h,
            units=1,
            activation=tf.sigmoid)

        self.loss = tf.losses.log_loss(
            labels=self.sampled_actions,
            predictions=self.up_probability,
            weights=self.advantage)

        optimizer = tf.train.AdamOptimizer(0.003)
        self.train_op = optimizer.minimize(self.loss)

        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        
        if args.test_pg:
            
            model_file = "./4.63/policy_network.ckpt"
            self.saver.restore(self.sess, model_file)
            

    def forward_pass(self, observations,x):
        up_probability = self.sess.run(
            self.up_probability,
            feed_dict={self.observations: observations.reshape([1, -1])})
        return up_probability + x

    def init_game_setting(self):
        
        self.observation_memory = []


    def train(self):
        
        
        action_dict = {3: 0, 2: 1}
        print(self.env.reset().shape)
        episode_n = 1
        
        batch_state_action_reward_tuples = []
        smoothed_reward = None
        learning_history = []
        last_30 = [0]*30
        while True:
            
            last_observation = self.env.reset()
            last_observation = prepro(last_observation)
            episode_done = None
            episode_reward_sum = 0
            round_n = 0
            
            action = self.env.action_space.sample()
            observation, _, _, _ = self.env.step(action)
            observation = prepro(observation)
            
            while not episode_done:
                observation_delta = observation - last_observation
                last_observation = observation
                
                
                if np.random.uniform() < self.forward_pass(observation_delta,0)[0]:
                    action = 2
                else:
                    action = 3

                observation, reward, episode_done, _ = self.env.step(action)
                observation = prepro(observation)
                episode_reward_sum += reward
               
                tup = (observation_delta, action_dict[action], reward)
                batch_state_action_reward_tuples.append(tup)
                
                if reward != 0:
                    round_n += 1
            
            
            
            if smoothed_reward is None:
                smoothed_reward = episode_reward_sum
            else:
                smoothed_reward = smoothed_reward * 0.99 + episode_reward_sum * 0.01
            
            last_30[(episode_n-1) % 30] = episode_reward_sum
            print("last 30", sum(last_30)/30)
            print(" ")

            
            if episode_n % 100 == 0:
                np.save("pg_learning_history.npy",learning_history)
            episode_n += 1
    
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)
        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        pos = 0.5
        if self.observation_memory == []:
            
            
            
            action = self.env.get_random_action()
            second_observation, _, _, _ = self.env.step(action)
            second_observation = prepro(second_observation)
            observation_delta = second_observation - prepro(observation)
            self.observation_memory = second_observation
            up_probability = self.forward_pass(observation_delta,0)[0]
            if up_probability > pos:
                action = 2
            else:
                action = 3
        else:
            GG = []
            observation = prepro(observation)
            GG.append(observation)
            observation_delta = GG[0] - self.observation_memory
            self.observation_memory = GG[0]
            up_probability = self.forward_pass(observation_delta,0)[0]
            if up_probability > pos:
                action = 2
            else:
                action = 3
        # action = self.env.get_random_action()
        #print("made one action!")
        return action