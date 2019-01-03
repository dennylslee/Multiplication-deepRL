import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from collections import deque

class MultiplyENV:

    def __init__ (self):
        self.num1 = 0
        self.num2 = 0
        self.product = 0
        self.productlist = []
        self.predict_digit = [0, 0, 0, 0]
        self.num1_digit10, self.num1_digit1, self.num2_digit10, self.num2_digit1 = 0, 0, 0, 0 
        self.totalreward = 0
        self.match =0                                         # a single indicator for a match for the state
        self.digit_match = [0, 0, 0, 0]
        self.digit_pos = 3                                  # position of digit the product prediction is working on
        self.state = np.array([[\
            self.num1_digit10, self.num1_digit1, \
            self.num2_digit10, self.num2_digit1, \
            self.digit_pos] +  self.digit_match \
            ])
        self.action_space = [0] * 10                        # one hot encoder for 10 digits

    def reset (self):
        # generate two numbers for each episode
        self.num1, self.num2 = random.randint(1, 2), random.randint(1,2)
        self.product = self.num1 * self.num2
        self.num1_digit10 = self.num1 // 10  
        self.num1_digit1  = self.num1 % (self.num1_digit10 * 10) if self.num1_digit10 != 0 else self.num1
        self.num2_digit10 = self.num2 // 10  
        self.num2_digit1  = self.num2 % (self.num2_digit10 * 10) if self.num2_digit10 != 0 else self.num2
        self.match, self.digit_pos = 0, 3
        self.predict_digit, self.digit_match = [0] * 4,  [0] * 4
        # digit-ize the real product value
        product_digit1000 = self.product // 1000
        product_digit100  = (self.product - (product_digit1000 * 1000)) // 100
        product_digit10   = (self.product - (product_digit1000 * 1000) - (product_digit100 * 100)) // 10
        product_digit1    = (self.product - (product_digit1000 * 1000) - (product_digit100 * 100) - (product_digit10 * 10)) // 1
        self.productlist = [product_digit1000, product_digit100, product_digit10, product_digit1]
        print('First num: %s Second num: %s Product: %s %s' % (self.num1, self.num2, self.product, self.productlist))
        return np.array([\
            self.num1_digit10, self.num1_digit1, \
            self.num2_digit10, self.num2_digit1, \
            self.digit_pos] + self.digit_match \
            ).reshape(1, self.state.shape[1])       

    def step (self, action, step):       
        reward, sum, self.match, done = 0, 0, 0, False
        # check for matches and assign reward; action is in one hot format
        self.predict_digit[self.digit_pos] = np.argmax(np.array(action))
        actual_digit = self.productlist[self.digit_pos]
        #print('STEP %s Current position %s Actual digit %s Predict digit %s' % \
        #    (step, self.digit_pos, actual_digit, self.predict_digit[self.digit_pos]))
        if self.predict_digit[self.digit_pos] == actual_digit:      # one hot format of the product digit this round is on                              
            # self.match = actual_digit
            self.digit_match[self.digit_pos] = actual_digit
            reward = 1
            self.digit_pos -= 1 if self.digit_pos >= 1 else 3       # move to next digit
            done = True
        else:
            pass # reward = -0.1 
        self.totalreward += reward
        # form the new state
        self.state = np.array([\
            self.num1_digit10, self.num1_digit1, \
            self.num2_digit10, self.num2_digit1, \
            self.digit_pos] + self.digit_match \
            ).reshape(1, self.state.shape[1])
        #print('state: ', self.state)
        #print('reward: ', reward)
        print('Done status: ', done, self.predict_digit)
        return self.state, reward, done

class DQN:
    def __init__(self, env):
        self.env     = env
        self.memory  = deque(maxlen=5000)
        
        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = .125

        self.model        = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model   = Sequential()
        state_shape  = self.env.state.shape[1]
        model.add(Dense(18, input_dim=state_shape, activation="relu"))
        model.add(Dense(18, activation="relu"))
        model.add(Dense(18, activation="relu"))
        model.add(Dense(len(self.env.action_space), activation='softmax'))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        print(model.summary())
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            action_1hot = [0]*10
            action_1hot[random.randint(0, 9)] = 1        # random assign a digit in one-hot encoding format
        else:
            action_1hot = [0]*10                         # action from action model in one-hot encoding format
            action_1hot[np.argmax(np.array(self.model.predict(state)[0]))] = 1
        return action_1hot

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size: return
        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            action_index = np.argmax(np.array(action))
            target = self.target_model.predict(state)
            Q_future = max(self.target_model.predict(new_state)[0])
            target[0][action_index] = reward + Q_future * self.gamma
            #if not done:
            #    pass # target[0][action_index] = reward
            #else:
                #print('IN Q-LEARNING ... state:', state, ' new state:', new_state, \
                #    'action_index:', action_index)
                #print('target pre-Q:  ', target)
                #Q_future = max(self.target_model.predict(new_state)[0])
                #target[0][action_index] = reward + Q_future * self.gamma
                #print('target POST-Q: ', target)
            self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)

def main():
    env     = MultiplyENV()
    dqn_agent = DQN(env=env)
    # setup length of play
    episodes  = 130
    episode_len = 20
    # start the learning loops
    for episode in range(episodes):
        print('EPISODE ', episode)
        cur_state = env.reset()
        # dqn_agent.epsilon = 1.0 
        for step in range(episode_len):
            action = dqn_agent.act(cur_state)
            new_state, reward, done = env.step(action, step)
            dqn_agent.remember(cur_state, action, reward, new_state, done)           
            dqn_agent.replay()       # internally iterates default (prediction) model
            dqn_agent.target_train() # iterates target model
            cur_state = new_state
            if done: break
        # print('end state: ', cur_state)
        print('predicted digits: ', env.predict_digit, ' total reward earned: ', env.totalreward, \
            'end step: ', step)
        #if step >= 199:
        #    print("Failed to complete in episode {}".format(episode))
        #    if step % 10 == 0:
        #       dqn_agent.save_model("episode-{}.model".format(episode))
        #else:
        #    print("Completed in {} episodes".format(episode))
        #    dqn_agent.save_model("success.model")
        #    break

if __name__ == "__main__":
    main()


