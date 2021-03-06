import random
from collections import defaultdict
import numpy as np

class QAgent():
    def __init__(self, env, gamma):

        self.gamma = 0.9
        self.env = env
        self.q_vals = defaultdict(lambda: np.array([0. for _ in range(env.action_space.n)]))
        self.alpha = 0.2
        self.eps = 0.5
    #function of choosing action
    def choose_action(self, state):
        if random.random() < self.eps:
            # randomly select action from state
            action = np.random.choice(len(self.q_vals[state]))
        else:
            # greedily select action from state
            action = np.argmax(self.q_vals[state])
        return action
    #learning function
    def learn(self, cur_state, action, reward, next_state):

        #create new q value on this state and action
        max_q_new = np.max(self.q_vals[next_state])
        new_value = reward + self.gamma * max_q_new

        old_qval = self.q_vals[cur_state][action]
        self.q_vals[cur_state][action] = old_qval + self.alpha * (new_value - old_qval)