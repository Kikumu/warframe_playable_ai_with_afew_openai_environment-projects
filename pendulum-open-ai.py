import gym
import math
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

LR = 5e-3

env = gym.make('Pendulum-v0')

env.reset()

goal_steps = 1000

# required_angle_range = range(np.pi/9, np.pi/8)

data1 = float(np.pi/8)
data2 = float(np.pi/9)


initial_games = 10000


# def see_environment():
#     for episode in range(100):
#
#         env.reset()
#         for t in range(goal_steps):
#             #env.render()
#             action = env.action_space.sample()
#             observation, reward, done, info = env.step(action)
#             #print('action', action)
#
#             if done:
#                 break
#
# see_environment()


# def __init__(self):
#     self.max_theta = np.pi / 8  # rad
#     self.max_thetadot = 0.5  # rad/sec
#     self.max_torque = 300  # N-m
#     self.dt = 0.01
#     self.viewer = None
#
#     bounds = np.array([self.max_theta, self.max_thetadot])
#
#     self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
#
#     self.observation_space = spaces.Box(low=-bounds, high=bounds, dtype=np.float32)



def initial_data():
    #some_games_first()
    training_data = []
    scores = []
    accepted_scores = []
    # only store data if score is above 50
    # actual game
    for _ in range(100):
        env.reset()
        angle = 0
        game_memory = []  # store movements
        previous_observation = []
        for _ in range(goal_steps):
            action = env.action_space.sample()  # review this
            observation, reward, done, info = env.step(action)
            if action[0] in range(float(data1), float(data2)):
                print('action', action)
            # print('type', type(action[0]))
            # print('action', action[0])
            if len(previous_observation) > 0:
                game_memory.append([previous_observation, action])
                # print('game_memory', game_memory)
            previous_observation = observation
            angle += observation
            if done:
                break
          #when we are looking to analyse game in this case, we want it to remain within a constraint of a certain angle
        # if action[0] in range(data1, data2):
        #     print('action', action)
        #     accepted_scores.append(score)
        #     for data in game_memory:
        #         print('data', data)
        #         if data[1] == 1:
        #             output = [0, 1]
        #         elif data[1] == 0:
        #             output = [1, 0]
        #
        #         training_data.append([data[0], output])
        #         env.reset()
        #         scores.append(score)

    # training_data_save = np.array(training_data)
    # np.save('saved.npy', training_data_save)
    #
    # print('Average accepted score:', mean(accepted_scores))
    # print('Median score for accepted scores:', median(accepted_scores))
    # print(Counter(accepted_scores))
    #
    # return training_data

initial_data()
