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

goal_steps = 100

# required_angle_range = range(np.pi/9, np.pi/8)
score_requirement = 50

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


def initial_data():
    #some_games_first()
    training_data = []
    scores = []
    accepted_scores = []
    discount_rate  = 0 #for non episodic tasks
    #what are my policies?
    policy_angle = np.pi/8
    #observation is the state? yes
    #but, observation has 3 values,cos theta, sin theta and theta dot
    #action is effort applied on the stick
    #zero deals with y axis and 1 deals with x axis
    score = 0 #episodic task
    for _ in range(10000):
        env.reset()
        game_memory = []  # store movements
        previous_observation = []
        for _ in range(goal_steps):
            force_applied = env.action_space.sample()  # review this
            observation, reward, done, info = env.step(force_applied)
            if len(previous_observation) > 0:
                game_memory.append([previous_observation, force_applied])
            score += reward
            previous_observation = observation
            if done:
                break
            #analysing game
        if (observation[0]) > ((np.pi/180)*50) :
             for data in game_memory:
                 print('data', data[1])

             # print('trainingdata', training_data)

        # if(score > score_requirement):
        #     for data in game_memory:
        #         env.render()
        #         if data[1] <= ((np.pi/180)*50):
        #             output = data[1]
        #         elif data[0] <= ((np.pi/180)*120):
        #             output = data[0]

                # training_data.append([data[0]], data[1], )



initial_data()

# def neural_network_model(input_size):
#
#     network = input_data(shape=[None, input_size, 1], name='input')
#
#     network = fully_connected(network, 128, activation='relu')
#     network = dropout(network, 0.8)
#
#     network = fully_connected(network, 256, activation='relu')
#     network = dropout(network, 0.8)
#
#     network = fully_connected(network, 512, activation='relu')
#     network = dropout(network, 0.8)
#
#     network = fully_connected(network, 256, activation='relu')
#     network = dropout(network, 0.8)
#
#     network = fully_connected(network, 128, activation='relu')
#     network = dropout(network, 0.8)
#
#     network = fully_connected(network, 2, activation='softmax')
#     network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
#     model = tflearn.DNN(network, tensorboard_dir='log')
#
#     return model
#
# def train_model(training_data, model=False):
#     X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
#     y = [i[1] for i in training_data]
#
#     if not model:
#         model = neural_network_model(input_size=len(X[0]))
#
#     model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openai_learning')
#     return model
#
#
# training_data = initial_data()
# model = train_model(training_data)
