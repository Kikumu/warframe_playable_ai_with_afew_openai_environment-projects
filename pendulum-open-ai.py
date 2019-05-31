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
    for _ in range(1000):
        env.reset()
        game_memory = []  # store movements
        previous_observation = []
        for _ in range(goal_steps):
            force_applied = env.action_space.sample()  # review this
            # print('action', env.action_space.sample())
            observation, reward, done, info = env.step(force_applied)
            # print('observation', observation)

            #analysing game
            if (observation[0]) > ((np.pi/180)*50):
                # env.render()
                force_applied = env.action_space.sample()
                training_data.append([observation, force_applied])
                # print(training_data)
            score += reward #return value for episodic task
            # print('score', score)
            if done:
                break


initial_data()

def neural_network_model(input_size):

    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

def train_model(training_data, model=False):
    X = np.array([i[0] for i in training_data]).reshape(-2, len(training_data[0][0]), 2)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size=len(X[0]))

    model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openai_learning')
    return model


training_data = initial_data()
model = train_model(training_data)
