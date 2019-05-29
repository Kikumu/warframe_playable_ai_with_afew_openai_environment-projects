import gym
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

score_requirement = 60

initial_games = 10000


def see_environment():
    for episode in range(100):

        env.reset()
        for t in range(goal_steps):
            #env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            print('action', action)
            if done:
                break

see_environment()

# def initial_data():
#     #some_games_first()
#     training_data = []
#     scores = []
#     accepted_scores = []
#     # only store data if score is above 50
#     # actual game
#     for _ in range(initial_games):
#         env.reset()
#         score = 0
#         game_memory = []  # store movements
#         previous_observation = []
#         for _ in range(goal_steps):
#             action = random.randrange(0, 2)  # review this
#             observation, reward, done, info = env.step(action)
#
#             if len(previous_observation) > 0:
#                 game_memory.append([previous_observation, action])
#
#             previous_observation = observation
#             score += reward
#             if done:
#                 break
#
#                     # analysing game
#         if score >= score_requirement:
#             accepted_scores.append(score)
#             for data in game_memory:
#                 if data[1] == 1:
#                     output = [0, 1]
#                 elif data[1] == 0:
#                     output = [1, 0]
#
#                 training_data.append([data[0], output])
#                 env.reset()
#                 scores.append(score)
#
#     training_data_save = np.array(training_data)
#     np.save('saved.npy', training_data_save)
#
#     print('Average accepted score:', mean(accepted_scores))
#     print('Median score for accepted scores:', median(accepted_scores))
#     print(Counter(accepted_scores))
#
#     return training_data
#
# initial_data()
