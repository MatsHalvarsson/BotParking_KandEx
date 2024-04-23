from tensorflow import abs as tfAbs
from numpy import pi




def alfa_goal(state): #flawed in checking limit on alfa + theta and not theta - alfa
    return state[0] < 0.1 and tfAbs(state[2] + state[1]) < pi / 18

def norm_goal(state):
    return state[0] < 0.05 and tfAbs(state[2] - state[1]) < 0.05

def norm_goal2(state, threshold=0.05):
    return state[0] < threshold and tfAbs(state[2] - state[1]) < threshold

def good_goal(state):
    return state[0] < 0.5 and tfAbs(state[2] - state[1]) < pi / 20

def oob_on_e_distance(state, max_dist=3.0):
    return state[0] > max_dist