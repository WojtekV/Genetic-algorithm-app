import math


def fitness_function_1(x, y):
    return (1.5-x+x*y)**2 + (2.25-x+x*y)**2 + (2.625-x+x*y**3)**2


def fitness_function_2(x, y):
    return -math.cos(x)*math.cos(y)*math.exp(-(x-math.pi)**2 - (y-math.pi)**2)