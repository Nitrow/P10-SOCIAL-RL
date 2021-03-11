import random
from enum import Enum
from collections import namedtuple
import numpy as np


# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 20

REWARDS = {"collision": -10, "eat":10, "else":0}


class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.reset()


    def reset(self):
        # init game state

        self.score = 0
        self.frame_iteration = 0


    def play_step(self, action):
        reward = action
        game_over = False
        self.score = 10
        return reward, game_over, self.score

    def _move(self, action):
        # [straight, right turn, left turn]
        pass

