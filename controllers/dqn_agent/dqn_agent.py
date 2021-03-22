#!/usr/bin/env python3

import random
from enum import Enum
from collections import namedtuple
import numpy as np
from controller import Robot, Motor, Supervisor

class UR():

    def __init__(self):
        # Initialize training parameters
        self.needReset = False
        self.totalreward = 0
        self.counter = 0
        self.startPose = [0, 0, 0]
        
        self.timeStep = 32
        # Speed
        self.maxSpeed = 6.28 # Max is 6.28
        # Initialize supervisor
        self.supervisor = Robot()
        self.robot = self.supervisor.getFromDef('TB')
        self.reset()
        self.step_iteration = 0
        # Define reward parameters
        
        # Define goal
        self.goal = []
        # Define state space and action space
        self.action_space = []
        
        self.state_space = []

    ####################
    ### MAIN FUNCTIONS

    def reset(self):
        # Reset evetything needs resetting
        self._startEpisode()
        self.supervisor.simulationReset()


    ####################
    ### HELPER FUNCTIONS

    def _startEpisode(self):

        if self.needReset: # Reset object(s) / counters / rewards
            print('Resetting Robot...')
            self._resetObject(self.robot, self.startPose) 
        self.counter = 0
        self.totalreward = 0

    def _setGoal(self):
        # Needs implementation
        return []