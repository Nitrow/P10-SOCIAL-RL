#!/usr/bin/env python3
import numpy as np
from controller import Robot, Supervisor, Lidar

class Environment():

    def __init__(self):
        """
        Initilializing the environment
        """
        self.supervisor = Supervisor()
        self.robot = self.supervisor.getFromDef('UR3')
        self.episode_over = False
        self.step_iteration = 0
        self.joint_names = [ 'shoulder_pan_joint',
                            'shoulder_lift_joint',
                            'elbow_joint',
                            'wrist_1_joint',
                            'wrist_2_joint',
                            'wrist_3_joint']
        self.motors = [0] * len(self.joint_names)
        self.sensors = [0] * len(self.joint_names)
        
        self._getmotors()
                            
        
    def reset(self):
        """
        Resetting the environment
        """
        pass
        
    def play_step(self, action):
        """
        Playing the action
        """
        self.step_iteration += 1
        self.move(action)
        self.supervisor.step(1)
        self.calculate_reward()
        return reward, game_over
        
        
    def move(self, action):
        """
        Setting the motors
        """
        pass
        
    def calculate_reward(self):
        """
        Calculates the reward
        """
        pass
    
    def _getmotors(self):
        """
        Initializes the motors and their sensors
        """    
