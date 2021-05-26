# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import gym
from gym import spaces
import numpy as np
import random
import math as m
import matplotlib.pyplot as plt
from datetime import datetime
import torch

import os, sys

from controller import Robot, Motor, Supervisor, Connector

class P10_DRL_SHAP_TEST(gym.Env):

    def __init__(self, itername):

        random.seed(1)

        self.test = False
        
        self.own_path = os.getcwd().split("/P10-XRL/")[0] + "/P10-XRL/GymEnvironments/P10-RL-LvL3-Grasping-Primitives/P10_DRL_Lvl3_Grasping_Primitives/envs/P10_DRL_Lvl3_Grasping_Primitives.py"
        self.id = "DQN - " + str(datetime.now())[:-7].replace(':','_') + 'P10_DRL_Lvl3_Grasping_Primitives_' + itername
        self.path = "data/" + self.id + "/" if not self.test else "test/" + self.id + "/" 

        os.makedirs(self.path, exist_ok=True)

        self.supervisor = Supervisor()
        self.TIME_STEP = int(self.supervisor.getBasicTimeStep())
        self.robot_node = self.supervisor.getFromDef("UR3")
        self.conveyor = self.supervisor.getFromDef("CONVEYOR")
        #self.goal_node = self.supervisor.getFromDef("GREEN_ROTATED_CAN")

        self.can_green_node = self.supervisor.getFromDef("GREEN_ROTATED_CAN_0")
        self.can_yellow_node = self.supervisor.getFromDef("YELLOW_ROTATED_CAN_2")
        self.can_red_node = self.supervisor.getFromDef("RED_ROTATED_CAN_3")
        self.can_blue_node = self.supervisor.getFromDef("BLUE_ROTATED_CAN_1")
        self.cans = [self.can_green_node, self.can_yellow_node, self.can_red_node, self.can_blue_node]
        self.spawns = {"GREEN" : [2.0, 2.2], "YELLOW" : [2.3, 2.5], "BLUE" : [1.7, 1.9], "RED" : [1.4, 1.6]}
        self.rotations = self._util_readRotationFile('rotations.txt')#[0.577, 0.577, 0.577, 2.094]

        self.total_rewards = 0
        self.reward = 0
        
        self.counter = 0
        self.color_code = {"GREEN" : 2, "YELLOW" : 0, "BLUE" : -1, "RED" : 5}
        self.action_code = {0 : "GREEN", 1 : "BLUE", 2 : "YELLOW", 3 : "RED"}
        # Action: open/close finger, rotate joint, go up/down
        self.action_shape = 4
        self.state_shape = len(self.cans)*3
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_shape,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-200, high=200, shape=(self.state_shape,), dtype=np.float32)
        
    def reset(self):
        print('\n ------------------------------------ RESET ------------------------------------ \n')
        self.supervisor.simulationReset()
        self.conveyor.restartController()
        self.supervisor.step(self.TIME_STEP)  
        self.counter = 0
        self.reward = 0
        self.total_rewards = 0    
        self.done = False
        self.setTarget()
        state = self.getState()
        return state


    def step(self, action):
        # Check cans
        self.counter += 1
        state = [0.0] * self.state_shape
        self.reward = 0
        #print(self.rotations)
        # for rotation in self.rotations:
        #     print(rotation)
        #     for can in self.cans:
        #         can.getField("rotation").setSFRotation(rotation)
        #     [self.supervisor.step(self.TIME_STEP) for i in range(100)]
        for can in self.cans:
            number = int(can.getDef().split("_")[-1])
            color = can.getDef().split("_")[0]
            i = number*3
            translation = can.getField("translation").getSFVec3f()
            if translation[0] < 0:
                rotation = random.choice(self.rotations)
                translation[0] = translation[0]+2
                can.getField("translation").setSFVec3f(translation)
                can.getField("rotation").setSFRotation(rotation)
            #rotation = [round(x,3) for x in can.getField("rotation").getSFRotation()]
            rotation = self.axisangle2euler(can.getField("rotation").getSFRotation())
            if action == number:
                self.reward = self.reward + 1 if rotation[2] != 180 else self.reward - 4
                self.reward += self.color_code[color]
                #self.reward = self.reward + 1 if rotation[2] != 180 else self.reward - 2
            state[i] = float(rotation[2])
            state[i+1] = float(max(0.01/translation[0],2))
            state[i+2] = float(self.color_code[color])

        if self.counter >= 2000:
            self.done = True
        self.supervisor.step(self.TIME_STEP)
        print(self.action_code[action], self.reward)
        self.total_rewards += self.reward
        return [state, float(self.reward), self.done, {}]



    def setTarget(self):
        for can in self.cans:
            ranges = self.spawns[can.getDef().split("_")[0]]
            rotation = random.choice(self.rotations)
            z = random.uniform(0.35, 0.45)
            x = random.uniform(*ranges)
            #translation = [-0.01, 0.84, 0.4]
            translation = [x, 0.86, z]
            
            can.getField("rotation").setSFRotation(rotation)
            can.getField("translation").setSFVec3f(translation)
            self.supervisor.step(self.TIME_STEP)
            can.resetPhysics()

    def render(self, mode='human'):
        pass



    def getState(self):
        """
        1st place Rotation
        2nd place Distance
        3rd place Color
        """
        state = [0.0] * self.state_shape
        return state

    
    def _util_readRotationFile(self, file):
        rotationFile = open(file, 'r')
        rotationFileLines = rotationFile.readlines()
        rotations = []
        # Strips the newline character
        for line in rotationFileLines:
            l = line.strip()
            numbers = l[1:-1].split(', ')
            tempList = [float(num) for num in numbers]
            rotations.append(tempList) 
        return rotations

    def axisangle2euler(self, rotation):
        # YZX
        x,y,z,angle = rotation
        s = m.sin(angle)
        c = m.cos(angle)
        t = 1-c
        if ((x*y*t + z*s) > 0.998): # north pole singularity
            yaw = round(m.degrees(2*m.atan2(x*m.sin(angle/2), m.cos(angle/2))))
            pitch = round(m.degrees(m.pi/2))
            roll = round(m.degrees(0))
            #return [roll, pitch, yaw]

        elif ((x*y*t + z*s) < -0.998):
            yaw = round(m.degrees(-2*m.atan2(x*m.sin(angle/2), m.cos(angle/2))))
            pitch = round(m.degrees(-m.pi/2))
            roll = round(m.degrees(0))
            #return [roll, pitch, yaw]
        else:
            yaw = round(m.degrees(m.atan2(y*s - x*z, 1 - (y*y + z*z) * t)))
            pitch = round(m.degrees(m.asin(x * y * t + z * s)))
            roll = round(m.degrees(m.atan2(x * s - y * z * t, 1 - (x*x + z*z) * t)))
        yaw = yaw + 180 if yaw <= 0 else yaw
        pitch = pitch + 180 if pitch <= 0 else pitch
        roll = roll + 180 if roll <= 0 else roll
        return [roll, pitch, yaw]