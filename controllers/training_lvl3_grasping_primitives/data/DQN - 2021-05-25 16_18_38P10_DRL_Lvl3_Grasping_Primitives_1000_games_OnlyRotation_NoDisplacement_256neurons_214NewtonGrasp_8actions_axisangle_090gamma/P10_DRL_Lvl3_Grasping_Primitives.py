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

class P10_DRL_Lvl3_Grasping_Primitives(gym.Env):

    def __init__(self, itername):

        random.seed(1)
        self.test = False
        #self.own_path = "/home/harumanager/P10-XRL/GymEnvironments/P10-RL-LvL3-Grasping-Primitives/P10_DRL_Lvl3_Grasping_Primitives/envs/P10_DRL_Lvl3_Grasping_Primitives.py"
        self.own_path = os.getcwd().split("/P10-XRL/")[0] + "/P10-XRL/GymEnvironments/P10-RL-LvL3-Grasping-Primitives/P10_DRL_Lvl3_Grasping_Primitives/envs/P10_DRL_Lvl3_Grasping_Primitives.py"
        self.id = "DQN - " + str(datetime.now())[:-7].replace(':','_') + 'P10_DRL_Lvl3_Grasping_Primitives_' + itername
        #self.id = '2021-04-15 09_44_43_SAC_P10_MarkEnv_SingleJoint_' 
        self.path = "data/" + self.id + "/" if not self.test else "test/" + self.id + "/" 
        os.makedirs(self.path, exist_ok=True)
        self.supervisor = Supervisor()
        self.TIME_STEP = int(self.supervisor.getBasicTimeStep())
        self.robot_node = self.supervisor.getFromDef("UR3")
        self.robot_connector = self.supervisor.getDevice("connector")
        self.robot_connector.enablePresence(self.TIME_STEP)
        self.finger1 = self.supervisor.getFromDef("FINGER1")
        self.finger2 = self.supervisor.getFromDef("FINGER2")

        self.isPresence = 0
        self.pastPresence = 0
        self.pastLocked = False
        self.isLocked = False
        self.pastMoveState = 1
        self.goal_pos = self.supervisor.getFromDef("GREEN_ROTATED_CAN").getField("translation")
        self.goal_rot = self.supervisor.getFromDef("GREEN_ROTATED_CAN").getField("rotation")
        self.goal_node = self.supervisor.getFromDef("GREEN_ROTATED_CAN")
        self.fingers = [self.supervisor.getDevice('right_finger'), self.supervisor.getDevice('left_finger')]
        self.sensor_fingers = [self.supervisor.getDevice('right_finger_sensor'), self.supervisor.getDevice('left_finger_sensor')]
        self.tcp = self.supervisor.getFromDef("TCP")
        self.tcp_pos = self.supervisor.getFromDef("TCP").getField("translation")
        self.tcp_can_total_dist = 0
        self.tcp_can_vertical_dist = 0
        self.tcp_can_horizontal_dist = 0
        self.sensor_fingers[0].enable(self.TIME_STEP)
        self.sensor_fingers[1].enable(self.TIME_STEP)
        # self.touch_sensor_f1 = self.supervisor.getDevice("touch sensor finger1")
        # self.touch_sensor_f1.enable(self.TIME_STEP)
        # self.touch_sensor_f2 = self.supervisor.getDevice("touch sensor finger2")
        # self.touch_sensor_f2.enable(self.TIME_STEP)
        self.timeout = 100
        
        self.rotations = self._util_readRotationFile('rotations.txt')#[0.577, 0.577, 0.577, 2.094]

        self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        #self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        
        self.motors = [0] * len(self.joint_names)
        self.sensors = [0] * len(self.joint_names)
        
        self.touch_sensors = [0] * 7#(len(self.joint_names)+1)
     
        self.done = True
        self.prev_dist = 0
        
        self.plot_rewards = []
        self.total_rewards = 0
        # Different reward components:
        self.reward = 0
        self.partialReward = 0

        self.movement_state = 1  # 1 for upper state, 0 for lower state
        self.finger_state = 1  # 0 for open fingers, 1 for closed fingers
        self.pastFingerState = self.finger_state

        self.rewardstr = "Get presence: 1, close fingers: 1, lift up: 1"
        self.figure_file = self.path + "{} - Rewards {} - Timeout at {}".format(self.id, self.rewardstr, str(self.timeout))
        
        self._setTarget()

        self.up_pose = [16.63, -111.19, -63.15, -96.24, 89.47]
        self.down_pose = [16.60, -121.69, -94.63, -54.27, 89.52]
        
        self.counter = 0

        self.actionScale = 3
        # Action: open/close finger, rotate joint, go up/down
        self.action_shape = 8
        self.actions = [m.radians(22.5*i) for i in range(self.action_shape)]
        self.state_shape = 7
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_shape,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-200, high=200, shape=(self.state_shape,), dtype=np.float32)

        self.documentation = "Action space: move_down, move_up, close_fingers, rotate+, rotate-, open_fingers"
        self.documentation += "{} - Rewards {} - Timeout at {}\n".format(self.id, self.rewardstr, str(self.timeout))
        self.saveEpisode(self.documentation)
        print("Robot initilized!")
        
    def reset(self):
        print('\n ------------------------------------ RESET ------------------------------------ \n')
        self.supervisor.simulationReset()
        self.supervisor.step(self.TIME_STEP)  
        self.counter = 0
        self.reward = 0
        self._getMotors()
        self._getSensors() 
        self._setTarget()
        self.finger1.resetPhysics()
        self.finger2.resetPhysics()
        self.total_rewards = 0    
        self.done = False
        state = self.getState()
        return state


    def step(self, action):
        if self.test:
            print("--------------------")
            print("STEP: ", self.counter)
        self.partialReward = 0#-0.25
        #self.goal_node.resetPhysics()
        self._getMotors()
        self._getSensors()
        self.finger1.resetPhysics()
        self.finger2.resetPhysics()
        # Set actions
        self.motors[-1].setPosition(float(self.actions[action]))
        while not (self.sensors[-1].getValue() - action) < 0.01: self.supervisor.step(self.TIME_STEP)
        #if not self.isPresence:
        self._action_moveFingers(0)  # Open fingers
        for i in range(5): self.supervisor.step(self.TIME_STEP)
        self._action_moveTCP(0)  # Go down
        for i in range(5): self.supervisor.step(self.TIME_STEP)
        self._action_moveFingers(1)  # close fingers
        for i in range(5): self.supervisor.step(self.TIME_STEP)
        self._action_moveTCP(1)  # Go up
        for i in range(100):
            if self.partialReward and np.linalg.norm(np.array(self.goal_pos.getSFVec3f())-np.array(self.tcp.getPosition()))*100 < 10:
                self.partialReward = 0
            self.supervisor.step(self.TIME_STEP)
        #print(self.finger1.getNumberOfContactPoints(), self.finger2.getNumberOfContactPoints())
        self.counter = self.counter + 1
        self.reward = 1 if np.linalg.norm(np.array(self.goal_pos.getSFVec3f())-np.array(self.tcp.getPosition()))*100 < 10 else self.partialReward
        if self.test: print("REWARD: ", self.reward)
        if self.reward == 1:
            if self.test: print("Success")
        if self.counter >= self.timeout:
            self.done = True
        self._setTarget()
        state = self.getState()
        if self.test: print("STATE: ", state)
        self.total_rewards += self.reward
        if self.done:
            self.saveEpisode(str(round(self.total_rewards)) + ";")
            self.plot_learning_curve()
        return [state, float(self.reward), self.done, {}]


    def _getSensors(self):
        """
        Initializes the touch sensors
        """
        for i in range(len(self.touch_sensors)):
            self.touch_sensors[i] = self.supervisor.getDevice("touch_sensor"+str(i+1))
            self.touch_sensors[i].enable(self.TIME_STEP)


    def _getMotors(self):
        """
        Initializes the motors and their sensors
        """
        for i in range(len(self.joint_names)):
            # Get motors
            self.motors[i] = self.supervisor.getDevice(self.joint_names[i])
            #self.motors[i].setPosition(float('inf'))
            #self.motors[i].setVelocity(0)
            # Get sensors and enable them
            self.sensors[i] = self.supervisor.getDevice(self.joint_names[i]+'_sensor')
            self.sensors[i].enable(self.TIME_STEP)


    def _setTarget(self):
        rotation = random.choice(self.rotations)
        #x = random.uniform(-0.05, 0.03)
        translation = [-0.01, 0.84, 0.4]
        self.goal_rot.setSFRotation(rotation)
        self.goal_pos.setSFVec3f(translation)
        self.supervisor.step(self.TIME_STEP)
        self.goal_node.resetPhysics()

    def render(self, mode='human'):
        pass


    def saveEpisode(self, reward):
        with open(os.path.join(self.path, "documentation.txt"), 'a') as f:
            f.write(reward)


    def getState(self):
        """
        1. Joint angle - self.sensors[-1] (1)
        2. can orientation - absolute (change to relative to tool?) (4)
        3. can position - absolute (change to relative to tool?) (3)
        4. finger status (open, closed) - binary
        5. move status (up, down) - binary
        """
        #self.tcp_can_vertical_dist = abs(self.goal_pos.getSFVec3f()[1] - self.tcp.getPosition()[1])*100
        #self.tcp_can_horizontal_dist = abs(np.linalg.norm(np.array(self.goal_pos.getSFVec3f()[::2])-np.array(self.tcp.getPosition()[::2]))*100)
        x_dist = abs(self.goal_pos.getSFVec3f()[0] - self.tcp.getPosition()[0])*100
        y_dist = abs(self.goal_pos.getSFVec3f()[2] - self.tcp.getPosition()[2])*100
        # print(self.tcp_can_horizontal_dist)
        state = []
        state.append(self.sensors[-1].getValue())  # Get joint angle
        #state.append(self._util_axisangle2euler(self.goal_rot.getSFRotation()))  # Get can yaw 
        state += self.goal_rot.getSFRotation()
        #self.tcp_can_total_dist  = np.linalg.norm(np.array(self.goal_pos.getSFVec3f())-np.array(self.tcp.getPosition()))*100#abs(sum(list(np.array(self.goal_pos.getSFVec3f()) - np.array(self.tcp.getPosition()))))
        #self.candistReward = -self.tcp_can_total_dist/100
        #print("Can total dist {}\t vertical dist: {}".format(self.tcp_can_total_dist, self.tcp_can_vertical_dist))
        #state.append(self.tcp_can_total_dist)
        #state.append(self.tcp_can_vertical_dist)
        #state.append(self.tcp_can_horizontal_dist)
        # print("State\t",state)
        state.append(x_dist)
        state.append(y_dist)
        state = [float(s) for s in state]
        return state

    def _action_moveFingers(self, mode=0):
        # 0 for open state, 1 for closed state
        if mode == 0:
            self.finger_state = 0
            pos = [0.04, 0.04]
        elif mode == 1:
            self.finger_state = 1
            pos = [0.015, 0.015]
        self.fingers[0].setPosition(pos[0])
        self.fingers[1].setPosition(pos[1])
        self._util_positionCheck(pos, self.sensor_fingers, 0.02)

    def _action_moveTCP(self, mode):
        # 1 for upper state, 0 for lower state
        #modeDic = {0 : "down", 1 : "up"}
        #print("Moving {}".format(modeDic[mode]))
        # print(mode)
        pose = self.down_pose
        if mode == 0:
            pose = self.down_pose
            self.movement_state = 0
        elif mode == 1:
            self.movement_state = 1
            pose = self.up_pose
        [self.motors[i].setPosition(m.radians(pose[i])) for i in range(len(pose))]
        self._util_positionCheck(pose, self.sensors, 0.05)

    def _util_positionCheck(self, pos, sensors, limit = 0.1):
        if len(pos):
            counter = 0
            prev_pose = 0
            pose = 0
            #print("Target position at: {}, position is at {}".format(pos, [m.degrees(sens[i].getValue()) for i in range(len(pos))]))
            while not all([abs(sensors[i].getValue() - m.radians(pos[i])) < limit for i in range(len(pos))]):
                pose = round(sum([abs(sensors[i].getValue() - m.radians(pos[i])) for i in range(len(pos))]),2)
                counter = counter + 1 if pose == prev_pose else 0
                if counter >= 1: break
                self.supervisor.step(self.TIME_STEP)
                prev_pose = pose
        return False


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

    def _util_axisangle2euler(self, rotation):
        # YZX
        x,y,z,angle = rotation
        s = m.sin(angle)
        c = m.cos(angle)
        t = 1-c
        if ((x*y*t + z*s) > 0.998): # north pole singularity
            yaw = round(m.degrees(2*m.atan2(x*m.sin(angle/2), m.cos(angle/2))))

        elif ((x*y*t + z*s) < -0.998):
            yaw = round(m.degrees(-2*m.atan2(x*m.sin(angle/2), m.cos(angle/2))))

        else:
            yaw = round(m.degrees(m.atan2(y*s - x*z, 1 - (y*y + z*z) * t)))
        yaw = yaw + 180 if yaw <= 0 else yaw
        return yaw


    def plot_learning_curve(self):
        self.plot_rewards.append(self.total_rewards)
        x = [i+1 for i in range(len(self.plot_rewards))]
        running_avg = np.zeros(len(self.plot_rewards))
        for i in range(len(running_avg)):
            running_avg[i] = np.mean(self.plot_rewards[max(0, i-100):(i+1)])
        plt.plot(x, running_avg)
        plt.title('Running average of previous 100 scores')
        plt.savefig(self.figure_file)


    #     if len(scores)<101:
    #     return
    # else:
    #     running_avg = np.zeros(len(scores)-avg)
    #     for i in range(len(running_avg)):
    #         running_avg[i] = np.mean(scores[i-100:(i+1)])
    #     plt.plot(x, running_avg)
    #     plt.title('Running average of previous 100 scores')
    #     plt.savefig(figure_file)
    
