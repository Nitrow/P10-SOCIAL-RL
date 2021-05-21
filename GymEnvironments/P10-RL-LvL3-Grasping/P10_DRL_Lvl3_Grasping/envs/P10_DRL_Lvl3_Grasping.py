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

import os

from controller import Robot, Motor, Supervisor, Connector

class P10_DRL_Lvl3_Grasping(gym.Env):

    def __init__(self):

        random.seed(1)
        self.test = True
        self.id = "SAC - " + str(datetime.now())[:-7].replace(':','_') + 'P10_DRL_Lvl3_Grasping' 
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
        self.tcp_can_dist = []
        self.sensor_fingers[0].enable(self.TIME_STEP)
        self.sensor_fingers[1].enable(self.TIME_STEP)
        self.touch_sensor_f1 = self.supervisor.getDevice("touch sensor finger1")
        self.touch_sensor_f1.enable(self.TIME_STEP)
        self.touch_sensor_f2 = self.supervisor.getDevice("touch sensor finger2")
        self.touch_sensor_f2.enable(self.TIME_STEP)
        self.timeout = 300
        
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

        self.movement_state = 1  # 1 for upper state, 0 for lower state
        self.finger_state = 1  # 0 for open fingers, 1 for closed fingers

        self.epOutcome = ""

        self.reward = 0
        self.rewardstr = "Get presence: 1, close fingers: 1, lift up: 1"
        self.figure_file = self.path + "{} - Rewards {} - Timeout at {}".format(self.id, self.rewardstr, str(self.timeout))
        
        self._setTarget()

        self.up_pose = [16.63, -111.19, -63.15, -96.24, 89.47]
        self.down_pose = [16.62, -119.16, -92.69, -58.74, 89.52]

        self.counter = 0

        self.actionScale = 3
        # Action: open/close finger, rotate joint, go up/down
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(7,), dtype=np.float32)

        self.documentation = "Action space: move_down, move_up, close_fingers, rotate+, rotate-, open_fingers"
        self.documentation += "{} - Rewards {} - Timeout at {}\n".format(self.id, self.rewardstr, str(self.timeout))
        self.saveEpisode(self.documentation)
        print("Robot initilized!")
        
    def reset(self):
        print('\n ------------------------------------ RESET ------------------------------------ \n')
        self.supervisor.simulationReset()
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
        return np.asarray(state)


    def step(self, action):
        # print("Actions: ")
        # print("\t Finger closing: ", action[0])
        # print("\t Moving up or down: ", action[1])
        # print("\t Rotating the fingers: ", action[2])
        #print(self.counter)
        print("Finger1: ",self.touch_sensor_f1.getValue(), "Finger2: ",self.touch_sensor_f2.getValue())
        self.goal_node.resetPhysics()
        self._getMotors()
        self._getSensors()
        # Set actions
        self._action_moveFingers(round(action[0]))  # Open or close fingers
        self._action_moveTCP(round(action[1]))  # Go up or down
        self.motors[-1].setPosition(max(min(self.sensors[-1].getValue() + action[2], 6.28),-6.28))  # Set the rotation clamped
        # Execute actions
        self.supervisor.step(self.TIME_STEP)   
        # Get new state
        state = self.getState()
        self.counter = self.counter + 1
        self._getReward()    
        if self.counter >= self.timeout:
            self.epOutcome = "Timeout"
            print("Timeout")
            self.done = True
        if self.reward >= 4:
            self.epOutcome = "Success"
            print("Success")
            self.done = True
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


    def _getReward(self):
        """
        First reward: Get orientation right
            +1 if connector senses a presence, -1 if it doesn't (one time reward)
        Second reward: Learn to pick it up
            +1 if locks and closes finger
        Third reward: Learn to lift it up
            +1 if lifts up while locked and presence
        """
        #reward = 0
        cp_f1 = self.finger1.getNumberOfContactPoints()  # contact points for finger 1
        cp_f2 = self.finger2.getNumberOfContactPoints()  # contact points for finger 2
        self.isPresence = min(self.robot_connector.getPresence(),1)
        self.isLocked = self.robot_connector.isLocked()
        print(cp_f1, cp_f2)
        #print("--------- R E W A R D S:\t {} ---------".format(self.reward))
        if self.isPresence != self.pastPresence:
            self.reward = self.reward + 1 if self.isPresence else self.reward-1
            #print("\tPresence:{}\tself.reward {}".format(self.isPresence, self.reward))
        if self.reward and self.isLocked != self.pastLocked:
            self.reward = self.reward + 1 if self.isLocked else self.reward - 1
            #print("\tLocked:{}\tself.reward {}".format(self.isLocked, self.reward))
        if self.reward == 2 and self.movement_state != self.pastMoveState:
            self.reward = self.reward + 1 if self.movement_state == 1 else self.reward - 1
            #print("\tMovingUp:{}\tself.reward {}".format(self.movement_state, self.reward))
        if self.reward == 3:
            self.reward = self.reward + 1 if self._util_positionCheck(self.up_pose) else self.reward
            #print("\tself.reward {}".format(self.reward))
        self.pastPresence = self.isPresence
        self.pastLocked = self.isLocked
        self.pastMoveState = self.movement_state


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
        x = random.uniform(-0.05, 0.03)
        translation = [x, 0.84, 0.4]
        self.goal_rot.setSFRotation(rotation)
        self.goal_pos.setSFVec3f(translation)

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
        state = []
        state.append(self.sensors[-1].getValue())  # Get joint angle
        state.append(self._util_axisangle2euler(self.goal_rot.getSFRotation()))  # Get can yaw 
        self.tcp_can_dist  = list(np.array(self.goal_pos.getSFVec3f()) - np.array(self.tcp.getPosition()))
        print("Can relative position: {}\t sum: {}".format(self.tcp_can_dist, abs(sum(self.tcp_can_dist))))
        state += self.tcp_can_dist
        state.append(self.finger_state)
        state.append(self.movement_state)
        # print("State\t",state)
        return state

    def _action_moveFingers(self, mode=0):
        # 0 for open state, 1 for closed state
        if mode == 0:
            self.finger_state = 0
            self.fingers[0].setPosition(0.04)
            self.fingers[1].setPosition(0.04)
            #self.robot_connector.lock()
        elif mode == 1:
            self.finger_state = 1
            self.fingers[0].setPosition(0.015)
            self.fingers[1].setPosition(0.015)
            #self.robot_connector.unlock()


    def _action_moveTCP(self, mode):
        # 1 for upper state, 0 for lower state
        #modeDic = {0 : "down", 1 : "up"}
        #print("Moving {}".format(modeDic[mode]))
        # print(mode)
        if mode == 0:
            self.movement_state = 0
            [self.motors[i].setPosition(m.radians(self.down_pose[i])) for i in range(len(self.down_pose))]
        elif mode == 1:
            self.movement_state = 1
            [self.motors[i].setPosition(m.radians(self.up_pose[i])) for i in range(len(self.up_pose))]


    def _util_positionCheck(self, pos, limit = 0.1):
        if len(pos):
            #print("Target position at: {}, position is at {}".format(pos, [m.degrees(sens[i].getValue()) for i in range(len(pos))]))
            if all([abs(self.sensors[i].getValue() - m.radians(pos[i])) < limit for i in range(len(pos))]):
                movementLock = False
                return True  
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
    
