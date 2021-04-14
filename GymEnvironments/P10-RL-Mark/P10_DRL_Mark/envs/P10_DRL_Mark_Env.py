# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import gym
from gym import spaces
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from datetime import datetime
#import ikpy
#from ikpy.chain import Chain





from controller import Robot, Motor, Supervisor


class P10_DRL_Mark_Env(gym.Env):

    def __init__(self):

        self.TIME_STEP = 32
        self.id = 'SAC_P10_MarkEnv_' + str(datetime.now())[:-7].replace(':','_')
        self.supervisor = Supervisor()
        self.robot_node = self.supervisor.getFromDef("UR3")
        selfconveyor_node = self.supervisor.getFromDef("conveyor")
        self.tv_node = self.supervisor.getFromDef("TV")
        self.can_node = self.supervisor.getFromDef("can")
        self.goal_node = self.supervisor.getFromDef("TARGET").getField("translation")
 
        self.tcp = self.supervisor.getFromDef('TCP')

        self.timeout = 10000
        
        self.joint_names = ['shoulder_pan_joint',
                       'shoulder_lift_joint',
                       'elbow_joint',
                       'wrist_1_joint',
                       'wrist_2_joint',
                       'wrist_3_joint']
        
        self.motors = [0] * len(self.joint_names)
        self.sensors = [0] * len(self.joint_names)
        self.touch_sensors = [0] * (len(self.joint_names)+1)

        self._getSensors()
        self._getMotors()
        
        self.done = True
        self.prev_dist = 0
        
        self.plot_rewards = []
        self.total_rewards = 0
        self.collisionReward = -300 
        self.distanceReward = -0.1
        self.distanceDeltaReward = 100
        self.successReward = 1000
        self.rewardstr = "success {}, collision {} distance {}".format(self.successReward, self.collisionReward, self.distanceDeltaReward)
        self.figure_file="plots/{} - Rewards {} - Timeout at {}".format(self.id, self.rewardstr, str(self.timeout))
        
        #print(self.goal_node.getSFVec3f())
        self._setTarget()
        self.oldDistance = 0
        self.distance = 0
        self.tcp_pos_world = self.tcp.getPosition()
        self.counter = 0
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(7,), dtype=np.uint8)


    def reset(self):
        print('\n ------------------------------------ RESET ------------------------------------ \n')
        self.supervisor.simulationReset()
        self.counter = 0

        self.supervisor.step(self.TIME_STEP)
        self.goal_node.setSFVec3f(self.target)
        self.total_rewards = 0    
        self.done = False    
        state = self.getState()
        self.prevdist = np.linalg.norm(np.array(self.tcp.getPosition()) - self.target)
        return np.asarray(state)


    def step(self, action):

        for i in range(len(self.joint_names)):
                    self.motors[i].setVelocity(float(action[i]))
        
        self.supervisor.step(self.TIME_STEP)
        self.supervisor.step(self.TIME_STEP)
        
        rot_ur3e = np.transpose(np.array(self.robot_node.getOrientation()).reshape(3, 3))
        pos_ur3e = np.array(self.robot_node.getPosition())
        
        self.distance = np.linalg.norm(np.array(self.tcp.getPosition()) - self.target)
        distDifference = self.prevdist - self.distance
        self.prevdist = self.distance
        #print("Distance: {}".format(np.linalg.norm(np.array(self.tcp.getPosition()) - pos_ur3e)))
        state = self.getState()
        # Normalize the distance by the maximum robot reach so it's between 0 and 1
        reward = self.distanceDeltaReward * distDifference + (self.distanceReward * (self.distance / 1.45637))

        self.counter = self.counter + 1
              
        if self.counter >= self.timeout:
            print("Timeout")
            self.done = True
        if self.distance < 0.05:
            print("Success")
            self._setTarget()
            reward += self.successReward
        if self._isCollision():
            print("Collision")
            self.done = True
            reward += self.collisionReward
        #print(self.done)
        self.total_rewards += reward
        if self.done: self.plot_learning_curve()
        #print(reward)
        return [state, float(reward), self.done, {}]


    def _isCollision(self):
        """
        Returns True if any of the touch sensors are activated
        """
        return any([self.touch_sensors[i].getValue() for i in range(len(self.touch_sensors))])


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
            self.motors[i].setPosition(float('inf'))
            # Get sensors and enable them
            self.sensors[i] = self.supervisor.getDevice(self.joint_names[i]+'_sensor')
            self.sensors[i].enable(self.TIME_STEP)

    def _setTarget(self):
        self.target = [random.uniform(-0.3, 0.35), 0.85, random.uniform(0.4, 0.5)]
        self.goal_node.setSFVec3f(self.target)

    def render(self, mode='human'):
        pass


    def getState(self):
        return [self.sensors[0].getValue(), self.sensors[1].getValue(), self.sensors[2].getValue(), self.sensors[3].getValue(), self.sensors[4].getValue(), self.distance, self.target[0]]


    def plot_learning_curve(self):
        self.plot_rewards.append(self.total_rewards)
        x = [i+1 for i in range(len(self.plot_rewards))]
        running_avg = np.zeros(len(self.plot_rewards))
        for i in range(len(running_avg)):
            running_avg[i] = np.mean(self.plot_rewards[max(0, i-100):(i+1)])
        plt.plot(x, running_avg)
        plt.title('Running average of previous 100 scores')
        plt.savefig(self.figure_file)