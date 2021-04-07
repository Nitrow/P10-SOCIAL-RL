# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import gym
from gym import spaces
import numpy as np
import ikpy
from scipy.spatial import distance
import math
import time
from ikpy.chain import Chain
import random

from controller import Robot, Motor, Supervisor


class P10RLEnv(gym.Env):

    def __init__(self):

        TIME_STEP = 32

        self.supervisor = Supervisor()
        self.robot_node = self.supervisor.getFromDef("UR3")
        selfconveyor_node = self.supervisor.getFromDef("conveyor")
        self.tv_node = self.supervisor.getFromDef("TV")
        self.can_node = self.supervisor.getFromDef("can")
        self.goal = self.supervisor.getFromDef("goal").getField("translation")
        self.collision1 = self.supervisor.getDevice("touch_sensor1")
        self.collision2 = self.supervisor.getDevice("touch_sensor2")
        self.collision3 = self.supervisor.getDevice("touch_sensor3")
        self.collision4 = self.supervisor.getDevice("touch_sensor4")
        self.collision5 = self.supervisor.getDevice("touch_sensor5")
 
        self.collision1.enable(TIME_STEP)
        self.collision2.enable(TIME_STEP)
        self.collision3.enable(TIME_STEP)
        self.collision4.enable(TIME_STEP)
        self.collision5.enable(TIME_STEP)

        self.box = self.supervisor.getFromDef('UR_END')

        trans_field = self.robot_node.getField("translation")

        self.joint_names = ['shoulder_pan_joint',
                       'shoulder_lift_joint',
                       'elbow_joint',
                       'wrist_1_joint',
                       'wrist_2_joint',
                       'wrist_3_joint']
        
        
        self.motors = [0] * len(self.joint_names)
        self.sensors = [0] * len(self.joint_names)

        for i in range(len(self.joint_names)):
            # Get motors
            self.motors[i] = self.supervisor.getDevice(self.joint_names[i])
            self.motors[i].setPosition(float('inf'))
            self.motors[i].setVelocity(0)

            # Get sensors and enable them
            self.sensors[i] = self.supervisor.getDevice(self.joint_names[i] + '_sensor')
            self.sensors[i].enable(TIME_STEP)

        
        self.done = True
        self.success = False
        self.goal.setSFVec3f([random.uniform(-0.3, 0.35), 0.85, 0.59])
        self.target = np.array(self.goal.getSFVec3f())
        self.oldDistance = 0
        self.distance = 0
        self.box_pos_world = self.box.getPosition()
        self.counter = 0
        self.action_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(7,), dtype=np.uint8)

    def reset(self):
        
        
        
        
        
        self.supervisor.simulationReset()
        self.counter = 0
        self.supervisor.step(32)
        self.supervisor.step(32)
        self.supervisor.step(32)

        
        if self.success == True:
                
            self.goal.setSFVec3f([random.uniform(-0.3, 0.35), 0.85, 0.59])
        
            self.target = np.array(self.goal.getSFVec3f())
            
        self.done = False    
        
        
        
        
        
        state = [self.sensors[0].getValue(), self.sensors[1].getValue(), self.sensors[2].getValue(), self.sensors[3].getValue(), self.sensors[4].getValue(), self.distance, self.target[0]]
        
       

        return np.asarray(state)

    def step(self, action):



        #print(self.goal.getPosition(uniform(-0.3, 0.35), 0.85, 0.65 ))


        #target_position = [self.box_pos_world[0] + action[0], self.box_pos_world[1] + action[1], self.box_pos_world[2] + action[2]]
        #target_joints = self.my_chain.inverse_kinematics(target_position)

        #for i in range(len(self.joint_names)-1):
        #            self.motors[i].setPosition(self.sensors[i].getValue()+float(action[i]))
                    #print(self.sensors[i].getValue()+float(action[i]))
        for i in range(len(self.joint_names)-1):
                    self.motors[i].setVelocity(float(action[i]))


        #for i in range(len(self.joint_names)):
        #            self.motors[i].setPosition(target_joints[i+1])
            #self.supervisor.step(8)

        #while abs(self.box_pos_world[0] - self.target_position[0]) > 0.02 and abs(self.box_pos_world[1] - self.target_position[1]) > 0.02 and abs(self.box_pos_world[2] - self.target_position[2]) > 0.02:
        self.supervisor.step(32)



   

           
        rot_ur3e = np.array(self.robot_node.getOrientation())

        rot_ur3e.reshape(3, 3)

        rot_ur3e = np.transpose(rot_ur3e)


        pos_ur3e = np.array(self.robot_node.getPosition())

        self.box_pos_world = np.array(self.box.getPosition())
        
        
        

        #box_pos_world = np.subtract(box_pos_world, pos_ur3e)

        #box_pos_robot = np.dot(rot_ur3e, box_pos_world)t


        
        self.distance = distance.euclidean(self.box_pos_world, self.target)

        state = [self.sensors[0].getValue(), self.sensors[1].getValue(), self.sensors[2].getValue(), self.sensors[3].getValue(), self.sensors[4].getValue(), self.distance, self.target[0]]
        
        
        #print("STATE:", state)

        #self.distance = math.sqrt(pow((target[0] - self.target_position[0]),2) + pow((target[1] - self.target_position[1]),2) + pow((target[2] - self.target_position[2]),2))
        #print("DISTANCE:", self.distance)

        reward = -self.distance



        #make reward for getting closer to can.. d = ((x2 - x1)2 + (y2 - y1)2 + (z2 - z1)2)1/2

        self.counter = self.counter + 1
        



        #if (self.robot_node.getNumberOfContactPoints(True)):
        #print(self.robot_node.getNumberOfContactPoints())
           # print("{} contact points found!".format(contactpoints))
           # for x in range(contactpoints):
           #     print('\t',self.robot_node.getContactPoint(x))

        #self.supervisor.step(16)
        #contactPoints = self.robot_node.getNumberOfContactPoints(True)
        
        #self.supervisor.step(16)

        #print(contactPoints)

        #if contactPoints > 9:
        #    self.done = True
       #     reward = -100
        
        #print(reward)
                    
        if self.counter == 500:
            self.success = False
            self.done = True
        if self.distance < 0.05:
            self.success = True
            self.done = True
            reward = 1000
        if self.collision1.getValue() == 1 or self.collision2.getValue() == 1 or self.collision3.getValue() == 1 or self.collision4.getValue() == 1 or self.collision5.getValue() == 1:
            self.done = True
            self.success = False
            reward = -300


        return [state, float(reward), self.done, {}]

    def render(self, mode='human'):
        pass

    def getState(self):
        pass