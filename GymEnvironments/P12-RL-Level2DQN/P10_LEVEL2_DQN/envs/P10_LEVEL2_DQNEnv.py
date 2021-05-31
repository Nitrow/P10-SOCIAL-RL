# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import gym
from gym import spaces
import numpy as np
import random
import math
import math as m
import matplotlib.pyplot as plt
from datetime import datetime
import torch
from sac_torch import Agent
from networks import ActorNetwork, CriticNetwork, ValueNetwork
from P10_DRL_Mark_SingleJoint.envs import P10_DRL_Mark_SingleJointEnv
import torch as T
from scipy.spatial import distance
from dqn_agent import DeepQNetwork


import os
from scipy.spatial import distance

#import ikpy
#from ikpy.chain import Chain


from controller import Robot, Motor, Supervisor


class P10_LEVEL2_DQNEnv(gym.Env):

    def __init__(self):
        
        
        self.chkpt_dir='models/'
        
        
        ##Move2 policy setup##
        
        self.alpha = 0.0003
        self.beta = 0.0003
        self.input_dims = (26,)
        self.input_dims2 = (17,)
        self.layer1_size = 256
        self.layer2_size = 256
        self.n_actions = 6
        self.high = 10
        
        ##grasping policy setup##
        
        self.lr = 0.003
        self.n_actionsGrasp = 8
        self.input_dimsGrasp = (7,)

        
        


        self.actorMove2Goal = ActorNetwork(self.alpha, self.input_dims, self.layer1_size,
                                  self.layer2_size, n_actions=self.n_actions,
                                  name='Move2Goal_actor', 
                                  max_action=self.high,
                                  chkpt_dir=self.chkpt_dir)
                                  
        self.actorMove2Bin = ActorNetwork(self.alpha, self.input_dims2, self.layer1_size,
                                  self.layer2_size, n_actions=self.n_actions,
                                  name='Move2Bin_actor', 
                                  max_action=self.high,
                                  chkpt_dir=self.chkpt_dir)

        self.actorGrasp = DeepQNetwork(self.lr, self.n_actionsGrasp,
                                    input_dims=self.input_dimsGrasp,
                                    name='Grasping_actor',
                                    chkpt_dir=self.chkpt_dir)

        self.actorMove2Goal.load_checkpoint()
        self.actorMove2Bin.load_checkpoint()
        self.actorGrasp.load_checkpoint()




        
        
        random.seed(1)
       
        self.supervisor = Supervisor()
        self.TIME_STEP = int(self.supervisor.getBasicTimeStep())
        self.robot_node = self.supervisor.getFromDef("UR3")
        self.tcp = self.supervisor.getFromDef('TCP')
        self.can = self.supervisor.getFromDef('SelectedCan')
        self.goal_node = self.supervisor.getFromDef("TARGET").getField("translation")

        self.tcpy = self.supervisor.getFromDef("y")
        self.tcpy2 = self.supervisor.getFromDef("y2")
        
        
        self.goaly = self.supervisor.getFromDef("goaly")
        self.goaly2 = self.supervisor.getFromDef("goaly2") 

        self.timeout = 1500

        self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']


        self.motors = [0] * len(self.joint_names)
        self.sensors = [0] * len(self.joint_names)
        
        self.touch_sensors = [0] * 7

        self.robot_pos = np.array(self.robot_node.getPosition())       
   
        self.conveyor = self.supervisor.getFromDef('CONVEYOR')
       
        self.done = True
        
        self.goal_node.setSFVec3f([0,0.425,0.15])
        

        self.oldDistance = 0
        self.distancex = 0
        self.distancey = 0
        self.distancez = 0
        self.tcp_pos_world = self.tcp.getPosition()
        self.counter = 0
         
        self._getMotors()
        self._getSensors()

        
          
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(4,), dtype=np.float32)
        
        
    def reset(self):
        print('\n ------------------------------------ RESET ------------------------------------ \n')
        self.supervisor.simulationReset()
        
        self.conveyor.restartController()
        self.counter = 0

        self.goal_node.setSFVec3f([0,0.425,0.15])
        
        self.target = [0.55, 0.4, 1]
        
        
        self.supervisor.step(self.TIME_STEP) 
 
        self.done = False    

        state = [0]*4
        return state


    def step(self, action):
        
        self.counter = self.counter + 1
        
        
        if action == 0:
            

            while distance.euclidean(self.tcpy.getPosition(), self.goaly.getPosition()) > 0.02 and distance.euclidean(self.tcpy2.getPosition(), self.goaly2.getPosition()) > 0.02:                 
                
                self._getMotors()
                self._getSensors()
                
                observation = self.getObservationMove2Goal()
                
                state = T.Tensor([[observation]]).to(self.actorMove2Goal.device)
                actions, _ = self.actorMove2Goal.sample_normal(state, reparameterize=False)
                action = actions.cpu().detach().numpy()[0][0]
                
                for i in range(len(self.joint_names)):
                    self.motors[i].setVelocity(float(np.array(action)[i]))
            
                self.supervisor.step(self.TIME_STEP)
    
            for i in range(len(self.joint_names)):
                    self.motors[i].setVelocity(float(0))  
            
        elif action == 1:
            
            
            observation = self.getState()
            
            state = T.tensor([observation]).to(self.actorGrasp.device)
            actions = self.actorGrasp.forward(state)
            action = T.argmax(actions).item() 
            
            
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
                
            
        elif action == 2:
                
            while distance.euclidean(self.tcpy.getPosition(), self.target) < 0.1: 
                               
                self._getMotors()
                self._getSensors()
                
                observation = self.getObservationMove2Bin()
                
                state = T.Tensor([[observation]]).to(self.actorMove2Goal.device)
                actions, _ = self.actorMove2Goal.sample_normal(state, reparameterize=False)
                action = actions.cpu().detach().numpy()[0][0]                

                for i in range(len(self.joint_names)):
                    self.motors[i].setVelocity(float(np.array(action)[i]))
           
                self.supervisor.step(self.TIME_STEP)
    
            for i in range(len(self.joint_names)):
                    self.motors[i].setVelocity(float(0))  
        elif action == 3:
            print("")
               
        self.supervisor.step(self.TIME_STEP)   
        
        
           
        state = self.getState()
        print (state)    
         
        reward = 1   
        #print(reward)
               
        if self.counter == 2000:
            self.done = True
        #print(state)
        
        counter =+ 1
        
        return [state, float(reward), self.done, {}]


    def render(self, mode='human'):
        pass


    def getState(self):
        
        
        distance = abs(self.tcp.getPosition()[0] - self.can.getField("translation").getSFVec3f()[0])
        
        if self.can.getField("translation").getSFVec3f()[1] > 0.92:
            grasp = True
        else:
            grasp = False
        
        state = [self.tcp.getPosition()[0], self.tcp.getPosition()[2],  distance, grasp]

        return state 
        #return self.tcp.getPosition() + self.target 
    
    def getObservationMove2Goal(self):
    
    
        self.distancey = distance.euclidean(self.tcpy.getPosition(), self.goaly.getPosition())
        self.distancey2 = distance.euclidean(self.tcpy2.getPosition(), self.goaly2.getPosition())
        return [self.sensors[i].getValue() for i in range(len(self.sensors))] + [self.motors[i].getVelocity() for i in range(len(self.motors))]  + self.tcpy.getPosition() + self.tcpy2.getPosition() + self.goaly.getPosition() + self.goaly2.getPosition() + [self.distancey, self.distancey2]
    
    
    
    def getObservationMove2Bin(self):
    
    
        self.Distance = distance.euclidean(self.tcp.getPosition(), self.goal_target)
        
        return [self.sensors[i].getValue() for i in range(len(self.sensors)-1)] + [self.motors[i].getVelocity() for i in range(len(self.motors)-1)]  + self.tcp.getPosition() +  [self.Distance] + self.goal_target
    
    
    
    
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
            self.motors[i].setVelocity(0)
            # Get sensors and enable them
            self.sensors[i] = self.supervisor.getDevice(self.joint_names[i]+'_sensor')
            self.sensors[i].enable(self.TIME_STEP)