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
from networks import ActorNetwork
from P10_DRL_Mark_SingleJoint.envs import P10_DRL_Mark_SingleJointEnv
import torch as T
from scipy.spatial import distance
from agent_dqn import DeepQNetwork
import torch.nn.functional as F
import ikpy
from ikpy.chain import Chain


import os
from scipy.spatial import distance

#import ikpy
#from ikpy.chain import Chain


from controller import Robot, Motor, Supervisor


class P10_LEVEL2_DQNEnv(gym.Env):

    def __init__(self):
        
        
        self.chkpt_dir='models/'
        
        
        ##Move2 policy setup##
        
        self.alpha = 0.0001
        self.beta = 0.0001
        self.input_dims = (26,)
        self.input_dims2 = (17,)
        self.layer1_size = 256
        self.layer2_size = 256
        self.n_actions = 6
        self.high = 1
        
        ##grasping policy setup##
        
        self.lr = 0.003
        self.n_actionsGrasp = 8
        self.input_dimsGrasp = (7,)
        self.fc1_dimsGrasp = 128
        self.fc2_dimsGrasp = 128
        
        


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
        self.actorGrasp = DeepQNetwork(self.lr, n_actions=self.n_actionsGrasp, input_dims=self.input_dimsGrasp,
                                  fc1_dims=self.fc1_dimsGrasp, fc2_dims=self.fc2_dimsGrasp, chkpt_dir=self.chkpt_dir)
        

        self.actorMove2Goal.load_checkpoint()
        self.actorMove2Bin.load_checkpoint()
        self.actorGrasp.load_checkpoint()




        
        
        random.seed(1)
       
        self.supervisor = Supervisor()
        self.TIME_STEP = int(self.supervisor.getBasicTimeStep())
        self.robot_node = self.supervisor.getFromDef("UR3")
        self.tcp = self.supervisor.getFromDef('TCP')
        self.can = self.supervisor.getFromDef('GREEN_ROTATED_CAN')
        self.goal_node = self.supervisor.getFromDef("TARGET").getField("translation")
        self.finger1 = self.supervisor.getFromDef("FINGER1")
        self.finger2 = self.supervisor.getFromDef("FINGER2")
        self.goal_pos = self.supervisor.getFromDef("GREEN_ROTATED_CAN").getField("translation")
        self.goal_rot = self.supervisor.getFromDef("GREEN_ROTATED_CAN").getField("rotation")
        self.fingers = [self.supervisor.getDevice('right_finger'), self.supervisor.getDevice('left_finger')]
        self.sensor_fingers = [self.supervisor.getDevice('right_finger_sensor'), self.supervisor.getDevice('left_finger_sensor')]
        self.goal_nodeCan = self.supervisor.getFromDef("GREEN_ROTATED_CAN")

        self.tcpy = self.supervisor.getFromDef("y")
        self.tcpy2 = self.supervisor.getFromDef("y2")
        
        
        self.goaly = self.supervisor.getFromDef("goaly")
        self.goaly2 = self.supervisor.getFromDef("goaly2") 

        self.timeout = 1500

        self.sensor_fingers[0].enable(self.TIME_STEP)
        self.sensor_fingers[1].enable(self.TIME_STEP)
        
        self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']


        self.motors = [0] * len(self.joint_names)
        self.sensors = [0] * len(self.joint_names)
        
        self.touch_sensors = [0] * 7

        self.robot_pos = np.array(self.robot_node.getPosition())       
   
        self.conveyor = self.supervisor.getFromDef('CONVEYOR')
       
        self.done = True
        
        self.my_chain = ikpy.chain.Chain.from_urdf_file("robot.urdf")   
        
        #self.goal_node.setSFVec3f([0,0.425,0.15])
        

        self.oldDistance = 0
        self.distancex = 0
        self.distancey = 0
        self.distancez = 0
        self.tcp_pos_world = self.tcp.getPosition()
        self.counter = 0
        self.tcp_can_total_dist = 0
        self.tcp_can_vertical_dist = 0
        self.tcp_can_horizontal_dist = 0
         
        self._getMotors()
        self._getSensors()

          
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(4,), dtype=np.float32)
        
        
    def reset(self):
        print('\n ------------------------------------ RESET ------------------------------------ \n')
        self.supervisor.simulationReset()
        
        self.conveyor.restartController()
        self.counter = 0
        self.supervisor.step(self.TIME_STEP) 
        self.supervisor.step(self.TIME_STEP) 
        
        
        
        goal = self.can.getField("translation").getSFVec3f()
        
        goal[1] = goal[1]+0.1
        
        self.goal_node.setSFVec3f(goal)
        
        self.supervisor.step(self.TIME_STEP) 
        self.supervisor.step(self.TIME_STEP) 
        self.supervisor.step(self.TIME_STEP) 
        
        self.supervisor.step(self.TIME_STEP) 
 
        self.done = False    

        state = [0]*4
        return state


    def step(self, action):
        
        self.counter = self.counter + 1
        action = 2
        
        if action == 0:
            

            while distance.euclidean(self.tcpy.getPosition(), self.goaly.getPosition()) > 0.05 and distance.euclidean(self.tcpy2.getPosition(), self.goaly2.getPosition()) > 0.05:                 
                
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
            
            
            for i in range(len(self.joint_names)):
            # Get motors
                self.motors[i] = self.supervisor.getDevice(self.joint_names[i])
                self.motors[i].setPosition(float('inf'))
                self.motors[i].setVelocity(0)
            # Get sensors and enable them
                self.sensors[i] = self.supervisor.getDevice(self.joint_names[i]+'_sensor')
                self.sensors[i].enable(self.TIME_STEP)
            
            observation = self.getObservationGrasp()
            
            state = T.tensor([observation]).to(self.actorGrasp.device)
            actions = self.actorGrasp.forward(state)
            action = T.argmax(actions).item()
            
            target_position_tcp = [self.tcp.getPosition()[0], self.tcp.getPosition()[1]+0.5, self.tcp.getPosition()[2]]
            
            self.up_pose = [self.sensors[i].getValue() for i in range(len(self.sensors))]
            
            orientation_axis = "Y"
            target_orientation = [0, 0, -1]
        
            joints = list(self.my_chain.inverse_kinematics(target_position_tcp, target_orientation=target_orientation, orientation_mode=orientation_axis))
            
            joints.pop(0)
            
            self.down_pose = joints
            
            print(self.down_pose)
            
            
            
            self.finger1.resetPhysics()
            self.finger2.resetPhysics()
            # Set actions
            self.motors[-1].setPosition(float(action))
            while not (self.sensors[-1].getValue() - action) < 0.01: self.supervisor.step(self.TIME_STEP)
            #if not self.isPresence:
            self._action_moveFingers(0)  # Open fingers
            for i in range(5): self.supervisor.step(self.TIME_STEP)
            
            for i in range(len(self.joint_names)):
                    self.motors[i].setPosition(self.down_pose[i])
                    
            for i in range(5): self.supervisor.step(self.TIME_STEP)
            self._action_moveFingers(1)  # close fingers
            for i in range(5): self.supervisor.step(self.TIME_STEP)
            for i in range(len(self.joint_names)):
                    self.motors[i].setPosition(self.up_pose[i])          
                
            
        elif action == 2:
                
            while distance.euclidean(self.tcpy.getPosition(), self.target) < 0.1: 
                               
                self._getMotors()
                self._getSensors()
                
                observation = self.getObservationMove2Bin()
                
                state = T.Tensor([[observation]]).to(self.actorMove2Bin.device)
                actions, _ = self.actorMove2Bin.sample_normal(state, reparameterize=False)
                action = actions.cpu().detach().numpy()[0][0]                

                for i in range(len(self.joint_names)):
                    self.motors[i].setVelocity(float(np.array(action)[i]))
           
                self.supervisor.step(self.TIME_STEP)
    
            for i in range(len(self.joint_names)):
                    self.motors[i].setVelocity(float(0))  
        #elif action == 3:
        #    print("")
               
        #self.supervisor.step(self.TIME_STEP)   
        
        
           
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
    
    
    def getObservationGrasp(self):
    
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
                 