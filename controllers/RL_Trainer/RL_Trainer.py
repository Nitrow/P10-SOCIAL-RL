# Copyright 1996-2021 Cyberbotics Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from gym import spaces
from controller import Supervisor
from scipy.spatial import distance
import torch as th

try:
    import gym
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
except ImportError:
    sys.exit(
        'Please make sure you have all dependencies installed. '
        'Run: "pip3 install numpy gym stable_baselines3"'
    )


class OpenAIGymEnvironment(Supervisor, gym.Env):
    def __init__(self, max_episode_steps=1000):
        super().__init__()
        
        
        self.oldDistance = 0
        self.distance = 0
        self.counter = 0

        self.action_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-5, high=5, shape=(6,), dtype=np.float32)
        self.spec = gym.envs.registration.EnvSpec(id='WebotsEnv-v0', max_episode_steps=max_episode_steps)

        # Environment specific
        self.__timestep = int(self.getBasicTimeStep())

        # Tools
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.__timestep)
        
        
        self.done = False
        self.success = False
        
        
        self.__wheels = []
        self.joint_names = ['shoulder_pan_joint',
                       'shoulder_lift_joint',
                       'elbow_joint',
                       'wrist_1_joint',
                       'wrist_2_joint',
                       'wrist_3_joint']
                       
        
        
        
       
        

    def wait_keyboard(self):
        while self.keyboard.getKey() != ord('Y'):
            super().step(self.__timestep)

    def reset(self):
        # Reset the simulation
        self.simulationResetPhysics()
        self.simulationReset()
        super().step(self.__timestep)

        # Motors
        joint_names = ['shoulder_pan_joint',
                       'shoulder_lift_joint',
                       'elbow_joint',
                       'wrist_1_joint',
                       'wrist_2_joint',
                       'wrist_3_joint']
                       
        self.motors = [0] * len(joint_names)
        self.sensors = [0] * len(joint_names)
        
        
        for i in range(len(self.joint_names)):
            # Get motors
            self.motors[i] = self.getDevice(self.joint_names[i])
            self.motors[i].setPosition(float('inf'))
            self.motors[i].setVelocity(0)

            # Get sensors and enable them
            self.sensors[i] = super().getDevice(self.joint_names[i] + '_sensor')
            self.sensors[i].enable(self.__timestep)

        
        if self.success == True:
                
            self.goal.setSFVec3f([random.uniform(-0.35, 0.35), 1, 0.6])
        
            self.target = np.array(self.goal.getSFVec3f())
        
        
        self.robot_node = super().getFromDef("UR3")
            
        self.collision1 = super().getDevice("touch_sensor1")
        self.collision2 = super().getDevice("touch_sensor2")
        self.collision3 = super().getDevice("touch_sensor3")
        self.collision4 = super().getDevice("touch_sensor4")
        self.collision5 = super().getDevice("touch_sensor5")
        self.collision6 = super().getDevice("touch_sensor6")
 
        self.collision1.enable(self.__timestep)
        self.collision2.enable(self.__timestep)
        self.collision3.enable(self.__timestep)
        self.collision4.enable(self.__timestep)
        self.collision5.enable(self.__timestep)
        self.collision6.enable(self.__timestep)
        
       
        
        self.goal = self.getFromDef("TARGET").getField("translation")
        self.box = self.getFromDef('UR_END')
        
        
        self.goal.setSFVec3f([0, 1, 0.6])
        self.target = np.array(self.goal.getSFVec3f())
        self.box_pos_world = self.box.getPosition()
        
        # Internals
        super().step(self.__timestep)
        
        
        self.done = False

        # Open AI Gym generic
        return np.array([0, 0, 0, 0, 0, 0])

    def step(self, action):
        # Execute the action
        
        
        #print(action)
        for i in range(len(self.joint_names)-1):
            self.motors[i].setPosition(self.sensors[i].getValue()+action[i]*(self.__timestep/1000))
        super().step(self.__timestep)

        # Observation
        rot_ur3e = np.array(self.robot_node.getOrientation())

        rot_ur3e.reshape(3, 3)

        rot_ur3e = np.transpose(rot_ur3e)


        pos_ur3e = np.array(self.robot_node.getPosition())

        self.box_pos_world = np.array(self.box.getPosition())
        

        self.distance = distance.euclidean(self.box_pos_world, self.target)

        state = np.array([self.sensors[0].getValue(), self.sensors[1].getValue(), self.sensors[2].getValue(), self.sensors[3].getValue(), self.sensors[4].getValue(), self.target[0]])
        
        if self.counter == 0:
            self.oldDistance = self.distance
        
        #print("STATE:", state)

        #self.distance = math.sqrt(pow((target[0] - self.target_position[0]),2) + pow((target[1] - self.target_position[1]),2) + pow((target[2] - self.target_position[2]),2))
        #print("DISTANCE:", self.distance)

        reward = (self.oldDistance - self.distance)*1000
        
        self.oldDistance = self.distance
        
        
        #print(self.target)
        #print(reward)

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
                    
        if self.counter == 300:
            print("TIMEOUT")
            self.success = False
            self.done = True
        if self.distance < 0.05:
            print("SUCCESS")
            self.success = True
            self.done = True
            reward = 1000
        if self.collision1.getValue() == 1 or self.collision2.getValue() == 1 or self.collision3.getValue() == 1 or self.collision4.getValue() == 1 or self.collision5.getValue() == 1 or self.collision6.getValue() == 1:
            print("COLLISION")
            self.done = True
            self.success = False
            reward = -300
        
        

        return [state, float(reward), self.done, {}]




def main():
    # Initialize the environment
    env = OpenAIGymEnvironment()
    check_env(env)
    
    
    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=[dict(pi=[256, 256], vf=[256, 256])])

    # Train
    model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, n_steps=2048, verbose=1, tensorboard_log="/home/asger/P10-XRL/controllers/masterStable/tensorboard")
    model.learn(total_timesteps=1e6)

    # Replay
    print('Training is finished, press `Y` for replay...')
    env.wait_keyboard()

    obs = env.reset()
    for _ in range(100000):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        print(obs, reward, done, info)
        if done:
            obs = env.reset()


if __name__ == '__main__':
    main()