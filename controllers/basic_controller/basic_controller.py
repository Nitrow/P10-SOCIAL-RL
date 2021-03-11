#!/usr/bin/env python3
"""


 WeBots tutorial: 
 https://cyberbotics.com/doc/guide/ure
"""


from controller import Robot, Motor

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

MAX_SPEED = 6.28

joint_names = [ 'shoulder_pan_joint',
                'shoulder_lift_joint',
                'elbow_joint',
                'wrist_1_joint',
                'wrist_2_joint',
                'wrist_3_joint']
                
motors = [0] * len(joint_names)
sensors = [0] * len(joint_names)
 
for i in range(len(joint_names)):
    # Get motors
    motors[i] = robot.getDevice(joint_names[i])
    motors[i].setPosition(0)
    #motors[i].setPosition(float('inf'))
    #motors[i].setVelocity(1 * MAX_SPEED)
    
    # Get sensors and enable them
    sensors[i] = robot.getDevice(joint_names[i]+'_sensor')
    sensors[i].enable(timestep)

     


# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    # read sensors outputs
    print("Sensor outputs:\n")
    readings = ""
    for i in range(len(joint_names)):
        readings += '\t' + str(sensors[i].getValue())
    print(readings + '\n')
    
