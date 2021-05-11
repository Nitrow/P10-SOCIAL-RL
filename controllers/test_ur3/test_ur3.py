"""ghost_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
motor1 = robot.getDevice('shoulder_pan_joint')

ds1 = robot.getDevice('shoulder_pan_joint_sensor')
ds1.enable(timestep)

motor2 = robot.getDevice('shoulder_lift_joint')

ds2 = robot.getDevice('shoulder_lift_joint_sensor')
ds2.enable(timestep)

pos1 = 2
pos2 = 2

velocity = 3.14
# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    val1 = ds1.getValue()
    val2 = ds2.getValue()

    # Process sensor data here.
    motor1.setPosition(pos1)
    motor2.setPosition(pos2)
    # Enter here functions to send actuator commands, like:
    #motor1.setVelocity(velocity)
    #motor2.setVelocity(velocity)
    
# Enter here exit cleanup code.