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

supervisor = Supervisor()
robot_node = supervisor.getFromDef("UR3")

velocity = 3.14
# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    val1 = ds1.getValue()
    val2 = ds2.getValue()

#trans_field = robot_node.getField("translation")
counter = 1

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
    motors[i] = supervisor.getDevice(joint_names[i])
    motors[i].setPosition(float('inf'))
    #motors[i].setPosition(float('inf'))
    #motors[i].setVelocity(1 * MAX_SPEED)

    # Get sensors and enable them
    sensors[i] = supervisor.getDevice(joint_names[i]+'_sensor')
    sensors[i].enable(TIME_STEP)


dir = 2

while supervisor.step(TIME_STEP) != -1:
    counter += 1
    #print(counter)
    
    
    motors[0].setVelocity(0)
    motors[1].setVelocity(1)
    motors[2].setVelocity(0)
    motors[3].setVelocity(0)
    motors[4].setVelocity(0)
    motors[5].setVelocity(0)
    
    
    #print(robot_node.getNumberOfContactPoints(True))
        
    if counter > max_iter:
        counter = 0
        dir *=-1
        
        #supervisor.simulationReset()
        #conveyor_node.restartController()
        #tv_node.restartController()
    # read sensors outputs
    # print("Sensor outputs:\n")
    # readings = ""
    # for i in range(len(joint_names)):
        # readings += '\t' + str(sensors[i].getValue())
    # print(readings + '\n')
