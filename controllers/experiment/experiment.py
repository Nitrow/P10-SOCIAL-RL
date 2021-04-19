#!/usr/bin/env python3
"""experiment controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Camera, Supervisor, Display
import numpy as np
import cv2
# create the Robot instance.
robot = Robot()
# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())
cv2.startWindowThread()
cv2.namedWindow("preview")
padding = np.array([10, 10])
camera = Camera("camera")
camera.enable(timestep)
camera.recognitionEnable(timestep)
display = Display("display")

width = camera.getWidth()
height = camera.getHeight()

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getDevice('motorname')
#  ds = robot.getDevice('dsname')
#  ds.enable(timestep)

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    cameraData = camera.getImage()
    image = np.frombuffer(cameraData, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
    
    for object in camera.getRecognitionObjects():
        size = np.array(object.get_size_on_image()) + padding
        start_point = np.array(object.get_position_on_image()) - (size / 2) 
        start_point =  np.array([int(x) for x in start_point])
        end_point = start_point + size
        color = np.rint(np.array(object.get_colors())*255)
        thickness = 2
        #color = [0, 255, 0]
        color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ]))
        image = cv2.rectangle(image, tuple(start_point), tuple(end_point), tuple(color), thickness)        
    #print(camera.getRecognitionObjects())
    #print(cameraData)
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()

    # Process sensor data here.
    #ir = display.imageNew(image.tolist(), Display.RGB)#, width, height)
    #display.imagePaste(ir, 20, 0)
    #display.imageDelete(ir)
    
    cv2.imshow("preview", image)
    cv2.waitKey(timestep)
    # Enter here functions to send actuator commands, like:
    #  motor.setPosition(10.0)
    pass

# Enter here exit cleanup code.
