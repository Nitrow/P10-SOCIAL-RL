#!/usr/bin/env python3

from controller import Robot, Camera, Supervisor, Display, Connector, Keyboard
import numpy as np
import cv2
import random
import os
import math

#viewpoint:

# POS: 1.6931910127331369 1.781756528011428 5.350749519514143
# ROT: -0.9975109062294765 -0.05050270631532978 0.049208420093273746 0.06363818669844469
supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())
display_explanation = supervisor.getDevice("display_explanation")
screen = 3
key = 0
setScreen = True
keyboard = Keyboard()
keyboard.enable(timestep)
root_children = supervisor.getRoot().getField("children")
importCans = True
deleteCans = True
canSelection = None
prevCan = None
can_ID = None
gravityField = root_children.getMFNode(0).getField("gravity")
gravityField.setSFFloat(0)
while supervisor.step(timestep) != -1:
	prevKey = key
	key=keyboard.getKey()
	if (key == 316 or key == 32) and prevKey != key: screen += 1
	#if key == 314 and prevKey != key: screen -= 1

	if screen == 0:
		display_explanation.setOpacity(1)
		display_explanation.setAlpha(1)
		display_explanation.setColor(0xFFFFFF)
		display_explanation.fillRectangle(0, 0, display_explanation.getWidth(), display_explanation.getHeight())
		display_explanation.setColor(0x000000)
		display_explanation.setFont("Lucida Console", 64, True)
		display_explanation.drawText("Welcome!", 300, 50)
		display_explanation.setFont("Lucida Console", 32, True)
		display_explanation.drawText("In this experiment, your task is to", 20, 160)
		display_explanation.drawText("sort different aluminium cans in col-", 20, 214)
		display_explanation.drawText("laboriation with a UR-3 manipulator.", 20, 266)
		display_explanation.drawText("There are three types of cans:", 20, 366)
		display_explanation.drawText("Yellow", 200, 466)
		display_explanation.drawText("Red", 500, 466)
		display_explanation.drawText("Green", 700, 466)
		if importCans:
			root_children.importMFNode(-1, "Red.wbo")
			root_children.importMFNode(-1, "Yellow.wbo")
			root_children.importMFNode(-1, "Green.wbo")
			importCans = False
			deleteCans = True

	if screen == 1:
		importCans = True
		if deleteCans:
			supervisor.getFromDef("INTRO-CAN-RED").remove()
			supervisor.getFromDef("INTRO-CAN-YELLOW").remove()
			#supervisor.getFromDef("INTRO-CAN-GREEN").remove()
			deleteCans = False
		display_explanation.setColor(0xFFFFFF)
		display_explanation.fillRectangle(0, 0, display_explanation.getWidth(), display_explanation.getHeight())
		display_explanation.setColor(0x000000)
		display_explanation.drawText("Your task is to sort the yellow and ", 20, 160)
		display_explanation.drawText("green cans. You can do it by clicking", 20, 214)
		display_explanation.drawText("first on the can, then on the crate", 20, 266)
		display_explanation.drawText("you wish to put it into.", 20, 316)
		display_explanation.drawText("Try it out with this can:", 20, 416)
		selection = supervisor.getSelected()
		selectionName = selection.getDef()

		if "CAN" in selectionName:
			canSelection = selection

		elif ("CRATE" in selectionName) and canSelection:
			new_position = selection.getField("translation").getSFVec3f()
			new_position[1] = 1.5
			canSelection.getField("translation").setSFVec3f(new_position)
			gravityField.setSFFloat(9.81)
			prevCan = canSelection
			canSelection = None
		if prevCan:
			if prevCan.getField("translation").getSFVec3f()[1] < 0.5:
				prevCan = None
				screen += 1

	if screen == 2:
		display_explanation.setColor(0xFFFFFF)
		display_explanation.fillRectangle(0, 0, display_explanation.getWidth(), display_explanation.getHeight())
		display_explanation.setColor(0x000000)
		display_explanation.drawText("Well done! ", 20, 100)
		display_explanation.drawText("The robot will also sort cans, that ", 20, 214)
		display_explanation.drawText("are either red or green.", 20, 266)
		display_explanation.drawText("The cans will arrive from the conve-", 20, 336)
		display_explanation.drawText("yor belt, and you'll gain scores for", 20, 386)
		display_explanation.drawText("sorting them correctly.", 20, 436)
		display_explanation.drawText("You get 1 point for each correct sort", 20, 506)
		display_explanation.drawText("and -1 point for each mistake or ", 20, 556)
		display_explanation.drawText("missed can.", 20, 606)

	if screen == 3:
		display_explanation.setColor(0xFFFFFF)
		display_explanation.fillRectangle(0, 0, display_explanation.getWidth(), display_explanation.getHeight())
		display_explanation.setColor(0x000000)
		x = 100
		display_explanation.drawText("The robot is controlled by a specific", 20, x)
		x += 50
		display_explanation.drawText("type of machine learning algorithm ", 20, x)
		x += 50
		display_explanation.drawText("called reinforcement learning, where", 20, x)
		x += 50
		display_explanation.drawText("the robot tries to reinforce positive", 20, x)
		x += 50
		display_explanation.drawText("behavior, in this case sorting correctly.", 20, x)
		x += 100
		display_explanation.drawText("Therefore it is preferred to let the ", 20, x)
		x += 50
		display_explanation.drawText("robot sort as many cans correctly as ", 20, x)
		x += 50
		display_explanation.drawText("possible. The algorithm has some mistakes", 20, x)
		x += 50
		display_explanation.drawText("however, that should be prevented if possible.", 20, x)