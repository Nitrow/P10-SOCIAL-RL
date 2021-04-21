from controller import Robot, Display


class CameraDisplay(Robot):
    timeStep = 32
    motors = []

    def __init__(self):
        super(CameraDisplay, self).__init__()
        self.camera = self.getDevice('camera')
        self.display = self.getDevice('display')
        self.camera.enable(self.timeStep)

    def run(self):
        while True:
            a = self.camera.getImageArray()
            if a:
                image = self.display.imageNew(a, Display.RGB)
                self.display.imagePaste(image, 0, 0)
                self.display.imageDelete(image)
            if self.step(self.timeStep) == -1:
                break


controller = CameraDisplay()
controller.run()