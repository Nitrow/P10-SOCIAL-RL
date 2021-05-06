from pyutil import filereplace

def fileChanger(textToSearch, textToReplace):
    for file in os.listdir("resources"):
        f = "resources/" + file
        filereplace(f, textToSearch, textToReplace)


fileChanger('translation 0 0.02', 'translation 0 0.04')