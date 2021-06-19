import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file, avg=50):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-avg):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous ' + str(avg) + ' scores')
    plt.savefig(figure_file)