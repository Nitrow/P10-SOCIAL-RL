import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file, avg=100):
	if len(scores)<101:
		return
	else:
	    running_avg = np.zeros(len(scores)-avg)
	    for i in range(len(running_avg)):
	        running_avg[i] = np.mean(scores[i-100:(i+1)])
	    plt.plot(x, running_avg)
	    plt.title('Running average of previous 100 scores')
	    plt.savefig(figure_file)
