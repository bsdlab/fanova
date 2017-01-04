import os
from pyfanova.fanova import Fanova
from pyfanova.visualizer import Visualizer
from pyfanova.fanova_from_csv import FanovaFromCSV

import matplotlib
import matplotlib.pylab as plt
matplotlib.use('Qt4Agg')

fname = './data/smac_output/state-run649874530'
print('loading')
print(os.listdir(fname))
f = Fanova(fname)
vis = Visualizer(f)
vis.plot_marginal(0)
plt.show()
