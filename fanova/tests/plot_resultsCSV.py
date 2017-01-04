from pyfanova.fanova import Fanova
from pyfanova.visualizer import Visualizer
from pyfanova.fanova_from_csv import FanovaFromCSV

import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('Qt4Agg')

fname = 'data/sample_csv.csv'


f = FanovaFromCSV(fname)
vis = Visualizer(f, mpl_options=None)

fig = plt.figure()
ax = fig.add_subplot(221)
vis.plot_marginal("nbands",ax=ax)
ax = fig.add_subplot(222)
vis.plot_marginal("target_corr",ax=ax)
ax = fig.add_subplot(223)
vis.plot_categorical_marginal("grid_type",ax=ax, mode='boxplot')
ax = fig.add_subplot(224)
vis.plot_categorical_marginal("reg_model",ax=ax, mode='boxplot')


fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
vis.plot_pairwise_marginal("nbands","target_corr", ax=(ax1,ax2))

plt.show()
