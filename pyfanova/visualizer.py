import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm,colors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import logging


class Visualizer(object):

    def __init__(self, fanova,custom_scale = [], mpl_options='custom'):
        self._fanova = fanova
        self.outfile_type = ".pdf" #.png
        self.custom_scale = custom_scale
        self.do_custom_scaling = bool(custom_scale)

        if mpl_options == 'custom':
            matplotlib.rcParams.update({'font.size': 32, 'figure.figsize': [30,24],'lines.linewidth':14,'lines.markersize':22})

    def create_all_plots(self, directory, **kwargs):
        """
            Create plots for all main effects.
        """
        assert os.path.exists(directory), "directory %s doesn't exist" % directory

        #categorical parameters
        for param_name in self._fanova.get_config_space().get_categorical_parameters():
            plt.clf()
            outfile_name = os.path.join(directory, param_name.replace(os.sep, "_") + ".png")
            print "creating %s" % outfile_name
            self.plot_categorical_marginal(param_name)
            plt.savefig(outfile_name)

        #continuous and integer parameters
        params_to_plot = []
        params_to_plot.extend(self._fanova.get_config_space().get_continuous_parameters())
        params_to_plot.extend(self._fanova.get_config_space().get_integer_parameters())
        for param_name in params_to_plot:
            plt.clf()
            outfile_name = os.path.join(directory, param_name.replace(os.sep, "_") + ".png")
            print "creating %s" % outfile_name
            self.plot_marginal(param_name, **kwargs)
            plt.savefig(outfile_name)

    def create_most_important_pairwise_marginal_plots(self, directory, n=20):
        categorical_parameters = self._fanova.get_config_space().get_categorical_parameters()

        most_important_pairwise_marginals = self._fanova.get_most_important_pairwise_marginals(n)
        for param1, param2 in most_important_pairwise_marginals:
            if param1 in categorical_parameters or param2 in categorical_parameters:
                print "skipping pairwise marginal plot %s x %s, because one of them is categorical" % (param1, param2)
                continue
            outfile_name = os.path.join(directory, param1.replace(os.sep, "_") + "x" + param2.replace(os.sep, "_") + ".png")
            plt.clf()
            print "creating %s" % outfile_name
            self.plot_pairwise_marginal(param1, param2).show()
            plt.savefig(outfile_name)

    def plot_categorical_marginal(self, param, ax = None, mode = 'bar'):
        categorical_size = self._fanova.get_config_space().get_categorical_size(param)

        labels = self._fanova.get_config_space().get_categorical_values(param)
        logging.debug("LABELS:")
        logging.debug(labels)

        indices = np.asarray(range(categorical_size))
        width = 0.5
        marginals = [self._fanova.get_categorical_marginal_for_value(param, i) for i in range(categorical_size)]
        mean, std = zip(*marginals)

        if not ax:
            ax = plt.axes()

        if mode == 'bar':
            ax.bar(indices, mean, width, color='red', yerr=std)
        elif mode == 'boxplot':
            b = ax.boxplot([[x] for x in mean], 0, '', labels=labels)
            min_y = mean[0]
            max_y = mean[0]
            # blow up boxes
            for box, std_ in zip(b["boxes"], std):
                y = box.get_ydata()
                y[2:4] = y[2:4] + std_
                y[0:2] = y[0:2] - std_
                y[4] = y[4] - std_
                box.set_ydata(y)
                min_y = min(min_y, y[0] - std_)
                max_y = max(max_y, y[2] + std_)
            ax.set_ylim([min_y, max_y])
        ax.set_xticks(indices + width / 2.0, labels)
        ax.set_ylabel("Performance")

        return plt

    def _check_param(self, param):
        if isinstance(param, int):
            dim = param
            param_name = self._fanova.get_config_space().get_parameter_names()[dim]
        else:
            assert param in self._fanova.param_name2dmin, "param %s not known" % param
            dim = self._fanova.param_name2dmin[param]
            param_name = param

        return (dim, param_name)

    def plot_pairwise_marginal(self, param_1, param_2, lower_bound_1=0, upper_bound_1=1, lower_bound_2=0, upper_bound_2=1, resolution=200, ax=None):

        dim1, param_name_1 = self._check_param(param_1)
        dim2, param_name_2 = self._check_param(param_2)

        grid_1 = np.linspace(lower_bound_1, upper_bound_1, resolution)
        grid_2 = np.linspace(lower_bound_2, upper_bound_2, resolution)

        zz = np.zeros([resolution * resolution])
        zz_standard_dev = np.zeros([resolution * resolution])
        for i, y_value in enumerate(grid_2):
            for j, x_value in enumerate(grid_1):
                zz[i * resolution + j],zz_standard_dev[i * resolution + j] = self._fanova._get_marginal_for_value_pair(dim1, dim2, x_value, y_value)

        zz = np.reshape(zz, [resolution, resolution])
        zz_standard_dev = np.reshape(zz_standard_dev, [resolution, resolution])

        display_grid_1 = [self._fanova.unormalize_value(param_name_1, value) for value in grid_1]
        display_grid_2 = [self._fanova.unormalize_value(param_name_2, value) for value in grid_2]

        display_xx, display_yy = np.meshgrid(display_grid_1, display_grid_2)

        single_plot = False



        if single_plot:
            if not ax:
                fig = plt.figure()
                ax = Axes3D(fig)

            surface = ax.plot_surface(display_xx, display_yy, zz, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False, alpha = 0.5)
            #contour(display_xx, display_yy, zz, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False, alpha = 0.5)
            ax.set_xlabel(param_name_1)
            ax.set_ylabel(param_name_2)
            ax.set_zlabel("Marginalized Z-AUC")

            fig.colorbar(surface, shrink=0.5, aspect=5)

            """
            TK: add the points to plot where the function was evaluated

            """
            eval_values_param_1 = self._fanova.get_test_values_for_param(param_name_1)
            eval_values_param_2 = self._fanova.get_test_values_for_param(param_name_2)



            eval_x_values_1 = []
            eval_x_values_2 = []
            eval_y_values = []

            if param_name_1 in self._fanova.get_config_space().get_integer_parameters():
                for value in eval_values_param_1:
                    eval_x_values_1.append(int(value))
            else:
                for value in eval_values_param_1:
                    eval_x_values_1.append(float(value))

            if param_name_2 in self._fanova.get_config_space().get_integer_parameters():
                for value in eval_values_param_2:
                    eval_x_values_2.append(int(value))
            else:
                for value in eval_values_param_2:
                    eval_x_values_2.append(float(value))


            for ii in range(len(eval_x_values_1)):
                normalized_val_1 = self._fanova.normalize_value(param_name_1,eval_x_values_1[ii])
                normalized_val_2 = self._fanova.normalize_value(param_name_2,eval_x_values_2[ii])
                eval_y_values.append(self._fanova._get_marginal_for_value_pair(dim1,dim2,normalized_val_1,normalized_val_2)[0])
            #ax.scatter(eval_x_values_1, eval_x_values_2, eval_y_values,linewidth=0, antialiased=False)
            ax.plot

        else:
            if not ax:
                fig = plt.figure()
                (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(30,10))
            else:
                print(ax)
                ax1, ax2 = ax
            n_colorbar = 10 #granularity of the colorbar

            fig = plt.gcf()
            #ax = fig.gca(projection='3d')

            #ax = Axes3D(fig)
            plt.subplots_adjust(wspace = 0.4)
            #surface = ax.plot_surface(display_xx, display_yy, zz, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False, alpha = 0.5)

            #cmap_custom = colors.LinearSegmentedColormap('my_map', cdict, N=256, gamma=1.0)
            if self.do_custom_scaling:
                cont_mean = ax1.contourf(display_xx, display_yy, zz, rstride=1, cstride=1,cmap = cm.jet_r, linewidth=0, antialiased=False, alpha = 0.5,vmin= self.custom_scale[0], vmax=self.custom_scale[1])
            else:
                cont_mean = ax1.contourf(display_xx, display_yy, zz, rstride=1, cstride=1,cmap = cm.jet_r, linewidth=0, antialiased=False, alpha = 0.5)
            cont_sd = ax2.contourf(display_xx, display_yy, zz_standard_dev, rstride=1, cstride=1,cmap = cm.autumn_r, linewidth=0, antialiased=False, alpha = 0.5)

            ax1.set_xlabel(param_name_1)
            ax1.set_ylabel(param_name_2)
            ax1.set_title('Marginalized Z-AUC')

            ax2.set_xlabel(param_name_1)
            #ax2.set_ylabel(param_name_2)
            ax2.set_title('Std of Prediction')

            ## split plot for colorbar and plot

            #for ax1
            div = make_axes_locatable(ax1)
            cax = div.append_axes("right", size="12%", pad=0.05)


            cbar = plt.colorbar(cont_mean, cax=cax, ticks = np.linspace(np.amin(zz),np.amax(zz),num = n_colorbar), format="%.3g")

            #cbar.set_label('Marginalized Z-AUC',labelpad=10)#, size=22)
            #ax1.xaxis.set_visible(False)
            #ax1.yaxis.set_visible(False)


            #ax1.colorbar(cont_mean, shrink=0.5, aspect=5)

            #for ax2

            div = make_axes_locatable(ax2)
            cax = div.append_axes("right", size="12%", pad=0.05)
            cbar = plt.colorbar(cont_sd, cax=cax, ticks = np.linspace(np.amin(zz_standard_dev),np.amax(zz_standard_dev),num = n_colorbar), format="%.3g")
            cbar.set_label('Standard Deviation')#, size=22)

            """
            TK: add the points to plot where the function was evaluated

            """
            eval_values_param_1 = self._fanova.get_test_values_for_param(param_name_1)
            eval_values_param_2 = self._fanova.get_test_values_for_param(param_name_2)



            eval_x_values_1 = []
            eval_x_values_2 = []
            eval_y_values = []

            if param_name_1 in self._fanova.get_config_space().get_integer_parameters():
                for value in eval_values_param_1:
                    eval_x_values_1.append(int(value))
            else:
                for value in eval_values_param_1:
                    eval_x_values_1.append(float(value))

            if param_name_2 in self._fanova.get_config_space().get_integer_parameters():
                for value in eval_values_param_2:
                    eval_x_values_2.append(int(value))
            else:
                for value in eval_values_param_2:
                    eval_x_values_2.append(float(value))


            for ii in range(len(eval_x_values_1)):
                normalized_val_1 = self._fanova.normalize_value(param_name_1,eval_x_values_1[ii])
                normalized_val_2 = self._fanova.normalize_value(param_name_2,eval_x_values_2[ii])
                eval_y_values.append(self._fanova._get_marginal_for_value_pair(dim1,dim2,normalized_val_1,normalized_val_2)[0])

            #ax1.scatter(eval_x_values_1, eval_x_values_2, eval_y_values, antialiased=False,marker='x',c = 'black')
            #ax2.scatter(eval_x_values_1, eval_x_values_2, eval_y_values, antialiased=False,marker='x',c = 'black')

            ax1.scatter(eval_x_values_1, eval_x_values_2, s = 10,antialiased=False,marker='x',c = 'black')
            ax2.scatter(eval_x_values_1, eval_x_values_2, s = 10,antialiased=False,marker='x',c = 'black')



        return fig


    def plot_marginal(self, param, lower_bound=0, upper_bound=1, is_int=False, resolution=100, ax = None):
        if isinstance(param, int):
            dim = param
            param_name = self._fanova.get_config_space().get_parameter_names()[dim]
        else:
            assert param in self._fanova.param_name2dmin, "param %s not known" % param
            dim = self._fanova.param_name2dmin[param]
            param_name = param

        grid = np.linspace(lower_bound, upper_bound, resolution)
        display_grid = [self._fanova.unormalize_value(param_name, value) for value in grid]

        mean = np.zeros(resolution)
        std = np.zeros(resolution)
        for i in xrange(0, resolution):
            (m, s) = self._fanova.get_marginal_for_value(dim, grid[i])
            mean[i] = m
            std[i] = s
        mean = np.asarray(mean)
        std = np.asarray(std)

        lower_curve = mean - std
        upper_curve = mean + std

        if not ax:
            ax = plt.axes()

        if np.diff(display_grid).std() > 0.000001 and param_name in self._fanova.get_config_space().get_continuous_parameters():
            #HACK for detecting whether it's a log parameter, because the config space doesn't expose this information
            ax.semilogx(display_grid, mean, 'b')
            print "printing %s semilogx" % param_name
        else:
            ax.plot(display_grid, mean, 'b')
        ax.fill_between(display_grid, upper_curve, lower_curve, facecolor='red', alpha=0.6)
        ax.set_xlabel(param_name)

        ax.set_ylabel("Marginalized Z-AUC")

        """
        TK: add the points to plot where the function was evaluated
        """
        ax.hold(True)

        eval_values = self._fanova.get_test_values_for_param(param_name)

        eval_x_values = []
        eval_y_values = []
        if param_name in self._fanova.get_config_space().get_integer_parameters():
            for value in eval_values:
                eval_x_values.append(int(value))
                (m,s) = self._fanova.get_marginal_for_value(dim, self._fanova.normalize_value(param_name,value))
                eval_y_values.append(m)

        else:
            for value in eval_values:
                eval_x_values.append(float(value))
                (m,s) = self._fanova.get_marginal_for_value(dim, self._fanova.normalize_value(param_name,value))
                eval_y_values.append(m)


        ax.plot(eval_x_values,eval_y_values,'o',color = '0.85')

        print("do_custom_scaling is: {}".format(self.do_custom_scaling))
        if self.do_custom_scaling:
            ax.set_ylim(self.custom_scale)
            ax.tick_params(axis='both', which='major', pad=20)
            ax.yaxis.grid(color='grey', linestyle='dashed',linewidth=3)
            ax.xaxis.grid(color='grey', linestyle='dashed',linewidth=3)
            ax.set_axisbelow(True)
            #plt.ylim = self.custom_scale

        return ax
