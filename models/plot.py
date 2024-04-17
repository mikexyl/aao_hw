import os
import numpy as np


class Plot:
    Plots = {}

    @staticmethod
    def save_all_plots(path):
        if not os.path.exists(path):
            os.makedirs(path)
        for scene in Plot.Plots:
            for algorithm in Plot.Plots[scene]:
                Plot.Plots[scene][algorithm].save_plot(path)

    def save_plot(self, path):
        # check if path exists
        path = os.path.join(path, self.plot_name)
        if not os.path.exists(path):
            os.makedirs(path)
        # save plot data
        for key in self.values:
            np.save(os.path.join(path, key), self.values[key])

    def __init__(self, env, scene, algorithm):
        self.values = {}
        self.plot_style = {}
        self.plot_args = {}
        self.experiment = algorithm
        self.scene = scene
        self.env = env
        self.plot_name = scene + '_' + algorithm
        if scene not in self.Plots:
            self.Plots[scene] = {}
        self.Plots[scene][algorithm] = self

    def make_dirs(self):
        path = os.path.join(os.getcwd(), 'figures', self.scene, self.experiment)
        if not os.path.exists(path):
            os.makedirs(path)

    def add(self, key, value, plot=None, **kwargs):
        if key not in self.values:
            self.values[key] = []
            self.plot_style[key] = None
            self.plot_args[key] = {}
        self.values[key].append(value)
        if plot is not None:
            self.plot_style[key] = plot
        if kwargs:
            self.plot_args[key] = kwargs

    def _plot_grid(self):
        import matplotlib.pyplot as plt
        # set plot x limit from -1 to state shape 0
        plt.xlim(-1, self.env.state_shape[0])
        # set plot y limit from -1 to state shape 1
        plt.ylim(-1, self.env.state_shape[1])
        # generate x and y coordinates grid
        x = np.linspace(0, self.env.state_shape[0] - 1, self.env.state_shape[0])
        X, Y = np.meshgrid(x, x)
        # draw grid
        plt.scatter(X, Y, marker='s', c='k')

    def plot(self, save=False, **kwargs):
        self.make_dirs()
        import matplotlib.pyplot as plt
        for k, v in self.values.items():
            args = self.plot_args[k]
            if 'xlabel' in args:
                plt.xlabel(args['xlabel'])
                del args['xlabel']
            if 'ylabel' in args:
                plt.ylabel(args['ylabel'])
                del args['ylabel']
            if 'title' in args:
                plt.title(args['title'])
                del args['title']
            if self.plot_style[k] is None:
                plt.plot(v, label=k, **kwargs, **self.plot_args[k])
            elif self.plot_style[k] == 'trajectory':
                if self.scene == 'gridworld':
                    self._plot_grid()
                    v_np = np.array(v)
                    v_np[:, 1] = 4 - v_np[:, 1]
                    # plot arrows
                    plt.quiver(v_np[:-1, 0], v_np[:-1, 1], v_np[1:, 0] - v_np[:-1, 0], v_np[1:, 1] - v_np[:-1, 1],
                               scale_units='xy',
                               angles='xy', scale=1, **kwargs, **self.plot_args[k])
                    plt.xlabel(self.env.state_names[0])
                    plt.ylabel(self.env.state_names[1])
                elif self.scene == 'pendulum':
                    theta = np.array(v)[:, 0]
                    x = np.sin(theta)
                    y = np.cos(theta)
                    # draw x and y separately
                    plt.plot(x, label='x', color='r', **kwargs, **self.plot_args[k])
                    plt.plot(y, label='y', color='b', **kwargs, **self.plot_args[k])
            plt.legend()
            if save:
                plt.savefig('figures/' + self.scene + '/' + self.experiment + '/' + k + '.png')
            plt.show()

    def plot_policy(self, policy, save=False):
        self.make_dirs()
        import matplotlib.pyplot as plt
        if self.scene == 'gridworld':
            self._plot_grid()
            # arrows
            for i in range(len(policy) - 1):
                x, y = self.env.get_pos(i)
                # draw policy as arrows, 0: right, 1: up, 2: left, 3: down
                l = 0.3
                dxdy = [[l, 0], [0, l], [-l, 0], [0, -l]]
                plt.arrow(x, 4 - y, dxdy[policy[i]][0], dxdy[policy[i]][1], head_width=0.1, head_length=0.1, fc='k',
                          ec='k')
        elif self.scene == 'pendulum':
            # policy to color map
            cmap = plt.get_cmap('jet')
            # get max and min q values
            max_p = np.max(policy)
            min_p = np.min(policy)
            # draw q values as colors
            for i in range(len(policy)):
                x, y = self.env.get_pos(i)
                # draw boxes and set size to a grid box without gap, and shape to square
                plt.scatter(x, y, marker='s', c=cmap((policy[i] - min_p) / (max_p - min_p)), s=100)
            # set x and y labels
            plt.xlabel(r'$\theta$')
            plt.ylabel(r'$\dot{\theta}$')
            # show color bar and set max and min values to -env.max_tau and env.max_tau
            cbar = plt.colorbar()
            cbar.set_ticks([0, 1])
            # set color bar labels with precision 1
            cbar.set_ticklabels(['%.1f' % -self.env.max_tau, '%.1f' % self.env.max_tau])
            # set color bar label
            cbar.set_label(r'$\tau$')
        plt.title('Policy of ' + self.experiment + ' in ' + self.scene)
        if save:
            plt.savefig('figures/' + self.scene + '/' + self.experiment + '/policy.png')
        plt.show()

    def plot_state_value_function(self, V, save=False):
        self.make_dirs()
        import matplotlib.pyplot as plt
        # colormap from q values to colors
        cmap = plt.get_cmap('jet')
        # get max and min q values
        max_v_min = np.min(V)
        max_v_max = np.max(V)
        if self.scene == 'gridworld':
            self._plot_grid()
        # draw q values as colors
        for i in range(len(V)):
            x, y = self.env.get_pos(i)
            y = self.env.state_shape[0] - 1 - y if self.scene == 'gridworld' else y
            # draw boxes and set size to a grid box without gap, and shape to square
            plt.scatter(x, y, marker='s', c=cmap((V[i] - max_v_min) / (max_v_max - max_v_min)), s=1000)

        # draw colorbar and set max and min values
        cbar = plt.colorbar()
        cbar.set_ticks([0, 1])
        # set color bar labels with precision 1
        cbar.set_ticklabels(['%.1f' % max_v_min, '%.1f' % max_v_max])
        # set color bar label
        cbar.set_label('value')
        # set x and y labels
        plt.xlabel(self.env.state_names[0])
        plt.ylabel(self.env.state_names[1])
        plt.title('State-Value Function of ' + self.experiment + ' in ' + self.scene)
        if save:
            plt.savefig('figures/' + self.scene + '/' + self.experiment + '/value_function.png')
        plt.show()

    def clear(self, k):
        # remove the key k from all dictionaries
        for i in range(len(k)):
            if k[i] in self.values:
                del self.values[k[i]]


    @staticmethod
    def plot_compare(scenes, algorithms, key, title, save=False, **kwargs):
        # get plot_interval and remove it from kwargs
        plot_interval = kwargs.get('plot_interval', False)
        del kwargs['plot_interval']
        import matplotlib.pyplot as plt
        plt.figure()

        for scene, algorithm in zip(scenes, algorithms):
            v = Plot.Plots[scene][algorithm].values[key]
            if plot_interval:
                v = np.array(v)
                plt.fill_between(range(len(v)), v - v.std(), v + v.std(), alpha=0.5, label=scene + '_' + algorithm,
                                 **kwargs)
            else:
                plt.plot(v, label=algorithm, **kwargs)

        plt.legend()
        plt.title(title)
        if save:
            plt.savefig('figures/' + title + '.png')
        plt.show()

    @staticmethod
    def get_all_plot_names():
        plots = []
        for scene in Plot.Plots:
            for algorithm in Plot.Plots[scene]:
                plots.append(scene + '_' + algorithm)
        return plots
