from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


if __name__ == "__main__":

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # make data
    action = 2
    theta = np.linspace(-np.pi, np.pi, 33, False)
    dtheta = np.linspace(-8, 8, 33, False)
    # theta = np.pi/6
    # action = np.linspace(-2, 2, 30, False)
    theta, dtheta = np.meshgrid(theta, dtheta)
    # dtheta, action = np.meshgrid(dtheta, action)
    y =  10*(theta**2 + 0.1*dtheta**2 + 1*action**2) + 100/(1+np.exp(-100*(theta**2-(np.pi/3)**2))) - 150
    # Plot the surface.
    surf = ax.plot_surface(theta, dtheta, y, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    plt.xlabel = "theta"
    plt.ylabel = "dtheta"
    plt.xlabel = "cost value"

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()