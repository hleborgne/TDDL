import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm

plt.style.use('_mpl-gallery')

def plot_region(ax,X,Y,Z,title='',with_line=False):
    if with_line:
        ax.plot_surface(X, Y, Z, cmap='cool', edgecolor='royalblue', lw=0.5, rstride=2, cstride=2, alpha=0.8)
    else:
        ax.plot_surface(X, Y, Z,  cmap='cool')
    if title != '':
        ax.set_title(title)
    # ax.set(xticklabels=[],yticklabels=[])
    ax.set_xlabel('x')
    ax.set_ylabel('y')


def get_relu(X,Y,ax,ay,b):
    Z = ax * X + ay * Y +b
    return (Z>0)*Z

# data
X1 = np.arange(-5, 5, 0.1)
X2 = np.arange(-5, 5, 0.1)
X1, X2 = np.meshgrid(X1, X2)

# without hidden layer
a01 = 1
a02 = -2
b0 = 0.25
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# plot_region(ax,X1,X2,a01*X1+a02*X2+b0,'output = linear', with_line=True)

# with one hidden layer (with ReLU)
a111 = 1
a112 = 0
b11 = 0

a121 = 0
a122 = 1
b12 = 0

# a131 = 0.5
# a132 = 2
# b13 = 2

# a141 = 1
# a142 = 0
# b14 = 0.2

Z1 = get_relu(X1,X2,a111,a112,b11)
Z2 = get_relu(X1,X2,a121,a122,b12)
# Z3 = get_relu(X1,X2,a131,a132,b13)
# Z4 = get_relu(X1,X2,a141,a142,b14)

fig, ax = plt.subplots(1,3,subplot_kw={"projection": "3d"})
plot_region(ax[0],X1,X2,Z1,'Z1')
plot_region(ax[1],X1,X2,Z2, 'Z2')
plot_region(ax[2],X1,X2,Z1+Z2,'output with ReLU hidden layer', with_line=True)

# with two hidden layers (with ReLU)

# 9 régions
a211 = 2
a212 = 2
b21 = -1.5
a221 = 1
a222 = -1
b22 = 3

# 


W1 = get_relu(Z1,Z2,a211,a212,b21) 
W2 = get_relu(Z1,Z2,a221,a222,b22)


fig, ax = plt.subplots(1,3,subplot_kw={"projection": "3d"})
plot_region(ax[0],X1,X2,W1,'W1')
plot_region(ax[1],X1,X2,W2, 'W2')
plot_region(ax[2],X1,X2,W1+W2,'output with 2 ReLU hidden layer', with_line=True)

plt.show()

