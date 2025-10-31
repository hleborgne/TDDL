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
    ax.set(xticklabels=[],yticklabels=[])
    ax.set_xlabel('x')
    ax.set_ylabel('y')


def get_relu(X,Y,ax,ay,b):
    Z = ax * X + ay * Y +b
    return (Z>0)*Z

# data
X1 = np.arange(-5, 5, 0.25)
X2 = np.arange(-5, 5, 0.25)
X1, X2 = np.meshgrid(X1, X2)

# without hidden layer
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
a01 = 1
a02 = -2
b0 = 0.25
plot_region(ax,X1,X2,a01*X1+a02*X2+b0,'output = linear', with_line=True)

# with one hidden layer (with ReLU)
a11 = 2
a12 = -1
b1 = 0.5

a21 = -0.5
a22 = -2
b2 = -0.5

Z1 = get_relu(X1,X2,a11,a12,b1)
Z2 = get_relu(X1,X2,a21,a22,b2)

fig, ax = plt.subplots(1,3,subplot_kw={"projection": "3d"})
plot_region(ax[0],X1,X2,Z1,'Z1')
plot_region(ax[1],X1,X2,Z2, 'Z2')
plot_region(ax[2],X1,X2,Z1+Z2,'output with ReLU hidden layer', with_line=True)

# with two hidden layers (with ReLU)
a31 = -0.3
a32 = 0.3
b3 = -0.5

a41 = 0.5
a42 = -3
b4 = 1

W1 = get_relu(Z1,Z2,a31,a32,b3) 
W2 = get_relu(Z1,Z2,a41,a42,b4) 

fig, ax = plt.subplots(1,3,subplot_kw={"projection": "3d"})
plot_region(ax[0],X1,X2,W1,'W1')
plot_region(ax[1],X1,X2,W2, 'W2')
plot_region(ax[2],X1,X2,W1+W2,'output with 2 ReLU hidden layer', with_line=True)

plt.show()

