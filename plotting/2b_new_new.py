import os
import numpy as np 
import scipy.spatial     as scs
import scipy.optimize    as sco
#import scipy.signal      as scsi
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
#from kinetics import run_system
from functions import *
import scipy.interpolate as sci
import os
import itertools

fontsize = 9
plt.style.use(['science','no-latex'])#, "grid"])
import matplotlib
matplotlib.rc('xtick', labelsize=fontsize)
matplotlib.rc('ytick', labelsize=fontsize)
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
root = "figures/"

import colormaps as cmaps
cm = cmaps.WhiteYellowOrangeRed
cm = cm.cut(0.25, 'left')

from scipy.signal import savgol_filter

def find_zero_crossings(array):
    """
    Find the indices of zero crossings in a 1D array.

    Parameters:
        array (list or numpy.ndarray): The input array.

    Returns:
        list: Indices of zero crossings.
    """
    array = np.asarray(array)
    zero_crossings = []

    # Check for sign changes
    for i in range(len(array) - 1):
        if (array[i] > 0 and array[i + 1] < 0) or (array[i] < 0 and array[i + 1] > 0):
            # Approximate the zero crossing point
            zero_crossings.append(i)
        elif array[i] == 0:
            # Handle exact zero values
            zero_crossings.append(i)
    
    return np.array(zero_crossings)

psi = 0.32 
N   = 4000
x   = np.linspace(0, psi, N)
y   = psi - x 

fig, axes = plt.subplots(nrows=2, ncols=1, height_ratios=[1, 1])
plt.subplots_adjust(wspace=0, hspace=0.42)
axes[0].plot([-2, 2], [0, 0], "k")

plot_deltas = np.array([2, 2.5, 3.0, 3.5])[::-1]
#deltas      = np.arange(2, 4, 0.01)
#all_rates   = np.load("data/all_rates.npy")
#all_rates_noF   = np.load("data/all_rates_nofuel.npy")

all_rates = np.load("data/all_rates.npy")#[::-1]
deltas = np.load("data/all_rates_deltas.npy")#[::-1, :]
phi_B = np.load("data/all_rates_phiB.npy")
all_volumes = np.load("data/all_volumes.npy")
bino = np.load("data/bino_2b.npy")

#indx = np.argmin(np.abs(deltas-3.15))
#deltas = deltas[indx:indx+50]
#all_rates = all_rates[indx:indx+50, :]

Vs  = np.linspace(0, 1, int(1e3))
all_tielines = np.zeros((len(bino[:, 0]), len(Vs), 2))

for i in range(len(bino[:, 0])):
    all_tielines[i, :, 0] = Vs*bino[i, 0] + (1-Vs)*bino[i, 2]
    all_tielines[i, :, 1] = Vs*bino[i, 1] + (1-Vs)*bino[i, 3]

zeros_V   = np.zeros((len(deltas), 3))
zeros_phi = np.zeros((len(deltas), 3, 2))

try:
    zeros_V = np.load("data/zeros_V.npy")
    #a = wpp
except:
    for i in range(len(deltas)):
        #i = len(deltas) - i - 1
        indexes = find_zero_crossings(all_rates[i, :])

        if len(indexes) not in [1, 3]:
            print("revised pre", indexes)
            indexes = indexes[np.argsort(np.abs(np.diff(indexes)))[-3:][::-1] + 1]
            print("revised post", indexes)
        print("length", len(indexes), indexes)

        #plt.plot(x, all_rates[i, :])
        #plt.plot(x[indexes], all_rates[i, indexes], "ko")
        #plt.show()
        for j in range(len(indexes)):
            zeros_phi[i, j, 0], zeros_phi[i, j, 1] = x[indexes[j]], y[indexes[j]]

            if check_inside_type2ab(zeros_phi[i, j, 0], zeros_phi[i, j, 1], bino):
                current_min = 1
                for k in range(len(bino[:, 0])):
                    distance = np.amin(np.square( all_tielines[k, :, 0] - zeros_phi[i, j, 0]  ) + np.square( all_tielines[k, :, 1] - zeros_phi[i, j, 1]) )
                    if distance < current_min:
                        current_min = distance
                        indx = k
                zeros_V[i, j] = 1 - ( psi - (bino[indx, 2] + bino[indx, 3]) )/(bino[indx, 0] + bino[indx, 1] - bino[indx, 2] - bino[indx, 3]) 
                #plt.plot(zeros_phi[i, j, 0], zeros_phi[i, j, 1], "o")
                #plt.plot(all_tielines[indx, :, 0], all_tielines[indx, :, 1])
                #plt.plot(bino[:, 0], bino[:, 1])
                #plt.plot(bino[:, 2], bino[:, 3])
                #plt.show()
                #print(zeros_V[i, :])
            else:
                zeros_V[i, j] = 0
        """
        plt.plot(bino[:, 0], bino[:, 1])
        plt.plot(bino[:, 2], bino[:, 3])
        plt.plot(x, y)
        plt.plot(zeros_phi[i, :, 0], zeros_phi[i, :, 1], "ko")
        plt.xlim(0, 0.8)
        plt.ylim(0, 0.8)
        plt.show()
        """
    np.save("data/zeros_V.npy", zeros_V)

#plt.plot(zeros_V[:, 0])
#plt.plot(zeros_V[:, 1])
#plt.plot(zeros_V[:, 2])
#plt.show()

for i in range(len(plot_deltas)):
    indx = np.argmin(np.abs(deltas - plot_deltas[i]))
    color_indx = (plot_deltas[i] - np.amin(plot_deltas))/(np.amax(plot_deltas) - np.amin(deltas[:-1]))
    y_data = savgol_filter( all_rates[indx, :], 25, 3)
    #print(all_rates[0, 25]-all_rates[indx, 25])
    #axes[0].plot((y-x)/psi, 100*all_rates[indx, :], color=cm(color_indx))
    axes[0].plot((y-x)/psi, 100*y_data, color=cm(color_indx))

axes[0].plot((y-x)/psi, 100*all_rates[-1, :], color="C0")

deltas = deltas[1:]

axes[0].fill_between([-0.8752, 1], [-10, -10], [10, 10], color="gray", alpha=0.06)
axes[0].plot(-0.927, 0, "o", color="C0")
color_indx = (plot_deltas[0] - np.amin(deltas[1:]))/(np.amax(deltas) - np.amin(deltas[:-1]))
axes[0].plot(-0.553,   0, "o", color=cm(color_indx))
axes[0].plot(0.01170/psi, 0, "o", color="C0")
axes[0].plot(-0.84375,  0, "o", markerfacecolor="w",  markeredgewidth=1.2, markeredgecolor="black")

axes[0].set_xlim(-1, -0.426)
axes[0].set_ylim(-6, 6)
axes[0].set_xlabel(r"Conservation line $\bar{\xi}/\bar{\psi}$", fontsize=10, labelpad=0, color="dimgrey")
axes[0].set_ylabel(r"Reaction rate $\bar{r}/k_1$", fontsize=10, labelpad=3.5)
#axes[0].set_xticks([0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
axes[0].set_yticks([-6.0, -3.0, 0.0, 3.0, 6.0])

"""
print(np.shape(all_volumes))

plot_x = np.linspace(1e-5, psi-1e-5, int(1e5))
branches = np.zeros((len(deltas), 3))
worked_deltas = []
for i in range(len(deltas)):
    #color_indx = (deltas[i] - np.amin(deltas))/(np.amax(deltas) - np.amin(deltas))
    f = all_rates[i, :]
    foo = sci.interp1d(x, f)
    VI_foo = sci.interp1d(x, all_volumes[i, :])
    f = foo(plot_x)
    indxes = np.argwhere(np.abs(f) < 4*1e-5)[:, 0]
    deriv = indxes[1:]-indxes[:-1]
    jumps = np.argsort(deriv)[::-1][:2]
    #print("\n\n", indxes)
    if len(indxes) > 2 and np.amax(deriv[jumps]) > 5:
        #print("jumps", jumps)
        #print("here", jumps, indxes)
        indx0 = int(np.mean(indxes[0:np.amin(jumps)+1]))
        indx1 = int(np.mean(indxes[np.amin(jumps)+1:np.amax(jumps)+1]))
        indx2 = int(np.mean(indxes[np.amax(jumps):])+1)
        #print(indx0, indx1, indx2)
        branches[i, 0] = 1-VI_foo(plot_x[indx0])
        branches[i, 1] = 1-VI_foo(plot_x[indx1])
        branches[i, 2] = 1-VI_foo(plot_x[indx2])
        worked_deltas.append(i)

        
        if np.all(deriv[jumps] > 100):
            worked_deltas.append(i)
            prev_indx = indxes[0]

            for j in range(len(jumps)):
                indx = int(np.mean([prev_indx, indxes[jumps[j]]]))
                print(prev_indx, indxes[jumps[j]], indx)
                branches[i, j] = (psi-plot_x[indx])/psi
                prev_indx = indxes[jumps[j]+1]
        


branches[:, 2] = 0#0.01170/psi
"""
deltas = np.load("data/all_rates_deltas.npy")#[::-1]


def plot_color(x, y, full_d, linetype):
    for i in range(len(x)-1):
        c_indx = (x[i]-2)/(3.5-np.amin(full_d))
        axes[1].plot([x[i], x[i+1]], [y[i], y[i+1]], linetype, color=cm(c_indx), linewidth=1.5)

new_zeros_V = np.zeros(np.shape(zeros_V))
new_zeros_V[:, 0] = np.amax(zeros_V, axis=1)
new_zeros_V[:, 2] = np.amin(zeros_V, axis=1)
# Sort each row along axis=1 in descending order
zeros_V = np.sort(zeros_V, axis=1)[:, ::-1]
#
# Extract the second largest element from each row
#second_largest = sorted_array[:, 1]
#zeros_V = np.copy(new_zeros_V)

indx = np.where(np.amax(zeros_V[:, :], axis=1) != 0)[0]
jmp = np.where(np.amax(zeros_V[:, :], axis=1) > 1e-4)[0][-1]
#indx = worked_deltas
#print(worked_deltas)
#jmp = worked_deltas[0]

#branches[worked_deltas, 1] = savgol_filter(branches[worked_deltas, 1], 10, 3)

plt.plot([deltas[jmp], deltas[jmp]], [zeros_V[jmp, 2], zeros_V[jmp, 0]], ":", color="gray")
#plot_color(deltas, branches[:, 2], deltas)
axes[1].plot(deltas, zeros_V[:, 2], color="C0")

plot_color(deltas[indx], zeros_V[indx, 0], deltas[1:], "-")
plot_color(deltas[indx], zeros_V[indx, 1], deltas[1:], "-")
#plt.plot(deltas[indx], zeros_V[indx, 0], "-")
#plt.plot(deltas[indx], zeros_V[indx, 1], "-")
#plot_color(deltas[:], zeros_V[:, 2], deltas[1:], "-")
axes[1].plot(deltas[indx], zeros_V[indx, 1], "--", color="white", linewidth=1.55, dashes=(2.5, 5))
c_indx = (deltas[jmp]-np.amin(deltas))/(np.amax(deltas)-np.amin(deltas))
axes[1].plot([deltas[jmp], deltas[jmp]], [zeros_V[jmp, 0], zeros_V[jmp, 1]], "-", color=cm(c_indx), linewidth=1.5)

cindx = (3.4-np.amin(deltas))/(np.amax(deltas)-np.amin(deltas))
#axes[1].text(3.4, 0.21, "Homogeneous fueled", color=cm(0.99), rotation=20)
#axes[1].text(3.4, 0.075,  "Unstable fueled", color="k", rotation=-1.8)
axes[1].text(3.4, 0.091, "Phase-seperated", color=cm(0.99))
axes[1].text(2.675, 0.012,  "Homogeneous", color="C0")

#axes[1].set_ylim(-0.3, 0.02)
axes[1].set_xlim(2.65, 4.00)
#axes[1].set_xlim(3.20, 3.40)

axes[1].set_xlabel(r"Fuel strength $\delta/k_BT$", fontsize=10.5, labelpad=-0.5)
axes[1].set_ylabel(r"Drop. vol. $V^{\text{II}}/V$", fontsize=10, labelpad=-0.25)
#axes[1].set_xticks([2.8, 3.15, 3.5, 3.85])
axes[1].set_yticks([0.0, 0.1, 0.20])

import matplotlib as mpl
norm = mpl.colors.Normalize(vmin=plot_deltas.min(), vmax=plot_deltas.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
cbar = fig.colorbar(cmap, ticks=[2.0, 2.5, 3.0, 3.5], pad=0.0, ax=axes.ravel().tolist())
cbar.ax.set_ylabel(r'Fuel strength $\delta/k_BT$', rotation=90, fontsize=10, labelpad=1.5)


filename = root + "2b.pdf"
fig.savefig(filename, bbox_inches="tight", dpi=500)
os.system('pdfcrop %s %s &> /dev/null &'%(filename, filename))
plt.show()

from scipy.signal import savgol_filter
y = branches[worked_deltas, 0]
yhat = savgol_filter(y, 10, 3) # window size 51, polynomial order 3

plt.plot(y)
plt.plot(yhat, "k")
plt.show()


