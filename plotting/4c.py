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

import colormaps as cmaps
cm = cmaps.WhiteYellowOrangeRed
cm = cm.cut(0.25, 'left')

deltas_in_hyst = np.arange(2.5, 4.01, 0.05)

fontsize = 9
plt.style.use(['science','no-latex'])#, "grid"])
import matplotlib
matplotlib.rc('xtick', labelsize=fontsize)
matplotlib.rc('ytick', labelsize=fontsize)
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
root = "figures/"
k_bT  = 1

def get_curve_length(x, y):
    """
    Calculate the length of a 2D curve given by points (x, y).
    
    Parameters:
    - x: numpy array of shape (N,), x-coordinates of the curve.
    - y: numpy array of shape (N,), y-coordinates of the curve.
    
    Returns:
    - The length of the curve.
    """
    # Compute differences between consecutive points
    dx = np.diff(x)
    dy = np.diff(y)
    
    # Calculate the Euclidean distance between consecutive points
    distances = np.sqrt(dx**2 + dy**2)
    
    # Sum all distances to get the total length
    total_length = np.sum(distances)
    
    return total_length

xi_ab = 0
xi_ac = 2.01
xi_bc = 3

nu    = np.ones(3)
omega = np.array([-2.5, 0.1, -2.25])
xi    = np.array(([0, xi_ab, xi_ac],
                  [xi_ab, 0, xi_bc],
                  [xi_ac, xi_bc, 0]))

phi         = np.linspace(1e-3, 1-1e-3, 1000)

tiline_two_component = get_tieline_twocomp(phi, xi_ac, nu[1], nu[2], k_bT, omega[1], omega[2])
bino                 = calculate_binodal(tiline_two_component, np.array([0.001, 0.5]), 5000, k_bT, nu, xi_ab, xi_ac, xi_bc, omega)
P, bot, top = fill_between_arrays_type2(bino)


deltas = np.arange(2.0, 4.01, 0.5)[-3:]
k_ps   = np.logspace(-2, 2, 17)

thresh = 0.5
kappa = 5*1e-4

N = int(5*1e5)
k_c = 1

periods      = np.zeros((len(deltas), len(k_ps)))
delta_psis   = np.zeros(np.shape(periods))
delta_phis   = np.zeros(np.shape(periods))
curve_length = np.zeros(np.shape(periods))


fig, ax = plt.subplots() #figsize=(14, 2))

left, bottom, width, height = [0.53, 0.20, 0.35, 0.27]
ax0 = fig.add_axes([left, bottom, width, height])

current_min_A = 1
current_max_A = 0
current_min_B = 1
current_max_B = 0

inset_indx = 8
for i, delta in enumerate(deltas):
    color = cm( (delta - deltas[0])/(deltas[-1] - deltas[0]) )
    for j, k_p in enumerate(k_ps):
        data = np.transpose(np.load("data/find_freq_xiab%3.2f_xiac%3.2f_xibc%3.2f_kp%2.3f_delta%3.2f.npy" % (xi_ab, xi_ac, xi_bc, k_p, delta)))

        phi_bar = data[:,  :2]
        phiI    = data[:, 2:4]
        phiII   = data[:, 4:6]
        VI      = data[:,  -2]
        VII     = 1-VI
        T       = data[:, -1]

        periods[i, j], orbit_bool = get_period(T, phi_bar[:, 0],  phi_bar[:, 1], 1e-4)
        indx = np.argmin(np.abs(T-(T[-1]-periods[i, j])))
        delta_psis[i, j] = 100*(np.amax(phi_bar[indx:, 0]+phi_bar[indx:, 1]) - np.amin(phi_bar[indx:, 0]+phi_bar[indx:, 1]))
        delta_phis[i, j] = np.amax(phi_bar[indx:, 0]-phi_bar[indx:, 1]) - np.amin(phi_bar[indx:, 0]-phi_bar[indx:, 1])
        curve_length[i, j] = get_curve_length(phi_bar[indx:, 0], phi_bar[indx:, 1])
        
        if j == inset_indx:
            print(delta, k_p)
            hor_line = k_p
            ax0.plot(phi_bar[indx:, 0], phi_bar[indx:, 1], color=color, linewidth=1.5)
            current_max_A = np.amax([current_max_A, np.amax(phi_bar[indx:, 0])])
            current_min_A = np.amin([current_min_A, np.amin(phi_bar[indx:, 0])])
            current_max_B = np.amax([current_max_B, np.amax(phi_bar[indx:, 1])])
            current_min_B = np.amin([current_min_B, np.amin(phi_bar[indx:, 1])])

            if periods[i, j] < 1e-4:
                ax0.plot(phi_bar[-1, 0], phi_bar[-1, 1], "o", color=color)


ax0.fill_between(P, bot, top, color="gray", alpha=0.06)
ax0.plot(bino[:, 0], bino[:, 1], "k")
ax0.plot(bino[:, 2], bino[:, 3], "k")

ax0.text(0.27, 0.10, r"$k_2/k_1=10^{%1.0f}$" % np.log10(k_ps[inset_indx]), fontsize=9, color="k")

ax0.set_xlim(current_min_A-0.002, current_max_A+0.002)
ax0.set_ylim(current_min_B-0.002, current_max_B+0.002)

ax0.set_xlabel(r"$\bar{\phi}_A$", fontsize=10, labelpad=-7)
ax0.set_ylabel(r"$\bar{\phi}_B$", fontsize=10, labelpad=-7)
ax0.set_xticks([0.2,  0.30])
ax0.set_yticks([0.03, 0.13])
ax0.tick_params(axis='both', which='major', labelsize=8)

colors = []
for i, delta in enumerate(deltas):
    color = cm( (delta - deltas[0])/(deltas[-1] - deltas[0]) )
    colors.append(color)
    indx = np.where(periods[i, :] != 0)[0]
    ax.plot(k_ps[indx], 2*np.pi/periods[i, indx], label=r"$%1.1f$" % deltas[i], color=color)

x0 = 0.0316
x1 = x0/3.16
y0 = 0.1/3.16
y1 = y0/3.16
ax.fill_between([x0, x1], [y0, y1], [y0, y0], color="k", alpha=0.1)
ax.plot([x0, x1], [y0, y0], "k:")
ax.plot([x1, x1], [y0, y1], "k:")
ax.plot([x0, x1], [y0, y1], "k:")
ax.text(x1*3.16/2,  y0*1.1, r"$1$")
ax.text(x1*1.07, y1*3.16/2, r"$1$")

legend = ax.legend(loc="upper left", frameon=True, fontsize=9, ncol=3, title=r"Fuel strength $\delta/k_BT$", columnspacing=0.6)
frame = legend.get_frame()
frame.set_edgecolor('black')
frame.set_linewidth(0.35)

ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlabel(r"Rate coefficeint $k_2/k_1$", fontsize=11)
ax.set_ylabel(r"Cycle frequency $\Omega/k_1$", fontsize=10)

filename = root + "freq_vs_kp_delta.pdf"
fig.savefig(filename, bbox_inches="tight", dpi=500)
os.system('pdfcrop %s %s &> /dev/null &'%(filename, filename))
plt.show()







