import os
import numpy as np 
import scipy.optimize    as sco
#import scipy.signal      as scsi
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
#from kinetics import run_system
from functions import *
import scipy.interpolate as sci
import scipy.signal as scs
import os
import itertools

fontsize = 8.5
plt.style.use(['science','no-latex'])
import matplotlib
matplotlib.rc('xtick', labelsize=fontsize)
matplotlib.rc('ytick', labelsize=fontsize)
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
root = "figures/"

import colormaps as cmaps
cm = cmaps.WhiteYellowOrangeRed
cm = cm.cut(0.25, 'left')
cm = cm.cut(0.35, 'right')


k_bT  = 1

xi_ab = 0
xi_ac = 2.01
xi_bc = 3
delta = 4

xi    = np.array(([0, xi_ab, xi_ac],
                  [xi_ab, 0, xi_bc],
                  [xi_ac, xi_bc, 0]))
nu = np.ones(3)
omega = np.array([-2.5, 0.1, -2.25])

N         = 400
phi_delta = 1/N
phi_min, phi_max, n_points = phi_delta, 1.- phi_delta , N
phi = np.linspace(phi_min, phi_max, N)

phi         = np.linspace(1e-3, 1-1e-3, 500)

tiline_two_component = get_tieline_twocomp(phi, xi_ac, nu[1], nu[2], k_bT, omega[1], omega[2])
bino                 = calculate_binodal(tiline_two_component, np.array([0.001, 0.5]), 5000, k_bT, nu, xi_ab, xi_ac, xi_bc, omega)

intersect_tot = np.sum(bino[0, :])/2
thresh = intersect_tot
kappa = 5*1e-4
k     = 1 
k1, k2 = k, k

u, c        = np.unique(bino[:, 0], return_index=True)
bot_func    = sci.interp1d(bino[c, 0], bino[c, 1], kind="quadratic")

P, bot, top = fill_between_arrays_type2(bino)
plt.clf()

chem_pot_diff = np.zeros(len(bino[:, 0]))
for i in range(len(chem_pot_diff)):
    temp  = calc_gibbs_mu(bino[i, :2], k_bT, nu, omega, xi)
    chem_pot_diff[i] = temp[0]-temp[1]

indx = np.argmin(np.abs(chem_pot_diff))
phi_a0        = bino[indx, 0]
phi_b0        = bino[indx, 1]

delta_x = bino[indx+1, 0] - bino[indx, 0]
delta_y = bino[indx+1, 1] - bino[indx, 1]
beta    = np.arctan2(delta_y, delta_x)


delta_psi = -np.linspace(1e-7, 2*1e-1, int(1e3))
den = (np.sin(beta) + np.cos(beta))

theta_const = ( np.cos(beta)/phi_a0
                - np.sin(beta)/phi_b0
                + xi_ab*(np.sin(beta) - np.cos(beta))
                - (xi_ac - xi_bc)*(np.cos(beta) + np.sin(beta)) ) / den

psiI  = bino[indx, 0] + bino[indx, 1]
psiII = bino[indx, 2] + bino[indx, 3]

Theta_const = theta_const*(psiII - psiI)
ana_retry = 1 + Theta_const - Theta_const*Theta_const*delta_psi/(psiII - psiI)


### PSI VS DELTA 
k_bT  = 1

psi = 0.36
nu = np.ones(3)
omega = np.array([-2.5, 0.1, -2.25])

N         = 250
phi_delta = 1/N
phi_min, phi_max, n_points = phi_delta, 1.- phi_delta , N
phi = np.linspace(phi_min, phi_max, N)

kappa = 5*1e-4
thresh = 0.5 
delta = 4
psis   = np.arange(0.15, 0.41, 0.01)
deltas = np.arange(0,    5.01, 0.05)
k1, k2 = 1, 1
counter = 0
types_psi = np.zeros((len(psis), len(deltas)))

xi_ab = 0
xi_ac = 2.01
xi_bc = 3.00

phi_range = np.load("data/2c_phirange_xiab%3.2f_xiac%3.2f_xibc%3.2f.npy" % (xi_ab, xi_ac, xi_bc))
deltas    = np.load("data/2c_deltasss_xiab%3.2f_xiac%3.2f_xibc%3.2f.npy" % (xi_ab, xi_ac, xi_bc))

indx = np.where(phi_range[:, 0] > 1e-3)[0]
numerical_crit = deltas[indx[-1]]
phi_range, deltas_sub = phi_range[indx], deltas[indx]
upper_homo = np.zeros(len(deltas))
upper_homo[indx[-1]:] = phi_range[0, 1]
upper_homo[indx] = phi_range[:, 0]


fig, axes = plt.subplots(nrows=2, ncols=1)
fig.subplots_adjust(wspace=100, hspace=100)
alpha_val = 0.2

axes[1].fill_between(deltas, np.ones(len(deltas))*phi_range[0, 1]/2, np.ones(len(deltas)), color="C1", alpha=alpha_val)
axes[1].fill_between(deltas_sub, phi_range[:, 0]/2, phi_range[:, 1]/2, color="C2", alpha=alpha_val)
axes[1].fill_between(deltas, upper_homo/2, np.zeros(len(deltas)), color="C0", alpha=alpha_val)

axes[1].plot(deltas_sub, phi_range[:, 0]/2, "k")
axes[1].text(0.50, 0.135, r"Homogeneous", fontsize=10, color="C0")
axes[1].text(1.6, 0.362/2, r"Active droplet", fontsize=10, color="C1")
axes[1].text(3.8, 0.30/2, r"Bistable", fontsize=10, color="C2")
axes[1].set_ylim(0.11, 0.40/2)
axes[1].set_xlim(0, np.amax(deltas))


axes[1].set_ylabel(r"Tot. vol. frac. $\bar{\psi}$", fontsize=9, labelpad=0)
axes[1].set_xlabel(r"Fuel strength $\delta/k_BT$", fontsize=9, labelpad=-1)
axes[1].plot(np.log(ana_retry), phi_range[0, 1]/2+delta_psi/2, "--" , color="k")
axes[1].text(3.65, 0.114, r"Eq. (23)", fontsize=9, color="k", rotation=-26.5)
axes[1].text(2.55, 0.115, r"Eq. (11)", fontsize=9, color="k", rotation=-48.5)



NESS = np.zeros((len(bino[:, 0]), 2))
NESSbino = np.zeros((len(bino[:, 0]), 5))

for i in range(len(bino[:, 0])):
    delta1 = fuel_foo(bino[i, 0], bino[i, 1], thresh, kappa, delta)
    delta2 = fuel_foo(bino[i, 2], bino[i, 3], thresh, kappa, delta)
    mus = calc_gibbs_mu(np.array([bino[i, 0], bino[i, 1]]), k_bT, nu, omega, xi)
    mua, mub = mus[0], mus[1]
    
    VI = -k2*(np.exp(delta2) - np.exp(mub-mua))/(k1*np.exp(delta1) - k2*np.exp(delta2) - (k1-k2)*np.exp(mub - mua))

    if VI > -1e-5 and VI < 1+1e-5:
        NESS[i, 0] = VI*bino[i, 0] + (1-VI)*bino[i, 2]
        NESS[i, 1] = VI*bino[i, 1] + (1-VI)*bino[i, 3]
        NESSbino[i, :-1] = bino[i, :]
        NESSbino[i,  -1] = 1-VI


indx = np.where(NESS[:, 0] > 0)[0][0]
delta_V = NESSbino[indx,  -1] - NESSbino[indx+1,  -1]
delta_psi  = NESS[indx,  0] + NESS[indx,  1] - (NESS[indx+1,  0]+NESS[indx+1,  1])
delta_psiI = NESSbino[indx,  0] + NESSbino[indx,  1] - (NESSbino[indx+1,  0] + NESSbino[indx+1,  1])

indx = np.where(NESS[:, 0] != 0)[0]
NESS = NESS[indx]
NESSbino = NESSbino[indx, :]





xmin = 0.15
xmax = 0.85


Psi1 = NESSbino[:, 0] + NESSbino[:, 1]
Psi2 = NESSbino[:, 2] + NESSbino[:, 3]
TLSC = NESS[:, 0]+NESS[:, 1]

x = np.concatenate((Psi1[::-1], Psi2))
y = np.concatenate((NESSbino[::-1, -1], NESSbino[:, -1]))


axes[0].fill_between(x/2, np.ones(len(x)), y, color="gray", alpha=0.06)
axes[0].plot(Psi1/2, NESSbino[:, -1], linewidth=1.0, color="k")
axes[0].plot(Psi2/2, NESSbino[:, -1], linewidth=1.0, color="k")

shift = int(0.1*len(TLSC))
indx_min = np.argmin(TLSC)
indx_jmp = np.argmin(np.abs(TLSC[shift:] - Psi1[0]))+shift

axes[0].plot([Psi1[0]/2, Psi1[0]/2], [0.0, NESSbino[indx_jmp, -1]], ":", color="gray", alpha=1, linewidth=0.6)
axes[0].plot([TLSC[indx_min]/2, TLSC[indx_min]/2], [0.0, NESSbino[indx_min, -1]], ":", color="gray", alpha=1, linewidth=0.6)

axes[0].plot(TLSC[:indx_min]/2, NESSbino[:indx_min, -1], "--", color=cm(0.5), linewidth=1.5)
axes[0].plot(TLSC[indx_min:indx_jmp]/2, NESSbino[indx_min:indx_jmp, -1], "-", color=cm(0.5), linewidth=1.5)
axes[0].plot(TLSC[indx_jmp:]/2, NESSbino[indx_jmp:, -1], "-", color=cm(0.5), linewidth=1.5)
axes[0].plot([0, Psi1[0]/2], [0, 0], cm(0.5), linewidth=1.5)
axes[0].set_yticks([0, 0.10, 0.20])

axes[0].set_ylim(-0.02, 0.3)
axes[0].set_xlim(0.20/2, 0.37/2)

axes[0].text(0.21/2, 0.084, "Binodal", rotation=-15, color="k")
axes[0].text(0.202/2, 0.005, "Homog. nullcline", rotation=0.0, color="C0")

axes[0].plot(np.array([0, Psi1[0]])/2, [0, 0], "-",  color="C0")
axes[0].plot(np.array([Psi1[0], 1])/2, [0, 0], "--", color="C0")

axes[0].set_xticks([0.2/2, 0.25/2, 0.30/2, 0.35/2])
axes[0].set_ylabel(r"Drop. vol. $V\,\,/V$", fontsize=9, labelpad=0)
axes[0].set_xlabel(r"Total volume fraction $\bar{\psi}$", fontsize=9, labelpad=-1)

fig.subplots_adjust(hspace=0.35)
filename  = "figures/state_diagrams.pdf"
plt.savefig(filename, bbox_inches="tight", dpi=500, transparent=True)
os.system('pdfcrop %s %s &> /dev/null &'%(filename, filename))
plt.show()




