import os
import numpy as np
import matplotlib.pyplot as plt
from functions import *

fontsize = 10
plt.style.use(['science','no-latex'])
import matplotlib
matplotlib.rc('xtick', labelsize=fontsize)
matplotlib.rc('ytick', labelsize=fontsize)
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
root = "figures/"

import colormaps as cmaps
cmapp = cmaps.WhiteYellowOrangeRed
cmapp = cmapp.cut(0.25, 'left')

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

k_p   = 10
k_c   = 1
Nt           = int(5*1e4)
counter = 1

phi         = np.linspace(1e-3, 1-1e-3, 1000)

tiline_two_component = get_tieline_twocomp(phi, xi_ac, nu[1], nu[2], k_bT, omega[1], omega[2])
bino                 = calculate_binodal(tiline_two_component, np.array([0.001, 0.5]), 5000, k_bT, nu, xi_ab, xi_ac, xi_bc, omega)


fac = 1
intersect_tot = np.sum(bino[0, :])/2
thresh = intersect_tot*fac
kappa = 5*1e-4
k = 1 
k1, k2 = k, k

P, bot, top = fill_between_arrays_type2(bino)

chem_eq, cons_eq = contour_lines_ab_psi(phi, nu, k_bT, omega, xi, bot, top, bino)
indx = bino_intersect_chemindx(chem_eq, bino)
chemeq1 = chem_eq[indx:, :]
plt.clf()


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

indx1 = bino_intersect_chemindx(chem_eq, bino[:, :2])
indx2 = bino_intersect_chemindx(chem_eq, bino[:, 2:])
eq_post = chem_eq[:indx2, :]
eq_in = chem_eq[indx2:indx1, :]

indx = np.where(NESS[:, 0] != 0)[0]
NESS = NESS[indx]
NESSbino = NESSbino[indx, :]

N = int(5*1e3)
T = np.linspace(0, 50, N)

dt = T[1]-T[0]
phi_bar = np.zeros((N, 2))
phiII = np.zeros((N,   3))
phiI = np.zeros((N,    3))
V, VI, VII = 1, np.zeros(N), np.zeros(N)

phi_bar[0, :] = np.array([0.2, 0.25])

k_c = 1
k_p = 1

phi_bar, phiI, phiII, VI, VII, F, chems, rates, T = run_system(T, phi_bar, phiI, phiII, V, VI, VII, k_bT, xi_ab, xi_ac, xi_bc, xi, nu, omega, k_p, k_c, bino, P, bot, top, thresh, kappa, delta)

tau, orbit = get_period(T, phi_bar[:, 0], phi_bar[:, 1], 1e-5)
indx       = np.argmin(np.abs(T - (T[-1]-tau)))
phi_bar    = phi_bar[indx:, :]
VI         = VI[indx:]
T          = T[indx:] - T[indx]


PS_indx = np.where(VI > 1e-6)[0]
mid_PS = int((PS_indx[0]+PS_indx[-1])/2)
roll_len = -int(mid_PS - int(len(T)/2))
PS_indx = np.array(PS_indx) + roll_len
VI = np.roll(VI, roll_len)
phi_bar  = np.roll(phi_bar, roll_len, axis=0)

norm = plt.Normalize(T.min(), T.max())
from matplotlib import cm, colors
cmap = cm.twilight_shifted
points = np.array([phi_bar[:, 0], phi_bar[:, 1]]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Create the LineCollection
from matplotlib.collections import LineCollection
lc = LineCollection(segments, cmap=cmap, norm=norm)
lc.set_array(T)
lc.set_linewidth(2.5)

half = np.argmin(np.sum(cons_eq, axis=1))
indx1 = bino_intersect_chemindx(cons_eq[half:, :], bino[:, :2]) + half
indx2 = bino_intersect_chemindx(cons_eq[half:, :], bino[:, 2:]) + half
indx3 = bino_intersect_chemindx(cons_eq[:half, :], bino[:, :2])
indx4 = bino_intersect_chemindx(cons_eq[:half, :], bino[:, 2:])



fig, axes = plt.subplots(nrows=2, ncols=1, height_ratios=[1.5, 1])
plt.subplots_adjust(wspace=0, hspace=0.40)

line = axes[0].add_collection(lc)

lw = 1.3
axes[0].fill_between(P, bot, top, color="gray", alpha=0.06)
axes[0].plot(bino[:, 0], bino[:, 1], color="k", linewidth=0.8)
axes[0].plot(bino[:, 2], bino[:, 3], color="k", linewidth=0.8)

Ntl = 5
bino_len = len(bino[:, 0])
points = [(0.38, 0.007), (0.14, 0.046)]

for i in range(len(points)):
    tli = np.argmin( np.square( bino[:, 0] - points[i][0] ) + np.square( bino[:, 1] - points[i][1] ) )
    axes[0].plot([bino[tli, 0], bino[tli, 2]], [bino[tli, 1], bino[tli, 3]], color="k", alpha=0.3, linewidth=lw)

axes[0].plot([0, chemeq1[-1, 0]], [0, chemeq1[-1, 1]], color=cmapp(0.8), linewidth=lw)
axes[0].plot(chemeq1[:, 0], chemeq1[:, 1],             color=cmapp(0.8), linewidth=lw)

mindx = np.argmin(NESS[:, 0]+NESS[:, 1])
axes[0].plot(NESS[:mindx, 0], NESS[:mindx, 1], "--",    color=cmapp(0.8), linewidth=lw)
axes[0].plot(NESS[mindx:, 0], NESS[mindx:, 1], "-",    color=cmapp(0.8), linewidth=lw, label=r"Reaction 1 nullcline")

axes[0].plot(cons_eq[indx2:, 0], cons_eq[indx2:, 1], color="C1", label=r"Reaction 2 nullcline", linewidth=lw)
axes[0].plot([cons_eq[indx1, 0], cons_eq[indx2, 0]], [cons_eq[indx1, 1], cons_eq[indx2, 1]], color="C1", linewidth=lw)
axes[0].plot(cons_eq[:indx4, 0], cons_eq[:indx4, 1], color="C1", linewidth=lw)
axes[0].plot(cons_eq[indx3:indx1, 0], cons_eq[indx3:indx1, 1], color="C1", linewidth=lw)
axes[0].plot([cons_eq[indx3, 0], cons_eq[indx4, 0]], [cons_eq[indx3, 1], cons_eq[indx4, 1]], color="C1", linewidth=lw)


points = [(0.287, 0.06), (0.32, 0.01), (0.28, 0.01)]
axes[0].plot(phi_bar[:, 0], phi_bar[:, 1], color="C2", linewidth=lw+0.5)
arrowlength = 0.01
time_indx = []

for i in range(len(points)):
    tli = np.argmin( np.square( phi_bar[:, 0] - points[i][0] ) + np.square( phi_bar[:, 1] - points[i][1] ) )
    delta_x = phi_bar[tli+1, 0] - phi_bar[tli, 0]
    delta_y = phi_bar[tli+1, 1] - phi_bar[tli, 1]
    time_indx.append(tli)
    dx = delta_x*arrowlength/np.sqrt( delta_x*delta_x + delta_y*delta_y)
    dy = delta_y*arrowlength/np.sqrt( delta_x*delta_x + delta_y*delta_y) 

point = (0.207, 0.09)
tli = np.argmin( np.square( phi_bar[:, 0] - point[0] ) + np.square( phi_bar[:, 1] - point[1] ) )
delta_x = phi_bar[tli+1, 0] - phi_bar[tli, 0]
delta_y = phi_bar[tli+1, 1] - phi_bar[tli, 1]

dx = delta_x*arrowlength/np.sqrt( delta_x*delta_x + delta_y*delta_y)
dy = delta_y*arrowlength/np.sqrt( delta_x*delta_x + delta_y*delta_y) 
time_indx.append(tli)

axes[0].fill_between([0, 1], [1, 1], [1, 0], color="black", alpha=0.2)
axes[0].plot(np.array([0, 0]), np.array([0, 1]), "k", linewidth=1)
axes[0].plot(np.array([0, 1]), np.array([0, 0]), "k", linewidth=1)
axes[0].plot(np.array([0, 1]), np.array([1, 0]), "k", linewidth=1)

axes[0].set_xlabel(r"Volume fraction A $\bar{\phi}_A$",  fontsize=10.5, labelpad=-0.5)#, labelpad=12)
axes[0].set_ylabel(r"Volume fraction B $\bar{\phi}_B$",  fontsize=10.5, labelpad=-0.5)#, labelpad=9)
legend = axes[0].legend(loc='upper center', ncol=2, frameon=True, fontsize=9, framealpha=1.0, bbox_to_anchor=(0.5, 1.29), columnspacing=1.2, handlelength=1.0)

frame = legend.get_frame()
frame.set_edgecolor('black')
frame.set_linewidth(0.35)


axes[0].set_xlim(0.19, 0.3405)#0.5)
axes[0].set_ylim(0, 0.1383)#0.16)
axes[0].set_xticks([0.20, 0.25, 0.30, 0.35])
axes[0].plot(0.29005, 0.0242,  "o", markerfacecolor="w",  markeredgewidth=1.2, markeredgecolor="black", markersize=7)

time_indx = np.array(time_indx)+roll_len
for i in range(len(time_indx)):
    time_indx[i] = int(len(T)+time_indx[i])%len(T)

tau = T[-1]
arrowlength = 0.1

for j in range(1):
    axes[1].fill_between([T[PS_indx[0]], T[PS_indx[-1]]], [-1, -1], [2, 2], color="gray", alpha=0.06)
    axes[1].plot([T[0], T[PS_indx[0]]], [0, 0], color="C2", linewidth=lw+0.5)
    axes[1].plot(T[PS_indx], 1-VI[PS_indx], color="C2", linewidth=lw+0.5)
    axes[1].plot([T[PS_indx[-1]], T[PS_indx[-1]]], [1-VI[PS_indx[-1]], 0], color="C2", linewidth=lw+0.5)

    axes[1].plot([T[PS_indx[-1]], T[-1]], [0, 0], color="C2", linewidth=lw+0.5)
    axes[1].set_ylim(-0.02, 0.27)
    
PS_mean = 1-np.mean(VI[PS_indx])
volmax  = 1-np.amin(VI[PS_indx])
absmean = PS_mean*len(PS_indx)/len(T)


axes[1].plot([0, T[-1]], np.array([1, 1])*PS_mean, color="C4", linewidth=0.5)
axes[1].plot([0, T[-1]], np.array([1, 1])*volmax, color="C3", linewidth=0.5)
axes[1].plot(T, np.ones(len(T))*absmean,  color="k", linewidth=0.5)

axes[1].set_xlim(0, 1.0*tau)
axes[1].set_xticks(np.array([0, 0.25*tau, 0.5*tau, 0.75*tau, tau]))#, 1.5*tau, 2*tau])
axes[1].set_xticklabels([r"$0$", r"$\tau/4$",  r"$\tau/2$", r"$3\tau/4$", r"$\tau$"])#, r"$3\tau/2$", r"$2\tau$"])
axes[1].set_xlabel(r"Time $t$", fontsize=10.5, labelpad=0)
axes[1].set_ylabel(r"Drop. vol. $V^{II}$", fontsize=10,labelpad=0)
line = axes[0].add_collection(lc)

VI[:PS_indx[0]] = 1 
VI[PS_indx[-1]+1:] = 1
points = np.array([T, 1-VI]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc2 = LineCollection(segments, cmap=cmap, norm=norm)
lc2.set_array(T)
lc2.set_linewidth(2.5)
line2 = axes[1].add_collection(lc2)


filename = root + "phasediagram_and_volume.pdf"
plt.savefig(filename, bbox_inches="tight", dpi=500)
os.system('pdfcrop %s %s &> /dev/null &'%(filename, filename))
plt.show()


