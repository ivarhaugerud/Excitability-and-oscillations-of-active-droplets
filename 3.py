import numpy as np 
import scipy.spatial     as scs
import scipy.optimize    as sco
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from functions import *
import scipy.interpolate as sci
import os
import itertools

fontsize = 7
plt.style.use(['science','no-latex'])
import matplotlib
matplotlib.rc('xtick', labelsize=fontsize)
matplotlib.rc('ytick', labelsize=fontsize)
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
cmap = matplotlib.cm.get_cmap('viridis')
root = "figures/"

import colormaps as cmaps
cm = cmaps.WhiteYellowOrangeRed
cm = cm.cut(0.25, 'left')

k_bT  = 1

xi_ab = 0
xi_ac = 2.01
xi_bc = 3
delta = 4

xi    = np.array(([0, xi_ab, xi_ac],
                  [xi_ab, 0, xi_bc],
                  [xi_ac, xi_bc, 0]))
nu = np.ones(3)


omega = np.array([-2.5, 0.1, -1.925])

k_p   = 5
k_c   = 1
Nt    = int(2*1e4)
counter = 1

phi         = np.linspace(1e-3, 1-1e-3, 300)

tiline_two_component = get_tieline_twocomp(phi, xi_ac, nu[1], nu[2], k_bT, omega[1], omega[2])
bino                 = calculate_binodal(tiline_two_component, np.array([0.001, 0.5]), 25000, k_bT, nu, xi_ab, xi_ac, xi_bc, omega)

fac = 1
intersect_tot = np.sum(bino[0, :])/2
thresh = intersect_tot*fac
kappa = 5*1e-4
k1, k2 = k_c, k_c

u, c = np.unique(bino[:, 0], return_index=True)
bot_func    = sci.interp1d(bino[c, 0], bino[c, 1], kind="quadratic")

P, bot, top = fill_between_arrays_type2(bino)

chem_eq, cons_eq = contour_lines_ab_psi(phi, nu, k_bT, omega, xi, bot, top, bino)
indx = bino_intersect_chemindx(chem_eq, bino)
chemeq1 = chem_eq[indx:, :]
plt.clf()

diff = np.zeros(len(bino[:, 0]))
for i in range(len(bino[:, 0])):
    temp             = calc_gibbs_mu(np.array([bino[i, 0], bino[i, 1]]), k_bT, nu, omega, xi)
    diff[i]          = (temp[0]+temp[1]-2*temp[2])

TLI = np.argmin(np.abs(diff[:int(len(diff)/2)]))
left = cons_eq[:1192, :]
righ = np.roll(cons_eq[1193:, :], -21, axis=0)

dist = np.zeros(len(left[:, 0]))
for i in range(len(dist)):
    dist[i] = np.amin( np.square(left[i, 0] - bino[:, 2]) + np.square(left[i, 1] - bino[:, 3]) )


dist2 = np.zeros(len(righ[:, 0]))
for i in range(len(dist2)):
    dist2[i] = np.amin( np.square(righ[i, 0] - bino[:, 0]) + np.square(righ[i, 1] - bino[:, 1]) )

NESS = np.zeros((len(bino[:, 0]), 2))
NESSbino = np.zeros((len(bino[:, 0]), 5))

for i in range(len(bino[:, 0])):
    delta1 = fuel_foo(bino[i, 0], bino[i, 1], thresh, kappa, delta)
    delta2 = fuel_foo(bino[i, 2], bino[i, 3], thresh, kappa, delta)
    mus = calc_gibbs_mu(np.array([bino[i, 0], bino[i, 1]]), k_bT, nu, omega, xi)
    mua, mub = mus[0], mus[1]
    
    VI = -k2*(np.exp(delta2) - np.exp(mub-mua))/(k1*np.exp(delta1) - k2*np.exp(delta2) - (k1-k2)*np.exp(mub - mua))

    if VI > 0 and VI < 1:
        NESS[i, 0] = VI*bino[i, 0] + (1-VI)*bino[i, 2]
        NESS[i, 1] = VI*bino[i, 1] + (1-VI)*bino[i, 3]
        NESSbino[i, :-1] = bino[i, :]
        NESSbino[i,  -1] = 1-VI

indx1 = bino_intersect_chemindx(chem_eq, bino[:, :2])
indx2 = bino_intersect_chemindx(chem_eq, bino[:, 2:])
eq_post = chem_eq[:indx2, :]
eq_in = chem_eq[indx2:indx1, :]

indx     = np.where(NESS[:, 0] != 0)[0]
NESS     = NESS[indx]
NESSbino = NESSbino[indx, :]


p_flux_rate = np.zeros(len(NESS))
for i in range(len(NESS)):
    mus = calc_gibbs_mu(np.array([NESSbino[i, 0], NESSbino[i, 1]]), k_bT, nu, omega, xi)
    p_flux_rate[i] = np.abs(mus[0]+mus[1]-2*mus[2])

start_indx = int(len(p_flux_rate)/2)
s_start_index = np.argmin(p_flux_rate[start_indx:])+start_indx

N = int(1e5)
T = np.linspace(0, 50, N)


dt = T[1]-T[0]
phi_bar = np.zeros((N, 2))
phiII = np.zeros((N,   3))
phiI = np.zeros((N,    3))
V, VI, VII = 1, np.zeros(N), np.zeros(N)
eq_point = [0.216868, 0.06605]
delta_x, delta_y = -0.00, -0.0185
phi_bar[0, :] = np.array([eq_point[0]+delta_x, eq_point[1]+delta_y])

phi_bar, phiI, phiII, VI, VII, F, chems, rates, T = run_system(T, phi_bar, phiI, phiII, V, VI, VII, k_bT, xi_ab, xi_ac, xi_bc, xi, nu, omega, k_p, k_c, bino, P, bot, top, thresh, kappa, delta)

fig = plt.figure(figsize=(3.5, 1.35))
lw = 1.25
plt.fill_between(P, bot, top, color="gray", alpha=0.06)
plt.plot(bino[:, 0], bino[:, 1], color="k", linewidth=0.8)
plt.plot(bino[:, 2], bino[:, 3], color="k", linewidth=0.8)


start_points = np.array([[0.20, 0.5*1e-3],
                         [0.25, 0.5*1e-3],
                         [0.30, 0.5*1e-3],
                         [0.35, 0.5*1e-3],
                         [0.38, 0.5*1e-3],
                         [0.38, 0.075],
                         [0.15, 0.5*1e-3],
                         [0.15, 0.075],
                         [0.15, 0.125],
                         [0.18, 0.12],
                         [0.18, 0.10],
                         ])

arrowlength = 0.01
for i in range(len(start_points[:, 0])):
    phi_start_x = start_points[i, 0]
    phi_start_y = start_points[i, 1]
    phi_bar_quiver = np.load("data/quiver_plot_droplet_%3.3f_%3.3f.npy" % (phi_start_x, phi_start_y))

    plt.plot(phi_bar_quiver[:, 0], phi_bar_quiver[:, 1], color="gainsboro", linewidth=1)

Ntl = 5
bino_len = len(bino[:, 0])

lw_path = 2.0
lw_bino = 1.2
plt.plot([0, chemeq1[-1, 0]], [0, chemeq1[-1, 1]], color=cm(0.8), linewidth=lw)
plt.plot(chemeq1[:, 0], chemeq1[:, 1],             color=cm(0.8), linewidth=lw)

mindx = np.argmin(NESS[:, 0]+NESS[:, 1])
plt.plot(NESS[:mindx, 0], NESS[:mindx, 1], "--",   color=cm(0.8), linewidth=lw)
plt.plot(NESS[mindx:, 0], NESS[mindx:, 1], "-",    color=cm(0.8), linewidth=lw, label=r"Reaction 1 nullcline")
plt.plot([bino[TLI, 0], bino[TLI, 2]], [bino[TLI, 1], bino[TLI, 3]], color="C1", linewidth=lw_bino, label=r"Reaction 2 nullcline")
plt.plot([phi_bar[-1, 0], eq_point[0]], [phi_bar[-1, 1], eq_point[1]], color=cm(0.65), linewidth=lw_path, alpha=0.5)

plt.plot([phi_bar[0, 0], eq_point[0]], [phi_bar[0, 1], eq_point[1]], "-", color="k")
dx = (-eq_point[0]+phi_bar[0, 0])/5
dy = (-eq_point[1]+phi_bar[0, 1])/5

plt.fill_between([0, 1], [1, 1], [1, 0], color="black", alpha=0.2)
plt.plot(np.array([0, 0]), np.array([0, 1]), "k", linewidth=1)
plt.plot(np.array([0, 1]), np.array([0, 0]), "k", linewidth=1)
plt.plot(np.array([0, 1]), np.array([1, 0]), "k", linewidth=1)


plt.plot(phi_bar[:, 0], phi_bar[:, 1], linewidth=lw_path, color="C2", alpha=0.5)

plt.plot([phi_bar[-1, 0], eq_point[0]], [phi_bar[-1, 1], eq_point[1]], "-", color="C2", alpha=0.5, linewidth=lw_path)
plt.plot([phi_bar[-1, 0]], [phi_bar[-1, 1]], "o", color="C2", markersize=1.5)

plt.xlabel(r"Volume fraction $A$    $\bar{\phi}_A$",  fontsize=8, labelpad=0)
plt.ylabel(r"Volume fraction $B$    $\bar{\phi}_B$",  fontsize=8, labelpad=0)

legend = plt.legend(loc='upper center', ncol=2, frameon=True, fontsize=8, framealpha=1.0, bbox_to_anchor=(0.5, 1.25), columnspacing=1.9)
frame = legend.get_frame()
frame.set_edgecolor('black')
frame.set_linewidth(0.35)

plt.yticks([0, 0.05, 0.10, 0.15])
plt.xticks([0.20, 0.25, 0.30, 0.35])
plt.xlim(0.18, 0.36)
plt.ylim(0.00, 0.16)
filename = root + "excitable.pdf"
plt.savefig(filename, bbox_inches="tight", dpi=500)
os.system('pdfcrop %s %s &> /dev/null &'%(filename, filename))
plt.show()

