import os
import numpy as np
import matplotlib.pyplot as plt
from functions import *

fontsize = 9
plt.style.use(['science','no-latex'])#, "grid"])
import matplotlib
matplotlib.rc('xtick', labelsize=fontsize)
matplotlib.rc('ytick', labelsize=fontsize)
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

#cm = matplotlib.cm.get_cmap('colorbrewer:Spectral')
root = "figures/"

import colormaps as cmaps
cm = cmaps.WhiteYellowOrangeRed
#cm = cm.shift(0.2)
cm = cm.cut(0.25, 'left')
#cm = cm.cut(0.2, 'right')


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

N         = 1000
phi_delta = 1/N
phi_min, phi_max, n_points = phi_delta, 1.- phi_delta , N
phi = np.linspace(phi_min, phi_max, N)

k_p   = 10
k_c   = 1
Nt           = int(5*1e4)
counter = 1

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
indx             = bino_intersect_chemindx(chem_eq, bino)
chemeq1          = chem_eq[indx:, :]
indx             = bino_intersect_chemindx(chem_eq, bino[:, 2:])
chemeq2          = chem_eq[indx:, :]

np.save("data/PBT__xiab%1.2f_xiac%1.2f_xibc%1.2f.npy" % (xi_ab, xi_ac, xi_bc), np.concatenate((P, bot, top)))
np.save("data/bino_xiab%1.2f_xiac%1.2f_xibc%1.2f.npy" % (xi_ab, xi_ac, xi_bc), bino)
plt.clf()


deltas = np.array([2, 2.5, 3.0, 3.5])

all_ness = {}

for j, delta in enumerate(deltas):
    NESS   = np.zeros((len(bino[:, 0]), 2))
    for i in range(len(bino[:, 0])):
        delta1 = fuel_foo(bino[i, 0], bino[i, 1], thresh, kappa, delta)
        delta2 = fuel_foo(bino[i, 2], bino[i, 3], thresh, kappa, delta)
        mus = calc_gibbs_mu(np.array([bino[i, 0], bino[i, 1]]), k_bT, nu, omega, xi)
        mua, mub = mus[0], mus[1]
        
        VI = -k2*(np.exp(delta2) - np.exp(mub-mua))/(k1*np.exp(delta1) - k2*np.exp(delta2) - (k1-k2)*np.exp(mub - mua))

        if VI > -1e-5 and VI < 1+1e-5:
            NESS[i, 0] = VI*bino[i, 0] + (1-VI)*bino[i, 2]
            NESS[i, 1] = VI*bino[i, 1] + (1-VI)*bino[i, 3]

    indx1 = bino_intersect_chemindx(chem_eq, bino[:, :2])
    indx2 = bino_intersect_chemindx(chem_eq, bino[:, 2:])
    eq_post = chem_eq[:indx2, :]
    eq_in   = chem_eq[indx2:indx1, :]

    indx     = np.where(NESS[:, 0] != 0)[0]
    NESS     = NESS[indx]

    all_ness[str(j)] = NESS


psi = 0.32

indxes = np.argwhere(np.abs(psi - np.sum(NESS, axis=1)) < 1e-4)[:, 0]
split  = np.argmax(np.gradient(indxes))+1
indx1  = int(np.mean(indxes[:split]))
indx2  = int(np.mean(indxes[split:]))

chem_eq_indx = np.argmin(np.abs(psi - np.sum(chemeq1, axis=1)))
fig, ax = plt.subplots()#nrows=2, ncols=1, height_ratios=[1, 0.5])
#plt.subplots_adjust(wspace=0, hspace=0.3)

lw = 1.3
ax.fill_between(P, bot, top, color="gray", alpha=0.06)
ax.plot(bino[:, 0], bino[:, 1], color="k", linewidth=lw)
ax.plot(bino[:, 2], bino[:, 3], color="k", linewidth=lw)

#Ntl = 11
#for i in range(Ntl):
#    tli = int((i+1)*len(bino[:, 0])/(Ntl+1))
#    ax.plot([bino[tli, 0], bino[tli, 2]], [bino[tli, 1], bino[tli, 3]], color="gray", linewidth=lw, alpha=0.15)

lw_nc = 1.5
ax.plot([0, chemeq1[-1, 0]], [0, chemeq1[-1, 1]], color="C0", linewidth=lw_nc)
ax.plot(chemeq1[:, 0], chemeq1[:, 1],             color="C0", linewidth=lw_nc)

#ax.plot(chemeq2[:, 0], chemeq2[:, 1],             color="C0", linewidth=lw_nc)
#ax.plot([chemeq1[0, 0], chemeq2[-1, 0]], [chemeq1[0, 1], chemeq2[-1, 1]], color="C0", linewidth=lw)

arrow_xs  = [0.235, 0.28, 0.295, 0.322]
sign      = [1,      -1,     1,     -1]
arrow_len = 0.01

#for i in range(len(arrow_xs)):
#    ax.arrow(arrow_xs[i], psi-arrow_xs[i], sign[i]*arrow_len, -sign[i]*arrow_len, color="gray", head_width=0.0025, head_length=0.005, shape='full', lw=0, length_includes_head=True)
ax.plot(phi, psi-phi, color="gray", linewidth=1.5)
ax.plot(chemeq1[chem_eq_indx, 0], chemeq1[chem_eq_indx, 1], "o", color="C0")

for j in range(len(deltas)):
    c_indx = (deltas[j] - np.amin(deltas))/(np.amax(deltas) - np.amin(deltas))

    if deltas[j] > 1:
        #plt.show()
        #plt.plot(all_ness[str(j)][:, 0] + all_ness[str(j)][:, 1])

        indx = np.argmin(all_ness[str(j)][:, 0] + all_ness[str(j)][:, 1])
        #plt.plot(indx, all_ness[str(j)][indx, 0] + all_ness[str(j)][indx, 1], "ko")
        #plt.show()

        ax.plot(all_ness[str(j)][:indx, 0], all_ness[str(j)][:indx, 1], "--", color=cm(c_indx), linewidth=lw_nc, alpha=0.75)
        ax.plot(all_ness[str(j)][indx:, 0], all_ness[str(j)][indx:, 1], "-", color=cm(c_indx), linewidth=lw_nc, alpha=0.75)

    else:
        ax.plot(all_ness[str(j)][:, 0], all_ness[str(j)][:, 1], color=cm(c_indx), linewidth=lw_nc, alpha=0.75)

ax.plot(0, 0, "k-", label="Stable")
ax.plot(0, 0, "k--", label="Unstable")
ax.plot(NESS[indx2, 0], NESS[indx2, 1], "o", color=cm(0.99))
ax.plot(NESS[indx1, 0], NESS[indx1, 1], "o", markerfacecolor="w",  markeredgewidth=1.2, markeredgecolor="black")

ax.fill_between([0, 1], [1, 1], [1, 0], color="black", alpha=0.2)
ax.plot(np.array([0, 0]), np.array([0, 1]), "k", linewidth=1)
ax.plot(np.array([0, 1]), np.array([0, 0]), "k", linewidth=1)
ax.plot(np.array([0, 1]), np.array([1, 0]), "k", linewidth=1)

ax.set_xlabel(r"Volume fraction A $\bar{\phi}_A$",  fontsize=1+9, labelpad=-0.5)#, labelpad=-4)
ax.set_ylabel(r"Volume fraction B $\bar{\phi}_B$",  fontsize=1+9, labelpad=-0.5)#, labelpad=-4)
#ax.set_yticks([0, 0.02, 0.06, 0.08])
#ax.set_xticks([0.24, 0.26, 0.28, 0.32, 0.34, 0.36])

import matplotlib as mpl
norm = mpl.colors.Normalize(vmin=deltas.min(), vmax=deltas.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
#mapcaller = matplotlib.cm.get_cmap('Spectral')
cbar = fig.colorbar(cmap, ticks=[2.0, 2.5, 3.0, 3.5, 4.0], pad=0, ax=plt.gca())
cbar.ax.set_ylabel(r'Fuel strength $\delta/k_BT$', rotation=90, fontsize=1+9, labelpad=1.5)

ax.text(0.24, 0.010, "    Homogeneous", rotation=2.5, fontsize=1+9, color="C0")
ax.text(0.24, 0.0040, "reaction nullcline", rotation=2.5, fontsize=1+9, color="C0")
ax.text(0.25, 0.0195, "Binodal", rotation=-12.5, fontsize=1+9, color="k")
ax.text(0.252, 0.0362, "Conservation line", rotation=-52, fontsize=1+9, color="gray")
ax.text(0.315, 0.045, "Reaction",  rotation=65, fontsize=1+9, color=cm(0.3333))
ax.text(0.3239, 0.0455, "nullcline", rotation=65, fontsize=1+9, color=cm(0.3333))

ax.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left=True,      # ticks along the bottom edge are off
    right=False) # labels along the bottom edge are off

#plt.legend(loc="upper right", fontsize=1+9, frameon=True, framealpha=1)
ax.set_xticks([0.25, 0.28, 0.31, 0.34])
legend = ax.legend(loc='upper center', ncol=2, frameon=True, fontsize=9, framealpha=1.0, bbox_to_anchor=(0.5, 1.029),)
frame = legend.get_frame()
frame.set_edgecolor('black')
frame.set_linewidth(0.35)

ax.set_xlim(0.235, 0.36)
ax.set_ylim(0.00, 0.085)
filename = root + "2a.pdf"
fig.savefig(filename, bbox_inches="tight", dpi=500)
os.system('pdfcrop %s %s &> /dev/null &'%(filename, filename))
plt.show()



