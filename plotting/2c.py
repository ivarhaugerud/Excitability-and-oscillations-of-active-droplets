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

fontsize = 9
plt.style.use(['science','no-latex'])#, "grid"])
import matplotlib
matplotlib.rc('xtick', labelsize=fontsize)
matplotlib.rc('ytick', labelsize=fontsize)
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
root = "figures/"


import matplotlib.colors as colors
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


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

k_p   = 10
k_c   = 1
Nt           = int(5*1e4)
counter = 1

phi         = np.linspace(1e-3, 1-1e-3, 500)

tiline_two_component = get_tieline_twocomp(phi, xi_ac, nu[1], nu[2], k_bT, omega[1], omega[2])
bino                 = calculate_binodal(tiline_two_component, np.array([0.001, 0.5]), 5000, k_bT, nu, xi_ab, xi_ac, xi_bc, omega)


fac = 1
intersect_tot = np.sum(bino[0, :])/2
thresh = intersect_tot*fac
kappa = 5*1e-4
k = 1 
k1, k2 = k, k

u, c = np.unique(bino[:, 0], return_index=True)
bot_func    = sci.interp1d(bino[c, 0], bino[c, 1], kind="quadratic")

P, bot, top = fill_between_arrays_type2(bino)

chem_eq, cons_eq = contour_lines_ab_psi(phi, nu, k_bT, omega, xi, bot, top, bino)
indx = bino_intersect_chemindx(chem_eq, bino)
chem_eq_pre = chem_eq[indx:, :]
indx_dense = bino_intersect_chemindx(chem_eq, bino[:, 2:])
chem_eq_post = chem_eq[:indx_dense, :]

#chem_eq = contour_lines_ab_fuel(phi, nu, k_bT, omega, xi_ab, xi_ac, xi_bc, P, bot, top, bino, xi, indx, delta, chem_eq, thresh, kappa, fuel_foo)
plt.clf()
indx = bino_intersect_binoindx(chem_eq, bino)

plt.plot(bino[:, 0], bino[:, 1])
plt.plot(bino[:, 2], bino[:, 3])
plt.plot(chem_eq_pre[:, 0], chem_eq_pre[:, 1])
plt.plot(chem_eq_post[:, 0], chem_eq_post[:, 1])
plt.plot(chem_eq_pre[0, 0], chem_eq_pre[0, 1], "ro")
plt.plot(chem_eq_post[-1, 0], chem_eq_post[-1, 1], "bo")

delta_x = bino[indx+1, 0] - bino[indx, 0]
delta_y = bino[indx+1, 1] - bino[indx, 1]
epsilon = np.sqrt(delta_x*delta_x + delta_y*delta_y)
angle = np.pi - np.arctan2(delta_y, delta_x)

phi_a0 = bino[indx, 0]
phi_b0 = bino[indx, 1]

#print((0.5 - np.sqrt(3*(xi_bc - 2)/8 )), (0.5 - np.sqrt(3*(xi_ac - 2)/8 )))
#approx_angle = np.arctan( (0.5 - np.sqrt(3*(xi_bc - 2)/8 ) )/(0.5 - np.sqrt(3*(xi_ac - 2)/8 )))
approx_angle = np.arctan2( (0.5 - np.sqrt(3*(xi_bc - 2)/8 ) ), (0.5 - np.sqrt(3*(xi_ac - 2)/8 )))
approx_angle = np.arctan2( bino[-1, 1], bino[0, 0] )

print("angles", approx_angle, angle)
plt.show()

chem_pot_diff = np.zeros(len(bino[:, 0]))
for i in range(len(chem_pot_diff)):
    temp  = calc_gibbs_mu(bino[i, :2], k_bT, nu, omega, xi)
    chem_pot_diff[i] = temp[0]-temp[1]

plt.plot(bino[:, 0]+bino[:, 1], chem_pot_diff)
#angle = approx_angle
term0 = xi_bc - xi_ac - xi_ab*(np.sin(angle)+np.cos(angle))/(np.cos(angle) - np.sin(angle))
term1 = (1/phi_a0)*np.cos(angle)/(np.cos(angle) - np.sin(angle))
term2 = (1/phi_b0)*np.sin(angle)/(np.cos(angle) - np.sin(angle))
psi2, psi1 = bino[indx, 3] + bino[indx, 2], bino[indx, 0] + bino[indx, 1]
fac   = (psi2 - psi1)
print(fac)
print(term0, term1, term2)
print(term0 + term1 + term2)
print(fac*(term0 + term1 + term2))
crit_fuel = np.log(1 + fac*(term0 + term1 + term2))
print("critical_fuel", crit_fuel)
delta_psi = np.linspace(1e-10, 0.1, int(1e4))
crit_fuel2 = np.log(1 + fac*(term0 + term1 + term2)/(1-delta_psi*5/(np.cos(angle) - np.sin(angle))))
plt.show()
plt.plot(delta_psi, crit_fuel2)
plt.show()
#plt.plot(bino[indx, 0], bino[indx, 1], "ko")
#plt.show()

term0 = xi_bc - xi_ac - xi_ab*(np.sin(angle)+np.cos(angle))/(np.cos(angle) - np.sin(angle))
term1 = (1/phi_a0)*np.cos(angle)/(np.cos(angle) - np.sin(angle))
term2 = (1/phi_b0)*np.sin(angle)/(np.cos(angle) - np.sin(angle))
psi2, psi1 = bino[indx, 3] + bino[indx, 2], bino[indx, 0] + bino[indx, 1]
xi_tilde = term0 + term1 + term2
alpha_tilde = np.cos(angle) - np.sin(angle)
xi_tilde = xi_tilde*alpha_tilde
epsilon = delta_psi#/((psi2 - psi1))
plt.plot(psi1+np.array([-0.2, 0.2]), xi_tilde*np.array([-0.2, 0.2]))
plt.show()

VI = 1 - (1-np.exp(chem_pot_diff))/(1-np.exp(chem_pot_diff) + np.exp(chem_pot_diff + delta) - 1)
VII = 1-VI
phi_bar = VI*(bino[:, 0] + bino[:, 1]) + VII*(bino[:, 2]+bino[:,3])

start = np.argmin(np.abs(bino[:, 0] + bino[:, 1] - psi1))
e_crit = 1-(np.exp(-chem_pot_diff) - 1)*(psi2 - psi1)/(phi_bar - psi1 + (bino[:, 0] + bino[:, 1] - psi1))

plt.plot(bino[:, 0] + bino[:, 1], VII)
psi1_indx = np.argmin(np.abs(bino[:, 0] + bino[:, 1] - psi1))
plt.plot(bino[psi1_indx, 0] + bino[psi1_indx, 1], 0, "ko")
plt.ylim(0, 1)
plt.show()

term0 = xi_bc - xi_ac - xi_ab*(np.sin(angle)+np.cos(angle))/(np.cos(angle) - np.sin(angle))
term1 = (1/phi_a0)*np.cos(angle)/(np.cos(angle) - np.sin(angle))
term2 = (1/phi_b0)*np.sin(angle)/(np.cos(angle) - np.sin(angle))
psi2, psi1 = bino[indx, 3] + bino[indx, 2], bino[indx, 0] + bino[indx, 1]
alpha_tilde = np.cos(angle) - np.sin(angle)
xi_tilde = (term0 + term1 + term2)*alpha_tilde
delta_psi = np.linspace(1e-10, 0.1, int(1e4))
epsilon = -delta_psi/alpha_tilde
argument = 1 - (np.exp(epsilon*xi_tilde) - 1)*(psi2 - psi1)/(epsilon*alpha_tilde - epsilon*alpha_tilde)

delta_crit_foo = np.log(argument)



plt.plot(np.log(e_crit))
plt.plot(start, np.log(e_crit[start]), "ko")
plt.show()
#for epsilon in [1e-6]:#, 1e-4, 1e-3, 1e-2, 1e-1, 1]:#, 1e-1]:
epsilon = -delta_psi / (np.sin(angle) - np.cos(angle))
print("\n\n HERE")
print(epsilon)
argument = 1 + (np.exp(epsilon*xi_tilde) - 1)*(psi2 - psi1)/(epsilon*alpha_tilde - delta_psi)
delta_crit_foo = np.log(argument)
#print(epsilon, alpha_tilde, argument)
print(xi_tilde)
plt.plot(delta_crit_foo, -delta_psi)
plt.show()

"""
term0 = xi_bc - xi_ac - xi_ab*(np.sin(angle)+np.cos(angle))/(np.cos(angle) - np.sin(angle))
term1 = (1/phi_a0)*np.cos(angle)/(np.cos(angle) - np.sin(angle))
term2 = (1/phi_b0)*np.sin(angle)/(np.cos(angle) - np.sin(angle))
psi2, psi1 = bino[indx, 3] + bino[indx, 2], bino[indx, 0] + bino[indx, 1]
alpha_tilde = np.sin(angle) - np.cos(angle)
xi_tilde = alpha_tilde*( term0 + term1 + term2 )
print("FAC", xi_tilde)
delta_psi = np.linspace(1e-7, 0.1, int(1e4))

for epsilon in [1e-6, 1e-4, 1e-3, 1e-2, 1e-1]:
    argument = epsilon*xi_tilde*(psi2 - psi1)/(- delta_psi)
    delta_crit_foo = np.log(1+argument)
    print(argument)
    plt.plot(np.log(1+argument), -delta_psi)
    plt.show()
"""

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
print("NUMERICAL CRITICAL", deltas[indx[-1]])
numerical_crit = deltas[indx[-1]]
phi_range, deltas_sub = phi_range[indx], deltas[indx]
upper_homo = np.zeros(len(deltas))
upper_homo[indx[-1]:] = phi_range[0, 1]
upper_homo[indx] = phi_range[:, 0]

#plt.plot(deltas, upper_homo)
#plt.show()
fig, axes = plt.subplots(nrows=2, ncols=1)
fig.subplots_adjust(wspace=100, hspace=100)
alpha_val = 0.2

axes[1].fill_between(deltas, np.ones(len(deltas))*phi_range[0, 1], np.ones(len(deltas)), color="C1", alpha=alpha_val)
axes[1].fill_between(deltas_sub, phi_range[:, 0], phi_range[:, 1], color="C2", alpha=alpha_val)
axes[1].fill_between(deltas, upper_homo, np.zeros(len(deltas)), color="C0", alpha=alpha_val)

#CS3 = axes[1].pcolormesh(X_psi, Y_psi, Z_psi, cmap=cm, rasterized=True)
#CS3.set_edgecolor('face')

#ax2.set_xlabel(r"Fuel strength $\delta/k_BT$", fontsize=10)
#ax2.set_ylabel(r"Conserved quantity $\Psi$", fontsize=10)
axes[1].text(0.1, 0.30, r"Homogeneous", fontsize=11, color="C0")
axes[1].text(2.5, 0.362, r"Active droplet", fontsize=11, color="C1")
axes[1].text(3.8, 0.30, r"Bistable", fontsize=11, color="C2")
axes[1].set_ylim(0.20, 0.40)
axes[1].set_xlim(0, np.amax(deltas))


axes[1].set_ylabel(r"Tot. vol. frac. $\bar{\Psi}$", fontsize=10, labelpad=-1)
axes[1].set_xlabel(r"Fuel strength $\delta/k_BT$", fontsize=10.5, labelpad=-1)
axes[1].plot([crit_fuel, crit_fuel], [0, 0.5], "--", color="C2")
axes[1].plot([numerical_crit, numerical_crit], [0, 0.5], "-", color="C2")


#plt.plot(delta_crit_foo, phi_range[0, 1]-delta_psi, "k:")
#plt.plot(np.log(e_crit), phi_bar)
#plt.plot(crit_fuel2, phi_range[0, 1]-delta_psi)

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
#axes[0].arrow(Psi1[0], 0.203, 0, 0.01, color="gray", alpha=1, head_width=0.015, head_length=0.04, shape='full', lw=0, length_includes_head=True)
#axes[0].arrow(Psi1[0], 0.068, 0, 0.01, color="gray", alpha=1, head_width=0.015, head_length=0.04, shape='full', lw=0, length_includes_head=True)
axes[0].plot([TLSC[indx_min]/2, TLSC[indx_min]/2], [0.0, NESSbino[indx_min, -1]], ":", color="gray", alpha=1, linewidth=0.6)
#axes[0].arrow(TLSC[indx_min], 0.01, 0, -0.01, color="gray", alpha=1, head_width=0.015, head_length=0.04, shape='full', lw=0, length_includes_head=True)


axes[0].plot(TLSC[:indx_min]/2, NESSbino[:indx_min, -1], "--", color=cm(0.5), linewidth=1.5)
axes[0].plot(TLSC[indx_min:indx_jmp]/2, NESSbino[indx_min:indx_jmp, -1], "-", color=cm(0.5), linewidth=1.5)
axes[0].plot(TLSC[indx_jmp:]/2, NESSbino[indx_jmp:, -1], "-", color=cm(0.5), linewidth=1.5)
axes[0].plot([0, Psi1[0]/2], [0, 0], cm(0.5), linewidth=1.5)
axes[0].set_yticks([0, 0.10, 0.20])

axes[0].set_ylim(-0.02, 0.3)
axes[0].set_xlim(0.20/2, 0.37/2)

axes[0].text(0.21/2, 0.084, "Binodal", rotation=-15, color="k")
#axes[0].text(0.768, 0.12, "Phase II", rotation=60, color="k")

#axes[0].text(0.30, 0.187, "PS nullcline", rotation=18.5, color=cm(0.5))
axes[0].text(0.202/2, 0.005, "Homog. nullcline", rotation=0.0, color="C0")

#axes[0].fill_between([0, 1], [NESSbino[indx_min, -1], NESSbino[indx_min, -1]], [0, 0], color="C1", alpha=0.04)
#axes[0].plot([0, 1], [NESSbino[indx_min, -1], NESSbino[indx_min, -1]], "--", color="C1", alpha=0.25)
#axes[0].plot([0, 1], [0, 0], "--", color="C1", alpha=0.25)
axes[0].plot(np.array([0, Psi1[0]])/2, [0, 0], "-",  color="C0")
axes[0].plot(np.array([Psi1[0], 1])/2, [0, 0], "--", color="C0")

axes[0].set_xticks([0.2/2, 0.25/2, 0.30/2, 0.35/2])
#axes[0].text(0.40, 0.06, "Inacessible volumes", color="C1")

axes[0].set_ylabel(r"Drop. vol. $V^{II}/V$", fontsize=10, labelpad=-1)
axes[0].set_xlabel(r"Total volume fraction $\bar{\Psi}$", fontsize=10, labelpad=-1)

fig.subplots_adjust(hspace=0.35)


filename  = "figures/state_diagrams.pdf"
plt.savefig(filename, bbox_inches="tight", dpi=500, transparent=True)
os.system('pdfcrop %s %s &> /dev/null &'%(filename, filename))
plt.show()
