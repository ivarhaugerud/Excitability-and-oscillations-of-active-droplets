import os
import numpy as np
import matplotlib.pyplot as plt
from functions import *
import scipy.integrate as scint

import colormaps as cmaps
cm = cmaps.WhiteYellowOrangeRed
cm = cm.cut(0.25, 'left')

fontsize = 7
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

import matplotlib as mpl

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

deltas1 = np.arange(0,    4.01, 0.10)
deltas2 = np.arange(2.61, 2.81, 0.02)
deltas3 = np.arange(2.83, 2.89, 0.02)
deltas4 = np.arange(0.72, 0.78, 0.02)
deltas  = np.concatenate((deltas1, deltas2, deltas3, deltas4))
deltas  = np.sort(deltas)

all_VS = np.zeros(len(deltas))
all_VS_max = np.zeros(len(deltas))
all_VS_mean = np.zeros(len(deltas))

k_p    = 1
thresh = 0.5
kappa  = 5*1e-4

N = int(5*1e5)
k_c = 1

energy      = np.zeros((len(deltas), 3))
periods     = np.zeros(len(deltas))
lengths     = np.zeros(len(deltas))


prev_orbit  = False

line_spacing = 0.2 
prev_delta = -2
not_yet_inside =  True
SS_PS_indx = 1
mu_F = 1

all_NESS = {}
k1, k2 = 1, 1

excited_PS_found = False
for I, delta in enumerate(deltas):
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
    i = I

    indx = np.where(NESS[:, 0] != 0)[0]
    NESS = NESS[indx]

    try:
        mindx = np.argmin(NESS[:, 0]+NESS[:, 1])
        all_NESS[str(i)+"u"] = NESS[:mindx, :]
        all_NESS[str(i)+"s"] = NESS[mindx:, :]

        if np.amin(NESS[:, 0]+NESS[:, 1]) < NESS[0, 0]+NESS[0, 1] and not excited_PS_found:
            excited_PS_found = True
            excited_PS = delta

    except:
        print("did not work")
    
    data = np.transpose(np.load("data/hysterisis_xiab%3.2f_xiac%3.2f_xibc%3.2f_kp%2.3f_delta%3.2f_more.npy" % (xi_ab, xi_ac, xi_bc, k_p, delta)))

    phi_bar = data[:,  :2]
    phiI    = data[:, 2:4]
    phiII   = data[:, 4:6]
    VI      = data[:,  6]
    F       = data[:,  7]
    mus_p   = data[:,  8]
    mus_x   = data[:,  9]
    VII     = 1-VI
    T       = data[:,  -1]

    periods[i], orbit = get_period(T, phi_bar[:, 0],  phi_bar[:, 1], 1e-4)
    indx = np.argmin(np.abs(T-(T[-1]-periods[i])))

    if VI[-1] > 1e-5 and not_yet_inside:
        SS_PS_indx = i
        not_yet_inside = False

    if orbit and not prev_orbit:
        orbit_delta = delta
        prev_orbit  = True
    
    if not orbit:
        if VI[-1] < 1e-4:
            mu         = calc_gibbs_mu(phi_bar[-1,  :], k_bT, nu, omega, xi)
            delta_hom  = fuel_foo(phi_bar[-1,  0], phi_bar[-1,  1], thresh, kappa, delta)

            k_f       = k_c*(np.exp(delta_hom) - 1)/(np.exp(mu_F))

            energy[i, 0]  = (mu_F)*k_f*(np.exp(mu[0]+mu_F))
            energy[i, 1]  = (mu[0] - mu[1])*( k_c*(np.exp(mu[0]) - np.exp(mu[1])) + k_f*(np.exp(mu[0]+mu_F) ) )
            energy[i, 2]  = (mu[0] + mu[1] - 2*mu[2])*k_p*(np.exp(mu[0]+mu[1]) - np.exp(2*mu[2]))
            all_VS[i] = np.nan

        else:
            deltaI  = fuel_foo(phiI[-1,  0], phiI[-1,  1], thresh, kappa, delta)
            deltaII = fuel_foo(phiII[-1,  0], phiII[-1,  1], thresh, kappa, delta)
            
            muI   = calc_gibbs_mu(phiI[-1,  :], k_bT, nu, omega, xi)
            muII  = calc_gibbs_mu(phiII[-1, :], k_bT, nu, omega, xi)

            k_f1 = k_c*(np.exp(deltaI)  - 1)/(np.exp(mu_F) )
            k_f2 = k_c*(np.exp(deltaII) - 1)/(np.exp(mu_F) )

            energy[i, 0]  =     VI[-1]*(mu_F)*k_f1*(np.exp(muI[0]+ mu_F) )
            energy[i, 0] += (1-VI[-1])*(mu_F)*k_f2*(np.exp(muII[0]+mu_F) )    
            
            energy[i, 1] +=     VI[-1]*(muI[ 0] - muI[ 1])*( k_c*(np.exp(muI[ 0]) - np.exp(muI[ 1])) + k_f1*(np.exp(muI[0]+ mu_F)))
            energy[i, 1] += (1-VI[-1])*(muII[0] - muII[1])*( k_c*(np.exp(muII[0]) - np.exp(muII[1])) + k_f2*(np.exp(muII[0]+mu_F)))

            energy[i, 2]  =     k_p*VI[-1]*(muI[0] + muI[1]  - 2*muI[2] )*(np.exp(muI[0]+muI[1])    - np.exp(2*muI[2]))
            energy[i, 2] += k_p*(1-VI[-1])*(muII[0]+muII[1]  - 2*muII[2])*(np.exp(muII[0]+muII[1]) - np.exp(2*muII[2]))
            all_VS[i] = (1-VI[-1])
    else:
        x = T[indx:]
        y = np.zeros((len(x), 3))

        for l in range(len(x)):
            if VI[indx+l] > 1e-5:
                deltaI  = fuel_foo(phiI[l+indx,   0], phiI[l+indx,   1], thresh, kappa, delta)
                deltaII = fuel_foo(phiII[l+indx,  0], phiII[l+indx,  1], thresh, kappa, delta)

                muI     = calc_gibbs_mu(phiI[l+indx,  :], k_bT, nu, omega, xi)
                muII    = calc_gibbs_mu(phiII[l+indx, :], k_bT, nu, omega, xi)

                k_f1 = k_c*(np.exp(deltaI)  - 1)/(np.exp(mu_F))
                k_f2 = k_c*(np.exp(deltaII) - 1)/(np.exp(mu_F))

                y[l, 0]  = (1-VI[l+indx])*(mu_F)*k_f2*(np.exp(muII[0]+mu_F))    
                y[l, 0] += (  VI[l+indx])*(mu_F)*k_f1*(np.exp( muI[0]+mu_F))    
                
                y[l, 1]  =     VI[l+indx]*(muI[ 0] - muI[ 1])*( k_c*(np.exp(muI[ 0]) - np.exp(muI[ 1])) + k_f1*(np.exp(muII[0]+mu_F)) )
                y[l, 1] += (1-VI[l+indx])*(muII[0] - muII[1])*( k_c*(np.exp(muII[0]) - np.exp(muII[1])) + k_f2*(np.exp( muI[0]+mu_F)) )

                y[l, 2]  = k_p*(1-VI[l+indx])*(muII[0] + muII[1] - 2*muII[2])*(np.exp(muII[0]+muII[1]) - np.exp(2*muII[2]))   
                y[l, 2] += k_p*(  VI[l+indx])*(muI[0] + muI[1]   -  2*muI[2])*(np.exp( muI[0]+muII[1]) - np.exp(2*muII[2])) 
              
            else:
                mu         = calc_gibbs_mu(phi_bar[l+indx,  :], k_bT, nu, omega, xi)
                delta_hom  = fuel_foo(phi_bar[l+indx,  0], phi_bar[-1,  1], thresh, kappa, delta)

                k_f = k_c*(np.exp(delta_hom) - 1)/(np.exp(mu_F))

                y[l, 0]  = (mu_F)*k_f*(np.exp(mu[0]+mu_F) )
                y[l, 1]  = (mu[0] - mu[1])*( k_c*(np.exp(mu[0]) - np.exp(mu[1])) + k_f*(np.exp(mu[0]+mu_F) ))
                y[l, 2]  = (mu[0] + mu[1] - 2*mu[2])*k_p*(np.exp(mu[0]+mu[1]) - np.exp(2*mu[2]))

        energy[i, 0] = scint.trapezoid(y[:, 0], x)/periods[i]
        energy[i, 1] = scint.trapezoid(y[:, 1], x)/periods[i]
        energy[i, 2] = scint.trapezoid(y[:, 2], x)/periods[i]

        phi_bar = phi_bar[indx:, :]
        lengths[i] = get_curve_length(phi_bar[:, 0], phi_bar[:, 1])

        homo_indexes = np.where(VI[indx:] == 0)[0]
        PS_indexes = np.where(VI[indx:] != 0)[0]
        VI[indx:][homo_indexes] = 1
        all_VS_max[i] = np.amax(1-VI[indx:])        
        all_VS[i]    = 1-np.mean(VI[indx:])
        all_VS_mean[i] = 1-np.mean(VI[indx:][PS_indexes])

fig, axes = plt.subplots(nrows=3, ncols=1, height_ratios=[1, 1, 1], figsize=(3, 2.5))
plt.subplots_adjust(wspace=0, hspace=0.2)

PS_start = np.where(energy > 1e-4)[0][0]
delta_PS_start = (deltas[SS_PS_indx] + deltas[SS_PS_indx-1])/2
orbit_delta_indx = np.argmin(np.abs(deltas - orbit_delta))
axes[1].fill_between([0, delta_PS_start], [10, 10], [-10, -10], color="C0", alpha=0.1)
axes[1].fill_between([delta_PS_start, excited_PS], [10, 10], [-10, -10], color="C1", alpha=0.1)
axes[1].fill_between([excited_PS, orbit_delta], [10, 10], [-10, -10], color="C2", alpha=0.1)
axes[1].fill_between([orbit_delta, 5], [10, 10], [-10, -10], color="C3", alpha=0.1)

first_point = np.where(all_VS > 0.1)[0][0]
axes[1].plot(deltas,  all_VS, color="k")
axes[1].plot(deltas[orbit_delta_indx:], all_VS_max[orbit_delta_indx:], color="C6")
axes[1].plot(deltas[orbit_delta_indx:], all_VS_mean[orbit_delta_indx:], color="C4")
axes[1].set_ylim(0.04, 1.2)  # most of the data
axes[1].set_xlim(-0.05, np.amax(deltas)+0.05)
axes[1].set_yscale("log")

FS = 6.5
axes[1].set_xticklabels([])
axes[1].set_ylabel(r"Drop. vol. $V^{\text{II}}/V$", fontsize=FS)

axes[1].tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=True,      # ticks along the bottom edge are off
    top=False) # labels along the bottom edge are off

axes[0].fill_between([0, delta_PS_start], [10, 10], [-10, -10], color="C0", alpha=0.1)
axes[0].fill_between([delta_PS_start, excited_PS], [10, 10], [-10, -10], color="C1", alpha=0.1)
axes[0].fill_between([excited_PS, orbit_delta], [10, 10], [-10, -10], color="C2", alpha=0.1)
axes[0].fill_between([orbit_delta, 5], [10, 10], [-10, -10], color="C3", alpha=0.1)

axes[0].set_xticklabels([])
axes[0].tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=True,      # ticks along the bottom edge are off
    top=False) # labels along the bottom edge are off

axes[0].plot(deltas,  energy[:, 0], "k")
axes[0].set_ylim(-0.005, 0.115)  # most of the data
axes[0].set_xlim(-0.05, np.amax(deltas)+0.05)
axes[0].set_ylabel(r"Entropy prod. $\frac{\dot{S}}{k_1k_B}$", fontsize=FS, labelpad=8)







axes[2].fill_between([0, delta_PS_start], [10, 10], [-10, -10], color="C0", alpha=0.1)
axes[2].fill_between([delta_PS_start, excited_PS], [10, 10], [-10, -10], color="C1", alpha=0.1)
axes[2].fill_between([excited_PS, orbit_delta], [10, 10], [-10, -10], color="C2", alpha=0.1)
axes[2].fill_between([orbit_delta, 5], [10, 10], [-10, -10], color="C3", alpha=0.1)

first_point = np.where(all_VS > 0.1)[0][0]

idx     = np.argmin(np.abs(deltas - 3.1))
deltas  = np.delete(deltas, idx)
lengths = np.delete(lengths, idx)

axes[2].plot(deltas, lengths, "k")

axes[2].set_ylim(-0.02, 0.45)  # most of the data
axes[2].set_xlim(-0.05, np.amax(deltas)+0.05)

FS = 6.5

axes[2].set_xlabel(r"Fuel strength $\delta/k_BT$",   fontsize=FS+1, labelpad=0)
axes[2].set_ylabel(r"Orbit length", fontsize=FS)

axes[2].tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=True,      # ticks along the bottom edge are off
    top=False) # labels along the bottom edge are off



filename = root + "hysterisis_volume.pdf"
fig.savefig(filename, bbox_inches="tight", dpi=500)
os.system('pdfcrop %s %s &> /dev/null &'%(filename, filename))
plt.show()