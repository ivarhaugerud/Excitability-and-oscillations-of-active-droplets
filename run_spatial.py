import numpy as np
from numpy.fft import rfft2, irfft2, fftfreq, rfftfreq


def fuel_foo(c, d, thresh, steep, delta):
    return (np.arctan( (c+d - thresh)/steep) + np.pi/2)*(delta/np.pi)

def non_linear_terms(A_tilde, B_tilde, delta, thresh, steep, fuel, k_p, omegaA, omegaB, omegaC, rho_A, rho_B, Gamma_A, Gamma_B, Lambda_A, Lambda_B, k_x, k_y, k2, U):
    A = irfft2(A_tilde)
    B = irfft2(B_tilde)

    fuel_vals = fuel_foo(A, B, thresh, steep, fuel)
    S = 1-A-B
    S_tilde = rfft2(S)
    
    # get mobilities
    A_mobfac = A
    B_mobfac = B
    C_mobfac = S

    # chemical potentials (without surface tension) (in real space)
    Lambda = 1 + (chi_AB*A*B + chi_AS*A*S + chi_BS*B*S)
    df0dA = np.log(A) + 1 + chi_AB*B + chi_AS*S + omegaA - Lambda + U*A
    df0dB = np.log(B) + 1 + chi_AB*A + chi_BS*S + omegaB - Lambda + U*B
    df0dC = np.log(S) + 1 + chi_AS*A + chi_BS*B + omegaS - Lambda #+ U*S

    # surface tension chemical contribution (in real space)
    gradA = irfft2(Lambda_A*k2*A_tilde)
    gradB = irfft2(Lambda_B*k2*B_tilde)

    mu_A = df0dA + gradA
    mu_B = df0dB + gradB
    mu_C = df0dC

    #laplacians becomes k-vectors, here you you calculate spatial derivative of mu in k-space
    chem_pot_A_x = irfft2(1j*k_x*rfft2(mu_A - mu_C)) 
    chem_pot_A_y = irfft2(1j*k_y*rfft2(mu_A - mu_C))
    chem_pot_B_x = irfft2(1j*k_x*rfft2(mu_B - mu_C))
    chem_pot_B_y = irfft2(1j*k_y*rfft2(mu_B - mu_C))

    #Add noise and mobility to chemical potential
    vec_x_A = rfft2(rho_A*Gamma_A*A_mobfac*chem_pot_A_x)
    vec_y_A = rfft2(rho_A*Gamma_A*A_mobfac*chem_pot_A_y)
    vec_x_B = rfft2(rho_B*Gamma_B*B_mobfac*chem_pot_B_x)
    vec_y_B = rfft2(rho_B*Gamma_B*B_mobfac*chem_pot_B_y)
   
    r1  = rfft2((np.exp(mu_A + fuel_vals)  - np.exp(mu_B)))
    r2  = rfft2(k_p*(np.exp(mu_A + mu_B)  - np.exp(2*mu_C)))

    #divergence of mobility times chemical potential gradient, and noise
    return (1j*k_x*vec_x_A + 1j*k_y*vec_y_A  - r1 - r2, 1j*k_x*vec_x_B + 1j*k_y*vec_y_B  + r1 - r2)
    

def implicitrk(u, v, delta, thresh, steep, fuel, k_p, omegaA, omegaB, omegaC, rho_A, rho_B, n_step, Gamma_A, Gamma_B, Lambda_A, Lambda_B, k_x, k_y, k2, U, u_val, l):
    """
    Time integration using the ETDRK4 scheme
    """
    niter = 0
    changed_stepsise = False

    while niter < n_step+1:
        Nu, Nv = non_linear_terms(u, v, delta, thresh, steep, fuel, k_p, omegaA, omegaB, omegaC, rho_A, rho_B, Gamma_A, Gamma_B, Lambda_A, Lambda_B, k_x, k_y, k2, U)

        # the implicit scheme gives a factor infront of u_{t+1}, this gives the denominator below
        # this line gives c(t+1), d(t+1) based on c(t) and d(t)
        # A_c and A_d comes from the algorithm
        u = ((1+A_c*delta*kappa_A*k2**2)*u + delta*Nu)/(1+A_c*delta*kappa_A*k2**2)
        v = ((1+A_d*delta*kappa_B*k2**2)*v + delta*Nv)/(1+A_d*delta*kappa_B*k2**2)

        if (niter%figskip == 0):
            u_tilde = irfft2(u)
            v_tilde = irfft2(v)

            print(niter, np.mean(u_tilde), np.mean(v_tilde), np.mean(u_tilde+v_tilde))
            np.save("data/phiA_fuel%3.2f_delta%3.5f_GammaA%3.5f_GammaB%3.5f_Lambda%3.5f_Lambda%3.5f_chiAB%3.3f_chiAC%3.3f_chiBC%3.3f_phiA%3.3f_phiB%3.3f_uval%3.2f_l%3.2f_time%3.6f" % (fuel, delta, Gamma_A, Gamma_B, Lambda_A, Lambda_B, chi_AB, chi_AS, chi_BS, Atot, Btot, u_val, l, niter*delta) + ".npy", u_tilde)
            np.save("data/phiB_fuel%3.2f_delta%3.5f_GammaA%3.5f_GammaB%3.5f_Lambda%3.5f_Lambda%3.5f_chiAB%3.3f_chiAC%3.3f_chiBC%3.3f_phiA%3.3f_phiB%3.3f_uval%3.2f_l%3.2f_time%3.6f" % (fuel, delta, Gamma_A, Gamma_B, Lambda_A, Lambda_B, chi_AB, chi_AS, chi_BS, Atot, Btot, u_val, l, niter*delta) + ".npy", v_tilde)
        
        yield (u,v)
        niter = niter + 1
       

# number of spatial gridpoints
n = 150
l = 0.15

# internal energies
omegaS = -2.25
omegaA = -2.5
omegaB =  0.1

#time step-stuff
delta      = 1e-6
T          = 100
n_step     = int(T/delta)
skip_frame = 1000
Nfigs      = 1000
figskip    = int(n_step/Nfigs)
skip_frame    = int(n_step/skip_frame)

#use the expression 0.5*( max(M) + min(M) ), where max(M)=lambda/4, and min(M) = 0
D_A = 1
D_B = 1
A_c = 0.5*D_A*0.25
A_d = 0.5*D_B*0.25

#interactions and thermodynamics
chi_AB =  0.00
chi_AS =  2.01
chi_BS =  3.00

#surfaace tension
kappa_A = 0.0001125
kappa_B = 0.0001125

#molecular volume
nu_A = 1
nu_B = 1
nu_S = 1
rho_A = nu_A/nu_S
rho_B = nu_B/nu_S 

#define system size
x, h = np.linspace(0, l, n, endpoint=False, retstep=True)

k_c      = 1
l_scale  = np.sqrt(D_A/k_c)
Gamma_A  = 1
Gamma_B  = D_B/D_A
Lambda_A = nu_S*kappa_A/(l_scale*l_scale)
Lambda_B = nu_S*kappa_B/(l_scale*l_scale)

np.random.seed(1)
phiaI  = 0.30
phibI  = 0.01

#set dilute phase
phiA = phiaI*(1 + 0.00001*np.random.rand(n,n))
phiB = phibI*(1 + 0.00001*np.random.rand(n,n))

Atot = phiaI
Btot = phibI

phiA = rfft2(phiA)
phiB = rfft2(phiB)

fuel   = 4
thresh = 0.5
steep  = 5*1e-4
k_p    = 1 #in units of k_c

# Initialize wavelength for second derivative to avoid a repetitive operation
k_x, k_y = np.meshgrid(rfftfreq(n, h), fftfreq(n, h))
k2 = k_x**2 + k_y**2

u_val = -0.4
U     = np.zeros((n, n))
mid   = int(n/2)

U[mid, mid]     = u_val
U[mid, mid+1]   = u_val/2
U[mid+1, mid]   = u_val/2
U[mid+1, mid+1] = u_val/2
U[mid-1, mid]   = u_val/2

sol = implicitrk(phiA, phiB, delta, thresh, steep, fuel, k_p, omegaA, omegaB, omegaS, rho_A, rho_B, n_step, Gamma_A, Gamma_B, Lambda_A, Lambda_B, k_x, k_y, k2, U, u_val, l)

#Initialize animation\
for i in range(n_step):
    next(sol)
   