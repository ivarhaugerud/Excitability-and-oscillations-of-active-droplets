import scipy.spatial  as scs 
import scipy.optimize as sco
import scipy.signal   as sci
import numpy as np
import matplotlib.pyplot as plt

def calc_gibbs_mu(phi, k_bT, nu, omega, xi):
    phi = np.append(phi, 1-np.sum(phi))
    return k_bT*(np.log(phi)   +1) + omega    + xi.dot(phi) - 0.5*nu*np.dot(phi, np.dot(xi, phi)) - k_bT*nu*np.sum(phi/nu)

def bino_intersect_chemindx(chem_eq, bino):
    dist = np.zeros(len(chem_eq[:, 0]))
    for i in range(len(dist)):
        dist[i] = np.amin(np.square(chem_eq[i, 0] - bino[:, 0]) + np.square(chem_eq[i, 1] - bino[:, 1]))
    indx = np.argmin(dist)
    return indx

def bino_intersect_binoindx(chem_eq, bino):
    dist = np.zeros(len(bino[:, 0]))
    for i in range(len(dist)):
        dist[i] = np.amin(np.square(chem_eq[:, 0] - bino[i, 0]) + np.square(chem_eq[:, 1] - bino[i, 1]))
    indx = np.argmin(dist)
    return indx


def fuel_foo(phi_a, phi_b, thresh, kappa, delta):
    return (np.arctan( (phi_a+phi_b - thresh)/kappa) + np.pi/2)*(delta/np.pi)

def check_inside(phi_a_bar, phi_b_bar, P, bot, top):
    if phi_a_bar > max(P):
        inside = False
    elif phi_a_bar < min(P):
        inside = False
    elif phi_b_bar > top[np.argmin(abs(phi_a_bar-P))]:
        inside = False
    elif phi_b_bar > bot[np.argmin(abs(phi_a_bar-P))]:
        inside = True
    elif phi_b_bar < bot[np.argmin(abs(phi_a_bar-P))]:
        inside = False
    else:
        print("HERE", phi_a_bar, phi_b_bar)
    return inside

def check_inside_type2ab(phi_a_bar, phi_b_bar, bino):
    if phi_a_bar > np.amax([np.amax(bino[:, 0]), np.amax(bino[:, 2])]):
        inside = False
    elif phi_a_bar < np.amin([np.amin(bino[:, 0]), np.amin(bino[:, 2])]):
        inside = False
    elif phi_b_bar > bino[np.argmin(abs(bino[:, 2] - phi_a_bar)), 3]:
        inside = False
    elif phi_b_bar > bino[np.argmin(abs(bino[:, 0] - phi_a_bar)), 1]:
        inside = True
    elif phi_b_bar < bino[np.argmin(abs(bino[:, 0] - phi_a_bar)), 1]:
        inside = False
    else:
        print("HERE", phi_a_bar, phi_b_bar)
    return inside

def contour_lines_ab(phi, nu, k_bT, omega, xi, bot, top, bino):
    chem_equil  = np.zeros((len(phi), len(phi)))
    distances   = np.zeros(len(bino[:, 0]))
    
    for i in range(len(phi)):
        for j in range(len(phi)):
            if phi[i] + phi[j] < 1:
                if True:#not check_inside_type2ab(phi[i], phi[j], bino):#True:
                    temp             = calc_gibbs_mu(np.array([phi[i], phi[j]]), k_bT, nu, omega, xi)
                    chem_equil[j, i] = (temp[0]-temp[1])/k_bT
                else:
                    for k in range(len(bino[:, 0])):
                        beta  = (bino[k, 3] - bino[k, 1])/(bino[k, 2]-bino[k, 0]+1e-6)
                        alpha =  bino[k, 3] - beta*bino[k, 2]
                        distances[k] = np.amin(np.square(phi-phi[i]) + np.square(alpha+phi*beta-phi[j]))

                    tiline_index     = np.argmin(distances)            
                    temp             = calc_gibbs_mu(np.array([bino[tiline_index, 0], bino[tiline_index, 1]]), k_bT, nu, omega, xi)
                    chem_equil[j, i] = (temp[0]-temp[1])/k_bT

    cp       = plt.contour(phi, phi, chem_equil, levels=np.zeros(1))
    contour  = cp.collections[0]
    chem_eq  = contour.get_paths()[0].vertices
    return chem_eq

def contour_lines_ab_interval(phi_a, phi_b, nu, k_bT, omega, xi, bot, top, bino, thresh, kappa, delta):
    chem_equil  = np.zeros((len(phi_b), len(phi_a)))
    
    for i in range(len(phi_a)):
        for j in range(len(phi_b)):
            if phi_a[i] + phi_b[j] < 1:
                delta_val        = fuel_foo(phi_a[i], phi_b[j], thresh, kappa, delta)
                temp             = calc_gibbs_mu(np.array([phi_a[i], phi_b[j]]), k_bT, nu, omega, xi)
                chem_equil[j, i] = (temp[0]+delta_val-temp[1])/k_bT

    cp       = plt.contour(phi_a, phi_b, chem_equil, levels=np.zeros(1))
    contour  = cp.collections[0]
    chem_eq  = contour.get_paths()[0].vertices
    return chem_eq

def contour_lines_ab_fuel(phi, nu, k_bT, omega, xi, bot, top, bino, thresh, kappa, delta):
    chem_equil  = np.zeros((len(phi), len(phi)))
    distances   = np.zeros(len(bino[:, 0]))
    
    for i in range(len(phi)):
        for j in range(len(phi)):
            if phi[i] + phi[j] < 1:
                if True:#not check_inside_type2ab(phi[i], phi[j], bino):#True:
                    delta_val        = fuel_foo(phi[i], phi[j], thresh, kappa, delta)
                    temp             = calc_gibbs_mu(np.array([phi[i], phi[j]]), k_bT, nu, omega, xi)
                    chem_equil[j, i] = (temp[0]+delta_val-temp[1])/k_bT
                else:
                    for k in range(len(bino[:, 0])):
                        beta  = (bino[k, 3] - bino[k, 1])/(bino[k, 2]-bino[k, 0]+1e-6)
                        alpha =  bino[k, 3] - beta*bino[k, 2]
                        distances[k] = np.amin(np.square(phi-phi[i]) + np.square(alpha+phi*beta-phi[j]))

                    tiline_index     = np.argmin(distances)            
                    temp             = calc_gibbs_mu(np.array([bino[tiline_index, 0], bino[tiline_index, 1]]), k_bT, nu, omega, xi)
                    chem_equil[j, i] = (temp[0]-temp[1])/k_bT

    cp       = plt.contour(phi, phi, chem_equil, levels=np.zeros(1))
    contour  = cp.collections[0]
    chem_eq  = contour.get_paths()[0].vertices
    return chem_eq

def contour_lines_ab_psi(phi, nu, k_bT, omega, xi, bot, top, bino):
    cons_equil  = np.zeros((len(phi), len(phi)))
    chem_equil  = np.zeros((len(phi), len(phi)))
    distances   = np.zeros(len(bino[:, 0]))
    for i in range(len(phi)):
        for j in range(len(phi)):
            if phi[i] + phi[j] < 1:
                if True:#not check_inside_type2ab(phi[i], phi[j], bino):#True:
                    temp             = calc_gibbs_mu(np.array([phi[i], phi[j]]), k_bT, nu, omega, xi)
                    chem_equil[j, i] = (temp[0]-temp[1])/k_bT
                    cons_equil[j, i] = (temp[0]+temp[1]-2*temp[2])/k_bT
                else:
                    for k in range(len(bino[:, 0])):
                        beta  = (bino[k, 3] - bino[k, 1])/(bino[k, 2]-bino[k, 0]+1e-6)
                        alpha =  bino[k, 3] - beta*bino[k, 2]
                        distances[k] = np.amin(np.square(phi-phi[i]) + np.square(alpha+phi*beta-phi[j]))

                    tiline_index     = np.argmin(distances)            
                    temp             = calc_gibbs_mu(np.array([bino[tiline_index, 0], bino[tiline_index, 1]]), k_bT, nu, omega, xi)
                    chem_equil[j, i] = (temp[0]-temp[1])/k_bT
                    cons_equil[j, i] = (temp[0]+temp[1]-2*temp[2])/k_bT

    cp       = plt.contour(phi, phi, chem_equil, levels=np.zeros(1))
    contour  = cp.collections[0]
    chem_eq  = contour.get_paths()[0].vertices

    cp       = plt.contour(phi, phi, cons_equil, levels=np.zeros(1))
    contour  = cp.collections[0]
    cons_eq  = contour.get_paths()[0].vertices
    return chem_eq, cons_eq


def get_flows(nu, k_bT, omega, xi, bot, top, bino, phiX, phiY, thresh, kappa, delta_max, k_c, k_p):
    vectors  = np.zeros((len(phiX),len(phiY), 2))

    phi = np.linspace(0, 1, int(1e4))
    distances = np.zeros(len(bino[:, 0]))

    for i in range(len(phiX)):
        for j in range(len(phiY)):
            if phiX[i] + phiY[j] < 1:
                if not check_inside_type2ab(phiX[i], phiY[j], bino):
                    temp             = calc_gibbs_mu(np.array([phiX[i], phiY[j]]), k_bT, nu, omega, xi)
                    delta            = fuel_foo(phiX[i], phiY[j], thresh, kappa, delta_max)

                    r = k_c*(np.exp(temp[0]+delta)-np.exp(temp[1]))
                    h = k_p*(np.exp(temp[0]+temp[1])-np.exp(2*temp[2]))
                    vectors[i, j, 0] = -r - h 
                    vectors[i, j, 1] =  r - h 

                else:
                    for k in range(len(bino[:, 0])):
                        beta  = (bino[k, 3] - bino[k, 1])/(bino[k, 2]-bino[k, 0]+1e-6)
                        alpha =  bino[k, 3] - beta*bino[k, 2]
                        distances[k] = np.amin(np.square(phi-phiX[i]) + np.square(alpha+phi*beta-phiY[j]))

                    tiline_index     = np.argmin(distances)            

                    temp             = calc_gibbs_mu(np.array([bino[tiline_index, 0], bino[tiline_index, 1]]), k_bT, nu, omega, xi)
                    delta1 = fuel_foo(bino[tiline_index, 0], bino[tiline_index, 1], thresh, kappa, delta_max)
                    delta2 = fuel_foo(bino[tiline_index, 2], bino[tiline_index, 3], thresh, kappa, delta_max)

                    rI  = k_c*(np.exp(temp[0]+delta1)-np.exp(temp[1]))
                    rII = k_c*(np.exp(temp[0]+delta2)-np.exp(temp[1]))
                    h = k_p*(np.exp(temp[0]+temp[1])-np.exp(2*temp[2]))
                    VI = (phiX[i] - bino[tiline_index, 2])/(bino[tiline_index, 0] - bino[tiline_index, 2])
                    vectors[i, j, 0] = -rI*VI - rII*(1-VI) - h 
                    vectors[i, j, 1] =  rI*VI + rII*(1-VI) - h 

    return vectors

def run_system_nobino(T, phi_bar, V, k_bT, xi, mu_0r, nu, omega, evap_coeff, react_coeff):
    R = np.zeros(len(T))

    for i in range(len(T)-1):
        dt = T[i+1]-T[i]
        mu2         = calc_gibbs_mu(phi_bar[i,  :], k_bT, nu, omega, xi)
        Ndot        = np.array([0, 0, evap_coeff*(np.exp(mu_0r[i]/k_bT) - np.exp(mu2[-1]/k_bT)) ])
        Vdot        = np.sum(nu*Ndot)
        react       = react_coeff*(-np.exp(mu2[0]/k_bT) + np.exp(mu2[1]/k_bT))
        r           = np.array([react, -react, 0])
        
        phi_bar[i+1, :]  = phi_bar[i, :] + dt*(r[:-1] - Vdot*phi_bar[i, :]/V[i])
        V[i+1]           = V[i] + dt*Vdot
        R[i+1]           = np.abs(react)

    return phi_bar, V, R


def osmotic_pressure( phi, mu, k_bT, nu, xi_ab, xi_ac, xi_bc, omega):
    return -f(phi, k_bT, nu, xi_ab, xi_ac, xi_bc, omega) + phi[0]*mu[0]/nu[0] + phi[1]*mu[1]/nu[1]

def solve_TD_equations(bin_pts, phi_aver, previous_frac, k_bT, nu, xi_ab, xi_ac, xi_bc, omega):
    guess      = np.array([bin_pts[0], bin_pts[1], previous_frac])                                                  # use the previous found state with new updated average volume fractions
    sol        = sco.fsolve(func=TD_equations, x0=guess, args=(phi_aver, k_bT, nu, xi_ab, xi_ac, xi_bc, omega), xtol=1e-5)#, maxfev=int(1e3), factor=0.06, epsfcn=0.01)   # found the zeros of the thermodynamic equations through newtons method 
    phi_II     = (phi_aver - sol[-1] * sol[:-1])/(1.-sol[-1])                                                       # from the volume fractions in phase 1, calculate volume fractions in phase two for the found relative volume

    return np.array([sol[0], sol[1], phi_II[0], phi_II[1]]), sol[-1]

def calculate_binodal(two_comp_sol, end_average, N, k_bT, nu, xi_ab, xi_ac, xi_bc, omega, step_border_ratio = 10):
    start_point = np.array([np.mean(two_comp_sol), 1e-5])    # where the average volume fraction starts
    phi_bar = np.transpose(np.array([np.linspace(start_point[0], end_average[0], N), np.linspace(start_point[1], end_average[1], N)]))

    binodal_points = np.ones((N, 4))/(N*step_border_ratio) # if we guess zero we get infinities
    binodal_points[0, ::2] = two_comp_sol                  # use the two-component solution as initiall guess
    V1_frac = 0.5

    for n in range(N-1):
        binodal_points[n+1], V1_frac = solve_TD_equations(binodal_points[n], phi_bar[n], V1_frac, k_bT, nu, xi_ab, xi_ac, xi_bc, omega)
        if abs(binodal_points[n+1, 0])+abs(binodal_points[n+1, 1]) > 1 or abs(binodal_points[n+1, 2])+abs(binodal_points[n+1, 3]) > 1:
            break
    return binodal_points 

def f(phi, k_bT, nu, xi_ab, xi_ac, xi_bc, omega):
    return k_bT*(phi[0]*np.log(phi[0])/nu[0] + phi[1]*np.log(phi[1])/nu[1] + (1-phi[0]-phi[1])*np.log(1-phi[0]-phi[1])/nu[2]) + phi[0]*phi[1]*xi_ab + phi[0]*(1-phi[0]-phi[1])*xi_ac + phi[1]*(1-phi[0]-phi[1])*xi_bc + omega[0]*phi[0] + omega[1]*phi[1] + omega[2]*(1-phi[0]-phi[1])


def mu_a_and_b(phi, k_bT, nu, xi_ab, xi_ac, xi_bc, omega):
    mua = k_bT*(1 + np.log(phi[0]) - nu[0]*(1 + np.log(1-phi[0]-phi[1]))/nu[2]) + nu[0]*( phi[1]*xi_ab + xi_ac*(1-2*phi[0]-phi[1]) - xi_bc*phi[1]) + omega[0] - omega[2]
    mub = k_bT*(1 + np.log(phi[1]) - nu[1]*(1 + np.log(1-phi[0]-phi[1]))/nu[2]) + nu[1]*( phi[0]*xi_ab + xi_bc*(1-2*phi[1]-phi[0]) - xi_ac*phi[0]) + omega[1] - omega[2]

    return np.array([mua, mub])

def TD_equations(state, phi_aver, k_bT, nu, xi_ab, xi_ac, xi_bc, omega):
    phi_I   = state[:-1]
    frac_VI = state[-1]
    
    phi_II = (phi_aver - frac_VI * phi_I)/(1.-frac_VI)                         # given the V1-volume, the dilute phase is determined
    mu_I  = mu_a_and_b(phi_I,   k_bT, nu, xi_ab, xi_ac, xi_bc, omega)          # calculate chemical potential in dense phase
    mu_II = mu_a_and_b(phi_II , k_bT, nu, xi_ab, xi_ac, xi_bc, omega)          # calculate chemical potential in dilute phase
    
    # for correct volume fractions this will return zero
    if np.sqrt(np.square(phi_I[0]-phi_II[0]) + np.square(phi_I[1] - phi_II[1])) < 0.03:
        return 10*np.ones(3)
    return np.array([mu_I[0]-mu_II[0], mu_I[1]-mu_II[1], osmotic_pressure(phi_I, mu_I,k_bT, nu, xi_ab, xi_ac, xi_bc, omega) - osmotic_pressure(phi_II, mu_II, k_bT, nu, xi_ab, xi_ac, xi_bc, omega)])

def get_tieline_twocomp(phi, xi, nu1, nu2, k_bT, omega1, omega2):
    f    = k_bT*(phi*np.log(phi)/nu1 + (1-phi)*np.log(1-phi)/nu2) + phi*(1-phi)*xi #+ omega1*phi + (1-phi)*omega2
    
    f_tilde       = np.zeros(len(f)+2)
    f_tilde[1:-1] = f
    f_tilde[0]    = max(f)+10
    f_tilde[-1]   = max(f)+10

    phi_tilde       = np.zeros(len(f)+2)
    phi_tilde[1:-1] = phi
    phi_tilde[0]    = phi[0]
    phi_tilde[-1]   = phi[-1]

    CH   = scs.ConvexHull( np.transpose(np.array([phi_tilde, f_tilde])), incremental=True)
    vert = CH.vertices[2:]-2
    phi  = (CH.points[2:, 0])[vert]
    indx = np.argmax(np.diff(vert))
    return np.array([phi[indx], phi[indx+1]])

def fill_between_arraysAB(bino):
    start = np.argmin(bino[:, 0])
    end   = np.argmax(bino[:, 2])
    phi   = np.linspace(bino[start, 0], bino[end, 2], 5000)

    top = []
    bot = []
    top_array_y = np.concatenate(( bino[start:, 1], np.flip(bino[end:, 3], axis=0)))

    bot_array_y = np.concatenate(( np.flip(bino[:start, 1], axis=0), np.zeros(len(top_array_y)-start-end), bino[:end, 3]))

    top_array_x = np.concatenate(( bino[start:, 0], np.flip(bino[end:, 2], axis=0)))
    bot_array_x = np.concatenate(( np.flip(bino[:start, 0], axis=0), np.linspace(bino[0, 0], bino[0, 2], len(top_array_y)-start-end), bino[:end, 2]))

    for i in range(len(phi)):
        bot.append(bot_array_y[np.argmin(abs(phi[i]-bot_array_x))])
        top.append(top_array_y[np.argmin(abs(phi[i]-top_array_x))])

    return phi, np.array(bot), np.array(top)

def fill_between_arrays_type2(bino):
    start = np.argmin(bino[:, 0])
    end   = np.argmax(bino[:, 2])
    phi   = np.linspace(bino[start, 0], bino[end, 2], 5000)
    top, bot = np.zeros(len(phi)), np.zeros(len(phi))

    lower_dense = bino[:end, 2:]
    upper_dense = bino[end:, 2:]

    for i in range(len(phi)):
        if phi[i] < np.amax(bino[:, 0]):
            top[i] = bino[np.argmin(abs(phi[i]-bino[:, 2])), 3]
            bot[i] = bino[np.argmin(abs(phi[i]-bino[:, 0])), 1]
        else:
            top[i] = upper_dense[np.argmin(abs(phi[i]-upper_dense[:, 0])), 1]
            bot[i] = lower_dense[np.argmin(abs(phi[i]-lower_dense[:, 0])), 1]
    return phi, bot, top

def fill_between_arrays_type2_less_overlap(bino):
    start = np.argmin(bino[:, 0])
    end   = np.argmax(bino[:, 2])
    phi   = np.linspace(bino[start, 0], bino[end, 2], 5000)
    top, bot = np.zeros(len(phi)), np.zeros(len(phi))

    for i in range(len(phi)):
        if phi[i] < np.amax(bino[:, 0]):
            top[i] = bino[np.argmin(abs(phi[i]-bino[:, 2])), 3]
            bot[i] = bino[np.argmin(abs(phi[i]-bino[:, 0])), 1]
        else:
            top[i] = bino[np.argmin(abs(phi[i]-bino[:, 2])), 3]
            bot[i] = 0
    return phi, bot, top

def fill_between_arrays_type2_rotated(bino):
    phi = np.concatenate((bino[:, 0], bino[::-1, 2]))
    top = np.concatenate((bino[:, 1], bino[::-1, 3]))
    bot = -phi
    return phi, bot, top

def get_binodal(phi, nu, omega, k_bT, xi_ab, xi_ac, xi_bc):
    tiline_two_component = get_tieline_twocomp(phi, xi_ac, nu[1], nu[2], k_bT, omega[1], omega[2])
    binodal              = calculate_binodal(tiline_two_component, np.array([0.47, 0.54]), 5000, k_bT, nu, xi_ab, xi_ac, xi_bc, omega)
    indices = np.where(np.square(binodal[:,0]-binodal[:,2]) + np.square(binodal[:,1]-binodal[:,3]) < 1e-7)[0][0]
    binodal = binodal[:indices, :]
    closest_point = np.argmin(np.square(binodal[-1, 0]-binodal[:, 2]) + np.square(binodal[-1, 1]-binodal[:, 3]))

    binodal = binodal[:closest_point, :]
    addons  = 25
    bino    = np.zeros((closest_point+addons, 4))

    bino[:closest_point, :]   = binodal[:,:]
    bino[closest_point:,  0]  = np.linspace(binodal[-1, 0], (binodal[-1, 0]+binodal[-1, 2])/2, addons)
    bino[closest_point:,  1]  = np.linspace(binodal[-1, 1], (binodal[-1, 1]+binodal[-1, 3])/2, addons)
    bino[closest_point:,  2]  = np.linspace(binodal[-1, 2], (binodal[-1, 0]+binodal[-1, 0])/2, addons)
    bino[closest_point:,  3]  = np.linspace(binodal[-1, 3], (binodal[-1, 1]+binodal[-1, 1])/2, addons)
    return bino



def run_system(T, phi_bar, phiI, phiII, V, VI, VII, k_bT, xi_ab, xi_ac, xi_bc, xi, nu, omega, k_p, k_c, bino, P, bot, top, thresh, kappa, delta):
    b      = np.zeros(3)
    A      = np.zeros((3, 3))
    j1     = np.zeros(3)
    j2     = np.zeros(3)
    distances = np.zeros(len(bino[:, 0]))
    x = np.linspace(0, 1, int(1e5))
    F = np.zeros(len(T))
    chems = np.zeros((len(T), 5))
    rates = np.zeros((len(T), 2))

    inside      = False
    prev_inside = False
    
    for i in range(len(T)-1):
        dt = T[i+1]-T[i]
        prev_inside = inside
        inside = check_inside(phi_bar[i, 0], phi_bar[i, 1], P, bot, top)

        if inside and not prev_inside:
            for j in range(len(bino[:,0])):
                a = (bino[j, 3]-bino[j, 1])/(bino[j, 2]-bino[j, 0])
                B = bino[j, 3] - a*bino[j, 2]
                dist = np.amin(np.square(x-phi_bar[i, 0]) + np.square(a*x+B-phi_bar[i, 1]))
                
                if dist != dist:
                    distances[j] = 100
                else:
                    distances[j] = dist

            tiline_index     = np.argmin(distances)

            phiI[i,  :-1] = np.array([bino[tiline_index, 0], bino[tiline_index, 1]])
            phiII[i, :-1] = np.array([bino[tiline_index, 2], bino[tiline_index, 3]])

            phiI[i,  -1]   = 1-np.sum(phiI[i,  :-1])
            phiII[i, -1]   = 1-np.sum(phiII[i, :-1])
            VI[i]       = V*(phi_bar[i, 0] - phiII[i, 0])/(phiI[i, 0]-phiII[i, 0])
            VII[i]      = V-VI[i]

        if inside:
            V1  = VI[i]
            V2  = VII[i]
            phi_a1 = phiI[i, 0]
            phi_b1 = phiI[i, 1]
            phi_c1 = 1-phi_a1-phi_b1
            phi_a2 = phiII[i, 0]
            phi_b2 = phiII[i, 1]
            phi_c2 = 1-phi_a2-phi_b2

            mu1    = calc_gibbs_mu(phiI[i,  :-1],  k_bT, nu, omega, xi)

            h1 = np.zeros(3)
            h2 = np.zeros(3)

            delta1 = fuel_foo(phiI[i,  0], phiI[i,  1], thresh, kappa, delta)
            delta2 = fuel_foo(phiII[i, 0], phiII[i, 1], thresh, kappa, delta)

            chems[i+1, 0] = mu1[0]
            chems[i+1, 1] = mu1[1]
            chems[i+1, 2] = mu1[2]
            chems[i+1, 3] = delta1
            chems[i+1, 4] = delta2

            react1       = k_c*(-np.exp((mu1[0]+delta1)/k_bT) + np.exp(mu1[1]/k_bT))
            react2       = k_c*(-np.exp((mu1[0]+delta2)/k_bT) + np.exp(mu1[1]/k_bT))

            psi1    = k_p*(np.exp((mu1[0]+mu1[1])/k_bT) - np.exp(2*mu1[2]/k_bT))
            psi2    = k_p*(np.exp((mu1[0]+mu1[1])/k_bT) - np.exp(2*mu1[2]/k_bT))

            r_1          = np.array([react1-psi1, -react1-psi1, 2*psi1])
            r_2          = np.array([react2-psi2, -react2-psi2, 2*psi2])
            rates[i, 0] = V1*react1 + V2*react2
            rates[i, 1] = V1*psi1 + V2*psi2

            #PHASE 1
            dmua_dphia1 = k_bT/phi_a1 - k_bT - nu[0]*(xi_ab*phi_b1 + xi_ac*phi_c1)
            dmua_dphib1 = xi_ab       - k_bT - nu[0]*(xi_ab*phi_a1 + xi_bc*phi_c1)
            dmua_dphic1 = xi_ac       - k_bT - nu[0]*(xi_ac*phi_a1 + xi_bc*phi_b1)

            dmub_dphia1 = xi_ab       - k_bT - nu[1]*(xi_ab*phi_b1 + xi_ac*phi_c1)
            dmub_dphib1 = k_bT/phi_b1 - k_bT - nu[1]*(xi_ab*phi_a1 + xi_bc*phi_c1)
            dmub_dphic1 = xi_bc       - k_bT - nu[1]*(xi_ac*phi_a1 + xi_bc*phi_b1)

            dmuc_dphia1 = xi_ac       - k_bT - nu[2]*(xi_ab*phi_b1 + xi_ac*phi_c1)
            dmuc_dphib1 = xi_bc       - k_bT - nu[2]*(xi_ab*phi_a1 + xi_bc*phi_c1)
            dmuc_dphic1 = k_bT/phi_c1 - k_bT - nu[2]*(xi_ac*phi_a1 + xi_bc*phi_b1)

            #PHASE 2
            dmua_dphia2 = k_bT/phi_a2 - k_bT  - nu[0]*(xi_ab*phi_b2 + xi_ac*phi_c2)
            dmua_dphib2 = xi_ab       - k_bT  - nu[0]*(xi_ab*phi_a2 + xi_bc*phi_c2)
            dmua_dphic2 = xi_ac       - k_bT  - nu[0]*(xi_ac*phi_a2 + xi_bc*phi_b2)

            dmub_dphia2 = xi_ab       - k_bT  - nu[1]*(xi_ab*phi_b2 + xi_ac*phi_c2)
            dmub_dphib2 = k_bT/phi_b2 - k_bT  - nu[1]*(xi_ab*phi_a2 + xi_bc*phi_c2)
            dmub_dphic2 = xi_bc       - k_bT  - nu[1]*(xi_ac*phi_a2 + xi_bc*phi_b2)

            dmuc_dphia2 = xi_ac       - k_bT  - nu[2]*(xi_ab*phi_b2 + xi_ac*phi_c2)
            dmuc_dphib2 = xi_bc       - k_bT  - nu[2]*(xi_ab*phi_a2 + xi_bc*phi_c2)
            dmuc_dphic2 = k_bT/phi_c2 - k_bT  - nu[2]*(xi_ac*phi_a2 + xi_bc*phi_b2)

            #CALCULATE MATRIX
            A[0, 0] = nu[0]*( dmua_dphia1*(phi_a1-1) + phi_b1*dmua_dphib1 + phi_c1*dmua_dphic1 + (V1/V2)*(dmua_dphia2*(phi_a2-1) + phi_b2*dmua_dphib2 + phi_c2*dmua_dphic2))
            A[0, 1] = nu[1]*( dmua_dphib1*(phi_b1-1) + phi_a1*dmua_dphia1 + phi_c1*dmua_dphic1 + (V1/V2)*(dmua_dphib2*(phi_b2-1) + phi_a2*dmua_dphia2 + phi_c2*dmua_dphic2))
            A[0, 2] = nu[2]*( dmua_dphic1*(phi_c1-1) + phi_a1*dmua_dphia1 + phi_b1*dmua_dphib1 + (V1/V2)*(dmua_dphic2*(phi_c2-1) + phi_a2*dmua_dphia2 + phi_b2*dmua_dphib2))
            
            b[0]    = r_2[0]*nu[0]*dmua_dphia2 + r_2[1]*nu[1]*dmua_dphib2 + r_2[2]*nu[2]*dmua_dphic2
            b[0]   -= r_1[0]*nu[0]*dmua_dphia1 + r_1[1]*nu[1]*dmua_dphib1 + r_1[2]*nu[2]*dmua_dphic1

            A[1, 0] = nu[0]*( dmub_dphia1*(phi_a1-1) + phi_b1*dmub_dphib1 + phi_c1*dmub_dphic1 + (V1/V2)*(dmub_dphia2*(phi_a2-1) + phi_b2*dmub_dphib2 + phi_c2*dmub_dphic2))
            A[1, 1] = nu[1]*( dmub_dphib1*(phi_b1-1) + phi_a1*dmub_dphia1 + phi_c1*dmub_dphic1 + (V1/V2)*(dmub_dphib2*(phi_b2-1) + phi_a2*dmub_dphia2 + phi_c2*dmub_dphic2))
            A[1, 2] = nu[2]*( dmub_dphic1*(phi_c1-1) + phi_a1*dmub_dphia1 + phi_b1*dmub_dphib1 + (V1/V2)*(dmub_dphic2*(phi_c2-1) + phi_a2*dmub_dphia2 + phi_b2*dmub_dphib2))
            b[1]    = r_2[0]*nu[0]*dmub_dphia2 + r_2[1]*nu[1]*dmub_dphib2 + r_2[2]*nu[2]*dmub_dphic2
            b[1]   -= r_1[0]*nu[0]*dmub_dphia1 + r_1[1]*nu[1]*dmub_dphib1 + r_1[2]*nu[2]*dmub_dphic1

            A[2, 0] = nu[0]*( dmuc_dphia1*(phi_a1-1) + phi_b1*dmuc_dphib1 + phi_c1*dmuc_dphic1 + (V1/V2)*(dmuc_dphia2*(phi_a2-1) + phi_b2*dmuc_dphib2 + phi_c2*dmuc_dphic2))
            A[2, 1] = nu[1]*( dmuc_dphib1*(phi_b1-1) + phi_a1*dmuc_dphia1 + phi_c1*dmuc_dphic1 + (V1/V2)*(dmuc_dphib2*(phi_b2-1) + phi_a2*dmuc_dphia2 + phi_c2*dmuc_dphic2))
            A[2, 2] = nu[2]*( dmuc_dphic1*(phi_c1-1) + phi_a1*dmuc_dphia1 + phi_b1*dmuc_dphib1 + (V1/V2)*(dmuc_dphic2*(phi_c2-1) + phi_a2*dmuc_dphia2 + phi_b2*dmuc_dphib2))
            b[2]    = r_2[0]*nu[0]*dmuc_dphia2 + r_2[1]*nu[1]*dmuc_dphib2 + r_2[2]*nu[2]*dmuc_dphic2
            b[2]   -= r_1[0]*nu[0]*dmuc_dphia1 + r_1[1]*nu[1]*dmuc_dphib1 + r_1[2]*nu[2]*dmuc_dphic1

            j1     =  np.linalg.solve(A, b)
            j2     = -j1*V1/V2 

            V1_rel_dot = -np.sum(nu*j1)
            V2_rel_dot = -np.sum(nu*j2)

            VI[i+1]  = VI[i]  + dt*V1*V1_rel_dot
            VII[i+1] = VII[i] + dt*V2*V2_rel_dot

            phiI[i+1,  :]     =  phiI[i, :]   + nu     *dt*(r_1 - j1 - phiI[i, :]*V1_rel_dot) 
            phiII[i+1, :]     = phiII[i, :]   + nu     *dt*(r_2 - j2 - phiII[i,:]*V2_rel_dot)
            phi_bar[i+1,  :]  = phi_bar[i, :] + nu[:-1]*dt*((VI[i]*r_1[:-1] + VII[i]*r_2[:-1])/V)
            F[i+1] = VI[i+1]*f(phiI[i+1, :], k_bT, nu, xi_ab, xi_ac, xi_bc, omega) + VII[i+1]*f(phiII[i+1, :], k_bT, nu, xi_ab, xi_ac, xi_bc, omega)

        else:
            mu          = calc_gibbs_mu(phi_bar[i,  :], k_bT, nu, omega, xi)

            delta0      = fuel_foo(phi_bar[i,  0], phi_bar[i,  1], thresh, kappa, delta)
            react       = k_c*(-np.exp((mu[0]+delta0)/k_bT) + np.exp(mu[1]/k_bT))
            psi         = k_p*(np.exp((mu[0]+mu[1])/k_bT) - np.exp(2*mu[2]/k_bT))
            r           = np.array([react-psi, -react-psi, 2*psi])
            
            chems[i+1, 0] = mu[0]
            chems[i+1, 1] = mu[1]
            chems[i+1, 2] = mu[2]
            chems[i+1, 3] = delta0
            chems[i+1, 4] = delta0
            rates[i, 0] = react
            rates[i, 1] = psi

            phi_bar[i+1, :]  =  phi_bar[i, :] + nu[:-1]*dt*r[:-1]
            F[i+1]           = f(phi_bar[i+1, :], k_bT, nu, xi_ab, xi_ac, xi_bc, omega)

        if (i+1) % 10000 == 0:
            tau, done =  get_period(T[:i+1], phi_bar[:i+1, 0], phi_bar[:i+1, 1], 1e-5)
            if done:
                print("DONE", tau)
                break
    return phi_bar[:i+1], phiI[:i+1], phiII[:i+1], VI[:i+1], VII[:i+1], F[:i+1], chems[:i+1, :], rates[:i+1, :], T[:i+1]
    
def run_system_nocycle(T, phi_bar, phiI, phiII, V, VI, VII, k_bT, xi_ab, xi_ac, xi_bc, xi, nu, omega, k_p, k_c, bino, thresh, kappa, delta):
    b      = np.zeros(3)
    A      = np.zeros((3, 3))
    j1     = np.zeros(3)
    j2     = np.zeros(3)
    distances = np.zeros(len(bino[:, 0]))
    x = np.linspace(0, 1, int(1e5))
    F = np.zeros(len(T))
    chems = np.zeros((len(T), 5))

    inside      = False
    prev_inside = False
    
    for i in range(len(T)-1):
        dt = T[i+1]-T[i]
        prev_inside = inside
        inside = check_inside_type2ab(phi_bar[i, 0], phi_bar[i, 1], bino)

        if inside and not prev_inside:
            for j in range(len(bino[:,0])):
                a = (bino[j, 3]-bino[j, 1])/(bino[j, 2]-bino[j, 0])
                B = bino[j, 3] - a*bino[j, 2]
                dist = np.amin(np.square(x-phi_bar[i, 0]) + np.square(a*x+B-phi_bar[i, 1]))
                
                if dist != dist:
                    distances[j] = 100
                else:
                    distances[j] = dist

            tiline_index     = np.argmin(distances)

            phiI[i,  :-1] = np.array([bino[tiline_index, 0], bino[tiline_index, 1]])
            phiII[i, :-1] = np.array([bino[tiline_index, 2], bino[tiline_index, 3]])

            phiI[i,  -1]   = 1-np.sum(phiI[i,  :-1])
            phiII[i, -1]   = 1-np.sum(phiII[i, :-1])
            VI[i]       = V*(phi_bar[i, 0] - phiII[i, 0])/(phiI[i, 0]-phiII[i, 0])
            VII[i]      = V-VI[i]

        if inside:
            V1  = VI[i]
            V2  = VII[i]
            phi_a1 = phiI[i, 0]
            phi_b1 = phiI[i, 1]
            phi_c1 = 1-phi_a1-phi_b1
            phi_a2 = phiII[i, 0]
            phi_b2 = phiII[i, 1]
            phi_c2 = 1-phi_a2-phi_b2

            mu1    = calc_gibbs_mu(phiI[i,  :-1],  k_bT, nu, omega, xi)

            h1 = np.zeros(3)
            h2 = np.zeros(3)

            delta1 = fuel_foo(phiI[i,  0], phiI[i,  1], thresh, kappa, delta)
            delta2 = fuel_foo(phiII[i, 0], phiII[i, 1], thresh, kappa, delta)

            chems[i+1, 0] = mu1[0]
            chems[i+1, 1] = mu1[1]
            chems[i+1, 2] = mu1[2]
            chems[i+1, 3] = delta1
            chems[i+1, 4] = delta2

            react1       = k_c*(-np.exp((mu1[0]+delta1)/k_bT) + np.exp(mu1[1]/k_bT))
            react2       = k_c*(-np.exp((mu1[0]+delta2)/k_bT) + np.exp(mu1[1]/k_bT))
            react        = V1*react1 + V2*react2

            psi1    = k_p*(np.exp((mu1[0]+mu1[1])/k_bT) - np.exp(2*mu1[2]/k_bT))
            psi2    = k_p*(np.exp((mu1[0]+mu1[1])/k_bT) - np.exp(2*mu1[2]/k_bT))

            r_1          = np.array([react1-psi1, -react1-psi1, 2*psi1])
            r_2          = np.array([react2-psi2, -react2-psi2, 2*psi2])


            #PHASE 1
            dmua_dphia1 = k_bT/phi_a1 - k_bT - nu[0]*(xi_ab*phi_b1 + xi_ac*phi_c1)
            dmua_dphib1 = xi_ab       - k_bT - nu[0]*(xi_ab*phi_a1 + xi_bc*phi_c1)
            dmua_dphic1 = xi_ac       - k_bT - nu[0]*(xi_ac*phi_a1 + xi_bc*phi_b1)

            dmub_dphia1 = xi_ab       - k_bT - nu[1]*(xi_ab*phi_b1 + xi_ac*phi_c1)
            dmub_dphib1 = k_bT/phi_b1 - k_bT - nu[1]*(xi_ab*phi_a1 + xi_bc*phi_c1)
            dmub_dphic1 = xi_bc       - k_bT - nu[1]*(xi_ac*phi_a1 + xi_bc*phi_b1)

            dmuc_dphia1 = xi_ac       - k_bT - nu[2]*(xi_ab*phi_b1 + xi_ac*phi_c1)
            dmuc_dphib1 = xi_bc       - k_bT - nu[2]*(xi_ab*phi_a1 + xi_bc*phi_c1)
            dmuc_dphic1 = k_bT/phi_c1 - k_bT - nu[2]*(xi_ac*phi_a1 + xi_bc*phi_b1)

            #PHASE 2
            dmua_dphia2 = k_bT/phi_a2 - k_bT  - nu[0]*(xi_ab*phi_b2 + xi_ac*phi_c2)
            dmua_dphib2 = xi_ab       - k_bT  - nu[0]*(xi_ab*phi_a2 + xi_bc*phi_c2)
            dmua_dphic2 = xi_ac       - k_bT  - nu[0]*(xi_ac*phi_a2 + xi_bc*phi_b2)

            dmub_dphia2 = xi_ab       - k_bT  - nu[1]*(xi_ab*phi_b2 + xi_ac*phi_c2)
            dmub_dphib2 = k_bT/phi_b2 - k_bT  - nu[1]*(xi_ab*phi_a2 + xi_bc*phi_c2)
            dmub_dphic2 = xi_bc       - k_bT  - nu[1]*(xi_ac*phi_a2 + xi_bc*phi_b2)

            dmuc_dphia2 = xi_ac       - k_bT  - nu[2]*(xi_ab*phi_b2 + xi_ac*phi_c2)
            dmuc_dphib2 = xi_bc       - k_bT  - nu[2]*(xi_ab*phi_a2 + xi_bc*phi_c2)
            dmuc_dphic2 = k_bT/phi_c2 - k_bT  - nu[2]*(xi_ac*phi_a2 + xi_bc*phi_b2)

            #CALCULATE MATRIX
            A[0, 0] = nu[0]*( dmua_dphia1*(phi_a1-1) + phi_b1*dmua_dphib1 + phi_c1*dmua_dphic1 + (V1/V2)*(dmua_dphia2*(phi_a2-1) + phi_b2*dmua_dphib2 + phi_c2*dmua_dphic2))
            A[0, 1] = nu[1]*( dmua_dphib1*(phi_b1-1) + phi_a1*dmua_dphia1 + phi_c1*dmua_dphic1 + (V1/V2)*(dmua_dphib2*(phi_b2-1) + phi_a2*dmua_dphia2 + phi_c2*dmua_dphic2))
            A[0, 2] = nu[2]*( dmua_dphic1*(phi_c1-1) + phi_a1*dmua_dphia1 + phi_b1*dmua_dphib1 + (V1/V2)*(dmua_dphic2*(phi_c2-1) + phi_a2*dmua_dphia2 + phi_b2*dmua_dphib2))
            
            b[0]    = r_2[0]*nu[0]*dmua_dphia2 + r_2[1]*nu[1]*dmua_dphib2 + r_2[2]*nu[2]*dmua_dphic2
            b[0]   -= r_1[0]*nu[0]*dmua_dphia1 + r_1[1]*nu[1]*dmua_dphib1 + r_1[2]*nu[2]*dmua_dphic1

            A[1, 0] = nu[0]*( dmub_dphia1*(phi_a1-1) + phi_b1*dmub_dphib1 + phi_c1*dmub_dphic1 + (V1/V2)*(dmub_dphia2*(phi_a2-1) + phi_b2*dmub_dphib2 + phi_c2*dmub_dphic2))
            A[1, 1] = nu[1]*( dmub_dphib1*(phi_b1-1) + phi_a1*dmub_dphia1 + phi_c1*dmub_dphic1 + (V1/V2)*(dmub_dphib2*(phi_b2-1) + phi_a2*dmub_dphia2 + phi_c2*dmub_dphic2))
            A[1, 2] = nu[2]*( dmub_dphic1*(phi_c1-1) + phi_a1*dmub_dphia1 + phi_b1*dmub_dphib1 + (V1/V2)*(dmub_dphic2*(phi_c2-1) + phi_a2*dmub_dphia2 + phi_b2*dmub_dphib2))
            b[1]    = r_2[0]*nu[0]*dmub_dphia2 + r_2[1]*nu[1]*dmub_dphib2 + r_2[2]*nu[2]*dmub_dphic2
            b[1]   -= r_1[0]*nu[0]*dmub_dphia1 + r_1[1]*nu[1]*dmub_dphib1 + r_1[2]*nu[2]*dmub_dphic1

            A[2, 0] = nu[0]*( dmuc_dphia1*(phi_a1-1) + phi_b1*dmuc_dphib1 + phi_c1*dmuc_dphic1 + (V1/V2)*(dmuc_dphia2*(phi_a2-1) + phi_b2*dmuc_dphib2 + phi_c2*dmuc_dphic2))
            A[2, 1] = nu[1]*( dmuc_dphib1*(phi_b1-1) + phi_a1*dmuc_dphia1 + phi_c1*dmuc_dphic1 + (V1/V2)*(dmuc_dphib2*(phi_b2-1) + phi_a2*dmuc_dphia2 + phi_c2*dmuc_dphic2))
            A[2, 2] = nu[2]*( dmuc_dphic1*(phi_c1-1) + phi_a1*dmuc_dphia1 + phi_b1*dmuc_dphib1 + (V1/V2)*(dmuc_dphic2*(phi_c2-1) + phi_a2*dmuc_dphia2 + phi_b2*dmuc_dphib2))
            b[2]    = r_2[0]*nu[0]*dmuc_dphia2 + r_2[1]*nu[1]*dmuc_dphib2 + r_2[2]*nu[2]*dmuc_dphic2
            b[2]   -= r_1[0]*nu[0]*dmuc_dphia1 + r_1[1]*nu[1]*dmuc_dphib1 + r_1[2]*nu[2]*dmuc_dphic1

            j1     =  np.linalg.solve(A, b)
            j2     = -j1*V1/V2 

            V1_rel_dot = -np.sum(nu*j1)
            V2_rel_dot = -np.sum(nu*j2)

            VI[i+1]  = VI[i]  + dt*V1*V1_rel_dot
            VII[i+1] = VII[i] + dt*V2*V2_rel_dot

            phiI[i+1,  :]     =  phiI[i, :]   + nu     *dt*(r_1 - j1 - phiI[i, :]*V1_rel_dot) 
            phiII[i+1, :]     = phiII[i, :]   + nu     *dt*(r_2 - j2 - phiII[i,:]*V2_rel_dot)
            phi_bar[i+1,  :]  = phi_bar[i, :] + nu[:-1]*dt*((VI[i]*r_1[:-1] + VII[i]*r_2[:-1])/V)
            F[i+1] = VI[i+1]*f(phiI[i+1, :], k_bT, nu, xi_ab, xi_ac, xi_bc, omega) + VII[i+1]*f(phiII[i+1, :], k_bT, nu, xi_ab, xi_ac, xi_bc, omega)

        else:
            mu          = calc_gibbs_mu(phi_bar[i,  :], k_bT, nu, omega, xi)

            delta0      = fuel_foo(phi_bar[i,  0], phi_bar[i,  1], thresh, kappa, delta)
            react       = k_c*(-np.exp((mu[0]+delta0)/k_bT) + np.exp(mu[1]/k_bT))
            psi         = k_p*(np.exp((mu[0]+mu[1])/k_bT) - np.exp(2*mu[2]/k_bT))
            r           = np.array([react-psi, -react-psi, 2*psi])

            phi_bar[i+1, :]  =  phi_bar[i, :] + nu[:-1]*dt*r[:-1]
            F[i+1]           = f(phi_bar[i+1, :], k_bT, nu, xi_ab, xi_ac, xi_bc, omega)
            chems[i+1, 0] = mu[0]
            chems[i+1, 1] = mu[1]
            chems[i+1, 2] = mu[2]
            chems[i+1, 3] = delta0
            chems[i+1, 4] = delta0

        if (i+1) % 1000 == 0:
            done = np.linalg.norm(phi_bar[i+1, :]-phi_bar[i, :]) < 1e-7
            if done:
                print("DONE", np.linalg.norm(phi_bar[i+1, :]-phi_bar[i, :]))
                break
    return phi_bar[:i+1], phiI[:i+1], phiII[:i+1], VI[:i+1], VII[:i+1], F[:i+1], chems[:i+1, :], T[:i+1]
    
def period_complete(T, x, y):
    y_end = y[-1]
    x_end = x[-1]

    skip     = int(0.4*len(T))
    x, y = x[:-skip], y[:-skip]

    diff = np.sqrt( np.square(y-y_end) + np.square(x - x_end) )
    if np.amin(diff) < 2*1e-5:
        return True
    else:
        return False

def get_period(T, x, y, tol):
    y_end = y[-1]
    x_end = x[-1]
    T_fin = T[-1]

    skip     = int(0.02*len(T))
    x, y, T = x[:-skip], y[:-skip], T[:-skip]

    diff  = np.sqrt( np.square(y-y_end) + np.square(x - x_end) )
    time_deriv = np.gradient(diff, T)
    indxs = np.sort(np.argwhere(diff < tol))[::-1, 0]

    for indx in indxs:
        if np.amax(diff[indx:]) > 0.01 and np.sign(np.amax(time_deriv[indx:])) != np.sign(np.amin(time_deriv[indx:])):
            #plt.plot(indx, diff[indx], "ro")
            #plt.show()
            return T_fin-T[indx], True
    return 0, False






