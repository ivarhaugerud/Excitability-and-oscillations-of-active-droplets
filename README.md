# Excitability and Oscillations of Active Droplets

Code accompanying the paper:

> **Excitability and oscillations of active droplets**  
> Ivar S. Haugerud, Hidde D. Vuijk, Job Boekhoven, Christoph A. Weber  
> *arXiv:2503.11604* (2025)  
> DOI: [10.48550/arXiv.2503.11604](https://doi.org/10.48550/arXiv.2503.11604)

## Overview

This repository contains the simulation and plotting code used to produce all figures in the paper. The paper develops a minimal physicochemical model for a three-component reactive mixture (components A, B, C) governed by Flory-Huggins thermodynamics and fuel-driven non-equilibrium reactions. The key findings are:

- A pitchfork bifurcation in the active droplet state as fuel input increases.
- An excitability regime where a perturbation triggers a large transient before returning to a stable steady state.
- Self-sustained oscillations (limit cycles) of droplet formation and dissolution above a critical fuel strength.

## Repository Structure

```
.
├── functions.py        # Core library: thermodynamics, phase equilibrium, ODE solver
├── run_spatial.py      # 2D spatial PDE simulation (Fourier-spectral, Cahn-Hilliard-type)
├── 2a.py               # Figure 2a: NESS locus in phase diagram for varying fuel strength
├── 2bc.py              # Figure 2b/c: reaction-rate zero-crossings and phase-separated volume
├── 2de.py              # Figure 2d/e: state diagram (homogeneous / active droplet / bistable)
├── 3.py                # Figure 3: excitable trajectories in the composition phase diagram
├── 4ab.py              # Figure 4a/b: periodic orbit with time-coloured phase portrait
├── 4c.py               # Figure 4c: oscillation frequency vs reaction-rate ratio and fuel strength
├── 5.py                # Figure 5: hysteresis — droplet volume, entropy production, orbit length
├── data/               # Pre-computed data files loaded by the plotting scripts
└── figures/            # Output directory for generated figures
```

## Key Modules

### `functions.py`

The central library. Key components:

| Function | Description |
|---|---|
| `calc_gibbs_mu` | Flory-Huggins chemical potentials for all three components |
| `get_tieline_twocomp` | Two-component coexistence via convex-hull construction |
| `calculate_binodal` | Three-component binodal curve by marching along average compositions |
| `fuel_foo` | Smooth (arctan) fuel-strength function as a function of local concentration |
| `contour_lines_ab_psi` | Reaction nullclines (reaction 1: `mu_A = mu_B`; reaction 2: `mu_A + mu_B = 2*mu_C`) |
| `run_system` | Forward-Euler ODE integrator handling 1-phase ↔ 2-phase transitions; detects periodic orbits |
| `get_period` | Period detection for closed trajectories |
| `get_flows` | Reaction-rate vector field (quiver data) over the composition space |

### `run_spatial.py`

Solves the 2D Cahn-Hilliard equations with fuel-driven reactions on a 150×150 periodic grid. The Laplacian is applied spectrally (FFT), and the interfacial (surface-tension) terms are treated implicitly for stability. Concentration snapshots are written to `data/` at regular intervals.

## Dependencies

- Python 3
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [Matplotlib](https://matplotlib.org/)
- [SciencePlots](https://github.com/garrettj403/SciencePlots) (`pip install SciencePlots`)
- [colormaps](https://colormaps.readthedocs.io/) (`pip install colormaps`) — provides the `WhiteYellowOrangeRed` colormap used in several figures

Install all dependencies with:

```bash
pip install numpy scipy matplotlib SciencePlots colormaps
```

## Usage

Most figure scripts load pre-computed data from `data/` and produce PDF figures in `figures/`. Run each script directly:

```bash
python 2a.py
python 3.py
python 4ab.py
# etc.
```

The spatial simulation writes its own data files and can be run independently:

```bash
python run_spatial.py
```

## Citation

If you use this code, please cite:

```bibtex
@misc{haugerud2025excitability,
  title   = {Excitability and oscillations of active droplets},
  author  = {Haugerud, Ivar S. and Vuijk, Hidde D. and Boekhoven, Job and Weber, Christoph A.},
  year    = {2025},
  eprint  = {2503.11604},
  archivePrefix = {arXiv},
  primaryClass  = {cond-mat.soft},
  doi     = {10.48550/arXiv.2503.11604}
}
```
