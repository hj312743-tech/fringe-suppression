# lensless-holography-parasitic-fringe-suppression
# Physics-driven self-supervised parasitic-fringe suppression

This repository provides the core scripts used for physics-driven self-supervised parasitic-fringe suppression in lensless holographic reconstruction. The code implements the main reconstruction model, two-stage focus search, classical angular-spectrum back-propagation, and matched simulated hologram generation.

## 1. Files

- `run_untrained.py`  
  Main reconstruction script for the proposed physics-driven self-supervised object/background/fringe decomposition model.

- `run_sweep_z_reconstruct.py`  
  Two-stage angular-spectrum focus search. It first performs a coarse scan over a user-defined propagation range and then performs a fine scan around the best coarse plane.

- `run_backprop_reconstruct.py`  
  Classical angular-spectrum back-propagation at a fixed propagation distance. This script is mainly used as a physical baseline or for quick inspection.

- `simulate_hologram.py`  
  Matched simulated hologram generator with object modulation, smooth background modulation, directional parasitic fringes, and mixed acquisition noise.

## 2. Environment

The code was tested with Python and PyTorch. The main required packages are:

```bash
numpy
scipy
matplotlib
pillow
torch
