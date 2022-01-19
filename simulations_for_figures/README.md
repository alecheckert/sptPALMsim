# Simulations for figures

This directory contains the scripts used to generate the dynamical-optical simulations
featured in the paper. They rely on the library `sps`. It is _highly_ recommended to run
these with a GPU, since the timescales required to produce these simulations are otherwise
impractical.

## Catalog of simulations

 - `simulate_variable_pulse_widths.py` generates SPT movies with a single Brownian state at various excitation pulse widths, holding the frame interval constant at 20 ms. It is used in Fig. S1.
 - `simulate_two_state.py` generates SPT movies with two Brownian diffusing states, varying the faster state's diffusion coefficient and the fraction of particles in each state. It is used in Fig. S11.
 - `simulate_discrete_dynamic_models.py` generates a sequence of SPT movies with increasingly difficult multi-state Brownian motion that is used in the vbSPT vs. state array comparison (Fig. S12).
 - `simulate_fbm.py` generates a sequence of SPT movies with different fractional Brownian motion models (and no state transitions), with and without motion blur. It is used in Fig. S13.
