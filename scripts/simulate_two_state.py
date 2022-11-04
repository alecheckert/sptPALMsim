import os, numpy as np, pandas as pd, tifffile
from sps import (
    MOTION_TYPE_BROWNIAN,
    SPTSimulator,
)

def simulate_two_state():
    """ Make a range of two-state Brownian motion simulations.

    We vary two parameters in this simulation:
        1. The fraction of particles in each state
        2. The diffusion coefficient of the "fast" state

    The "slow" state has a constant diffusion coefficient (0.02
    squared microns per second.

    """
    # Relative path to output directory
    out_dir = "two_state_simulations"

    # Diffusion coefficient of the slow state (constant)
    slow_diff_coef = 0.01

    # Fraction of trajectories in the slow state
    frac_slow_values = np.array([
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    ])

    # Diffusion coefficient of the faster state
    fast_diff_coef_values = np.array([
        0.078125, 0.15625, 0.3125, 0.625, 1.25, 2.5, 5.0, 10.0, 20.0
    ])

    # Number of simulations to perform per parameter set
    simulations_per_parameter_set = 8

    # Number of frames per simulation
    n_frames = 100

    # Number of tracks per simulation
    n_tracks = 1500

    # Index of the GPU to use
    gpu = 1

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    def make_out_paths(frac_slow, fast_diff_coef, replicate):
        movie_path = os.path.join(
            out_dir,
            f"frac-slow-{frac_slow}_D-fast-{fast_diff_coef}_rep-{replicate}.tif"
        )
        csv_path = f"{os.path.splitext(movie_path)[0]}_trajectories.csv"
        return movie_path, csv_path

    # Total number of simulations to run, and current iteration index
    n_simulations = len(frac_slow_values) * len(fast_diff_coef_values) * \
        simulations_per_parameter_set
    c = 0

    for frac_slow in frac_slow_values:
        for fast_diff_coef in fast_diff_coef_values:
            for replicate in range(16, 16 + simulations_per_parameter_set):
                print(f"Simulation {c+1}/{n_simulations}: " \
                    f"frac_slow = {frac_slow}, D_fast = {fast_diff_coef}")
                movie_path, csv_path = make_out_paths(
                    frac_slow, fast_diff_coef, replicate
                )
                if os.path.isfile(movie_path):
                    print(f"  output {movie_path} already exists; moving on")
                    c += 1
                    continue
                print(f"  movie_path = {movie_path}")
                print(f"  csv_path = {csv_path}")
                diff_coefs = np.array([slow_diff_coef, fast_diff_coef])
                with SPTSimulator(
                    motion_type=MOTION_TYPE_BROWNIAN,
                    frame_interval=0.00748,
                    pulse_width=0.002,
                    bleach_rate=0.2,
                    intensity=150.0,
                    wavelength=0.67,
                    na=1.49,
                    ref_index=1.515,
                    pixel_size=0.16,
                    fov_size=(128, 128),
                    spatial_bin_rate=8,
                    temporal_bin_rate=60,
                    offset=470.0,
                    gain=109.0,
                    read_noise=3.0,
                    gpu=gpu,
                    diff_coefs=diff_coefs,
                    TM=np.array([[frac_slow, 0], [0, 1-frac_slow]]),
                ) as simulator:
                    movie, trajectories = simulator.sim(n_tracks, n_frames)
                    tifffile.imsave(movie_path, movie)
                    trajectories.to_csv(csv_path, index=False)
                c += 1
    
if __name__ == "__main__":
    simulate_two_state()
