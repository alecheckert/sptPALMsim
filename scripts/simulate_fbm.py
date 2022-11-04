import os, numpy as np, pandas as pd, tifffile
from sps import (
    MOTION_TYPE_FRACTIONAL_BROWNIAN,
    SPTSimulator,
)

def simulate_fbm():
    """ Make a set of fractional Brownian motion simulations.
    We use a few different models here, all without state transitions.

    """
    # Relative path to output directory
    out_dir = "fbm_simulations_pulse_width_4ms"

    # Frame interval in seconds
    frame_interval = 0.00748

    # Diffusion coefficients, Hurst parameters, and state occupations
    # for each model to simulate
    models = [
        # Model 0 (1 state; Brownian)
        (
            np.array([1.0]),
            np.array([0.5]),
            np.array([1.0]),
        ),
        # Model 1 (1 state)
        (
            np.array([1.0]),
            np.array([0.2]),
            np.array([1.0]),
        ),
        # Model 2 (1 state)
        (
            np.array([1.0]),
            np.array([0.35]),
            np.array([1.0]),
        ),
        # Model 3 (1 state)
        (
            np.array([1.0]),
            np.array([0.65]),
            np.array([1.0]),
        ),
        # Model 4 (1 state)
        (
            np.array([1.0]),
            np.array([0.8]),
            np.array([1.0]),
        ),
        # Model 5 (2 states)
        (
            np.array([0.1, 5.0]),
            np.array([0.3, 0.6]),
            np.array([0.5, 0.5]),
        ),
        # Model 6 (3 states)
        (
            np.array([0.1, 2.5, 8.0]),
            np.array([0.8, 0.4, 0.5]),
            np.array([0.3, 0.4, 0.3]),
        ),
        # Model 7 (5 states)
        (
            np.array([0.05, 0.30, 1.50, 6.00, 10.00]),
            np.array([0.50, 0.70, 0.25, 0.60,  0.50]),
            np.array([0.20, 0.25, 0.20, 0.20,  0.15]),
        ),
        # Model 8 (1 state; Brownian)
        (
            np.array([6.0]),
            np.array([0.5]),
            np.array([1.0]),
        ),
        # Model 9 (1 state)
        (
            np.array([6.0]),
            np.array([0.2]),
            np.array([1.0]),
        ),
        # Model 10 (1 state)
        (
            np.array([6.0]),
            np.array([0.35]),
            np.array([1.0]),
        ),
        # Model 11 (1 state)
        (
            np.array([6.0]),
            np.array([0.65]),
            np.array([1.0]),
        ),
        # Model 12 (1 state)
        (
            np.array([6.0]),
            np.array([0.8]),
            np.array([1.0]),
        ),
        # Model 13 (2 state)
        (
            np.array([5.0, 10.0]),
            np.array([0.7, 0.35]),
            np.array([0.5, 0.5]),
        ),
        # Model 14 (3 state)
        (
            np.array([0.01, 0.08, 0.4]),
            np.array([0.5, 0.8, 0.4]),
            np.array([0.2, 0.4, 0.4]),
        ),
        # Model 15 (4 state)
        (
            np.array([0.2, 0.8, 2.0, 8.0]),
            np.array([0.7, 0.5, 0.4, 0.3]),
            np.array([0.25, 0.25, 0.25, 0.25]),
        ),
    ]
    n_models = len(models)

    # Number of simulations to perform per model
    simulations_per_model = 16

    # Number of frames per simulation
    n_frames = 50

    # Number of tracks per simulation
    n_tracks = 1500

    # Index of the GPU to use
    gpu = 1

    # Simulation to start at 
    start_idx = 0
    start_model_idx = 8

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    def make_out_paths(model_index: int, replicate: int) -> (str, str):
        movie_path = os.path.join(
            out_dir,
            f"fbm_model-index-{model_index}_rep-{replicate}.tif",
        )
        csv_path = f"{os.path.splitext(movie_path)[0]}_trajectories.csv"
        return movie_path, csv_path

    # Total number of simulations to run, and current iteration index
    n_simulations = n_models * simulations_per_model
    c = 0

    for model_index, (diff_coefs, hurst_pars, occs) in enumerate(models):
        if model_index < start_model_idx:
            c += simulations_per_model
            continue

        # Modify the diffusion coefficients so that the total jump variance
        # is the same for the same diffusion coefficient at this frame interval
        mod_diff_coefs = [D / np.power(frame_interval, 2 * H - 1) \
            for D, H in zip(diff_coefs, hurst_pars)]

        for replicate in range(simulations_per_model):
            print(f"Simulation {c+1}/{n_simulations}: model_index = {model_index}, diff_coefs = {diff_coefs}, hurst_pars = {hurst_pars}, mod_diff_coefs = {mod_diff_coefs}")
            movie_path, csv_path = make_out_paths(model_index, replicate)
            if os.path.isfile(movie_path):
                print(f"  output {movie_path} already exists; moving on")
                c += 1
                continue
            elif c < start_idx:
                c += 1
                continue
            print(f"  movie_path = {movie_path}")
            print(f"  csv_path = {csv_path}")
            with SPTSimulator(
                motion_type=MOTION_TYPE_FRACTIONAL_BROWNIAN,
                frame_interval=frame_interval,
                pulse_width=0.004,
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
                diff_coefs=mod_diff_coefs,
                hurst_pars=hurst_pars,
                state_occs=occs,
            ) as simulator:
                movie, trajectories = simulator.sim(n_tracks, n_frames)
                tifffile.imsave(movie_path, movie)
                trajectories.to_csv(csv_path, index=False)
            c += 1
    
if __name__ == "__main__":
    simulate_fbm()
