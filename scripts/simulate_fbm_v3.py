import os, numpy as np, pandas as pd, tifffile
from sps import (
    MOTION_TYPE_FRACTIONAL_BROWNIAN,
    SPTSimulator,
)

def simulate_fbm_v3():
    """ Make a set of fractional Brownian motion simulations. The idea is
    to simulate a range of diffusion coefficients for each Hurst parameter,
    with the hope of training a weakly-supervised spot classifier to distinguish
    between different Hurst parameters regardless of the scaling coefficient.

    Also includes some simulations with multiple Hurst parameters.
    No state transitions.
    """
    # Relative path to output directory
    out_dir = "fbm_simulations_v3"

    # Frame interval in seconds
    frame_interval = 0.01

    # Diffusion coefficients, Hurst parameters, and state occupations
    # for each model to simulate
    models = [
        # Model 0
        (
            np.array([0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24]),
            np.array([0.1 for i in range(10)]),
            np.ones(10, dtype=np.float64) / 10
        ),
        # Model 1
        (
            np.array([0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24]),
            np.array([0.15 for i in range(10)]),
            np.ones(10, dtype=np.float64) / 10
        ),
        # Model 2
        (
            np.array([0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24]),
            np.array([0.20 for i in range(10)]),
            np.ones(10, dtype=np.float64) / 10
        ),
        # Model 3
        (
            np.array([0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24]),
            np.array([0.25 for i in range(10)]),
            np.ones(10, dtype=np.float64) / 10
        ),
        # Model 4
        (
            np.array([0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24]),
            np.array([0.3 for i in range(10)]),
            np.ones(10, dtype=np.float64) / 10
        ),
        # Model 5
        (
            np.array([0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24]),
            np.array([0.35 for i in range(10)]),
            np.ones(10, dtype=np.float64) / 10
        ),
        # Model 6
        (
            np.array([0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24]),
            np.array([0.4 for i in range(10)]),
            np.ones(10, dtype=np.float64) / 10
        ),
        # Model 7
        (
            np.array([0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24]),
            np.array([0.45 for i in range(10)]),
            np.ones(10, dtype=np.float64) / 10
        ),
        # Model 8
        (
            np.array([0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24]),
            np.array([0.5 for i in range(10)]),
            np.ones(10, dtype=np.float64) / 10
        ),
        # Model 9
        (
            np.array([0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24]),
            np.array([0.55 for i in range(10)]),
            np.ones(10, dtype=np.float64) / 10
        ),
        # Model 10
        (
            np.array([0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24]),
            np.array([0.6 for i in range(10)]),
            np.ones(10, dtype=np.float64) / 10
        ),
        # Model 11
        (
            np.array([0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24]),
            np.array([0.65 for i in range(10)]),
            np.ones(10, dtype=np.float64) / 10
        ),
        # Model 12
        (
            np.array([0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24]),
            np.array([0.7 for i in range(10)]),
            np.ones(10, dtype=np.float64) / 10
        ),
        # Model 10
        (
            np.array([0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24]),
            np.array([0.75 for i in range(10)]),
            np.ones(10, dtype=np.float64) / 10
        ),
        # Model 14
        (
            np.array([0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24]),
            np.array([0.8 for i in range(10)]),
            np.ones(10, dtype=np.float64) / 10
        ),
        # Model 15
        (
            np.array([0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24]),
            np.array([0.85 for i in range(10)]),
            np.ones(10, dtype=np.float64) / 10
        ),
        # Model 16
        (
            np.array([0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24]),
            np.array([0.9 for i in range(10)]),
            np.ones(10, dtype=np.float64) / 10
        ),
        # Model 17 (mixture of a bunch of types of motion)
        (
            np.array([0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24] * 17),
            np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9] * 10),
            np.ones(10*17, dtype=np.float64) / (10*17)
        ),
    ]
    n_models = len(models)

    # Number of simulations to perform per model
    simulations_per_model = 80

    # Number of frames per simulation
    n_frames = 50

    # Number of tracks per simulation
    n_tracks = 1500

    # Index of the GPU to use
    gpu = 1

    # Simulation to start at 
    start_idx = 0
    start_model_idx = 0

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

        for replicate in range(48, 48+simulations_per_model):
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
                pulse_width=0.002,
                bleach_rate=0.2,
                intensity=150.0,
                wavelength=0.67,
                na=1.27,
                ref_index=1.515,
                pixel_size=0.1083,
                fov_size=(128, 128),
                spatial_bin_rate=8,
                temporal_bin_rate=20,
                offset=100.0,
                gain=1.8,
                read_noise=2.0,
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
    simulate_fbm_v3()
