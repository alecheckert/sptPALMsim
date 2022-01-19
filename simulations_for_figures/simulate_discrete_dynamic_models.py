import os, numpy as np, pandas as pd, tifffile
from sps import (
    MOTION_TYPE_BROWNIAN,
    SPTSimulator,
)

def simulate_discrete_dynamic_models():
    """ Make several discrete multi-state Brownian simulations, 
    to be used for the state array vs. vbSPT comparison.

    All optical/imaging parameters are kept constant in these
    simulations; only the dynamic model is varied.
    """
    # Relative path to output directory
    out_dir = "discrete_dynamic_models"

    # Dynamic models to use (tuple of diff_coefs, state_occupations)
    dynamic_models = [
        # 0: 1 state
        (
            np.array([1.0]),
            np.array([1.0]),
        ),
        # 1: 2 state
        (
            np.array([0.1, 10.0]),
            np.array([0.5, 0.5]),
        ),
        # 2: 2 state
        (
            np.array([0.3, 1.0]),
            np.array([0.4, 0.6]),
        ),
        # 3: 3 state
        (
            np.array([0.1, 1.0, 7.0]),
            np.array([0.3, 0.4, 0.3]),
        ),
        # 4: 3 state
        (
            np.array([0.1, 8.0, 20.0]),
            np.array([0.4, 0.3, 0.3]),
        ),
        # 5: 4 state
        (
            np.array([0.03, 3.0, 8.0, 20.0]),
            np.array([0.3, 0.15, 0.2, 0.35]),
        ),
        # 6: 4 state
        (
            np.array([0.07, 0.5, 2.0, 10.0]),
            np.array([0.1, 0.3, 0.3, 0.3]),
        ),
        # 7: 5 state
        (
            np.array([0.01, 0.1, 0.8, 3.0, 10.0]),
            np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
        ),
        # 8: 5 state
        (
            np.array([0.02, 0.5, 1.5, 4.0, 15.0]),
            np.array([0.1, 0.2, 0.2, 0.4, 0.1]),
        ),
        # 9: 6 state
        (
            np.array([0.02, 0.2, 0.9, 2.0, 8.0, 20.0]),
            np.array([0.1, 0.2, 0.2, 0.1, 0.2, 0.2]),
        ),
        # 10: 6 state
        (
            np.array([0.01, 0.1, 0.3, 0.8, 2.0, 8.0]),
            np.array([0.1, 0.2, 0.2, 0.2, 0.1, 0.2]),
        ),
        # 11: 7 state
        (
            np.array([0.02, 0.09, 0.3, 1.5, 3.0, 8.0, 20.0]),
            np.array([0.15, 0.1, 0.15, 0.2, 0.1, 0.2, 0.1]),
        ),
        # 12: 2 state
        (
            np.array([0.3, 3.0]),
            np.array([0.3, 0.7]),
        ),
        # 13: 3 state
        (
            np.array([0.1, 0.8, 4.0]),
            np.array([0.5, 0.2, 0.3]),
        ),
        # 14: 4 state
        (
            np.array([0.05, 0.2, 3.0, 8.0]),
            np.array([0.2, 0.3, 0.1, 0.4]),
        ),
        # 15: 6 state
        (
            np.array([0.07, 0.1, 0.2, 0.4, 3.0, 6.0]),
            np.array([0.3, 0.2, 0.1, 0.1, 0.2, 0.1]),
        ),
        # 16: 7 state
        (
            np.array([0.5, 0.7, 0.9, 1.3, 1.7, 2.0, 2.5]),
            np.array([0.0545, 0.3539, 0.2082, 0.0827, 0.1267, 0.0931, 0.0809]),
        ),
        # 17: 8 state
        (
            np.array([0.018, 0.048, 0.826, 3.058, 17.587, 47.585, 57.024, 59.411]),
            np.array([0.0742, 0.2746, 0.2586, 0.019 , 0.0148, 0.1801, 0.0836, 0.0951]),
        ),
        # 18: 9 state
        (
            np.array([0.016, 0.025, 0.068, 0.109, 0.383, 0.658, 0.79 , 2.042, 3.457]),
            np.array([0.0264, 0.2656, 0.1871, 0.124 , 0.0141, 0.1863, 0.0218, 0.0969, 0.0778]),
        ),
        # 19: 4 state
        (
            np.array([0.0123, 0.0462, 0.107, 0.2024]),
            np.array([0.2418, 0.3659, 0.0082, 0.3841]),
        ),
    ]

    # Number of simulations to perform per parameter set
    simulations_per_parameter_set = 48

    # Number of frames per simulation
    n_frames = 100

    # Number of tracks per simulation
    n_tracks = 1500

    # Index of the GPU to use
    gpu = 0

    # Index of the first simulation to use (to resume if dropped)
    start_idx = 0

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    def make_out_paths(model_index, replicate):
        movie_path = os.path.join(
            out_dir,
            f"dynamic-model-{model_index}_rep-{replicate}.tif"
        )
        csv_path = f"{os.path.splitext(movie_path)[0]}_trajectories.csv"
        return movie_path, csv_path

    # Total number of simulations to run, and current iteration index
    n_simulations = len(dynamic_models) * simulations_per_parameter_set
    c = 0

    for model_index, (diff_coefs, state_occs) in enumerate(dynamic_models):
        for replicate in range(simulations_per_parameter_set):
            print(f"Simulation {c+1}/{n_simulations}: model {model_index}, replicate {replicate}")
            movie_path, csv_path = make_out_paths(model_index, replicate)
            if os.path.isfile(movie_path):
                print(f"  output {movie_path} already exists; moving on")
                c += 1
                continue
            elif c < start_idx:
                c += 1
                continue
            # Diagonal transition matrix; no state transitions
            TM = np.identity(len(diff_coefs)) * state_occs
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
                TM=TM,
            ) as simulator:
                movie, trajectories = simulator.sim(n_tracks, n_frames)
                tifffile.imsave(movie_path, movie)
                trajectories.to_csv(csv_path, index=False)
            c += 1

if __name__ == "__main__":
    simulate_discrete_dynamic_models()
