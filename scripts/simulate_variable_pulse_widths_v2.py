import os, numpy as np, pandas as pd, tifffile
from sps import (
    MOTION_TYPE_BROWNIAN,
    SPTSimulator,
)

def simulate_variable_pulse_widths():
    """ Make a set of one-state Brownian motion simulations at variable pulse
    widths. The Brownian motion has diffusion coefficient 10.0 µm2 s-1. """
    # Relative path to output directory
    out_dir = "variable_pulse_width_simulations"

    # Pulse widths to simulate
    pulse_width_values = [0.000125, 0.00025, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.016]

    # Diffusion coefficients to simulate
    diff_coef_values = np.array([20.0])

    # Number of simulations to perform per parameter set
    replicates_per_parameter_set = 10

    # Number of frames per simulation
    n_frames = 100

    # Number of tracks per simulation
    n_tracks = 750

    # Index of the GPU to use
    gpu = 0

    # Simulation to start on
    start_idx = 0

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    def make_out_paths(pulse_width, diff_coef, replicate):
        movie_path = os.path.join(
            out_dir,
            f"pulse-width-{pulse_width}_diff-coef-{diff_coef}_rep-{replicate}.tif"
        )
        csv_path = f"{os.path.splitext(movie_path)[0]}_trajectories.csv"
        return movie_path, csv_path

    # Total number of simulations to run, and current iteration index
    n_simulations = len(pulse_width_values) * len(diff_coef_values) * \
        replicates_per_parameter_set
    c = 0

    for pulse_width in pulse_width_values:
        for diff_coef in diff_coef_values:
            for replicate in range(replicates_per_parameter_set):
                print(f"Simulation {c+1}/{n_simulations}: " \
                    f"pulse_width = {pulse_width}, diff_coef = {diff_coef}, replicate = {replicate}")
                movie_path, csv_path = make_out_paths(pulse_width, diff_coef, replicate)
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
                    motion_type=MOTION_TYPE_BROWNIAN,
                    frame_interval=0.02,
                    pulse_width=pulse_width,
                    bleach_rate=0.2,
                    intensity=150.0,
                    wavelength=0.67,
                    na=1.49,
                    ref_index=1.515,
                    pixel_size=0.16,
                    fov_size=(128, 128),
                    spatial_bin_rate=8,
                    temporal_bin_rate=200,
                    offset=470.0,
                    gain=109.0,
                    read_noise=3.0,
                    gpu=gpu,
                    diff_coefs=np.array([diff_coef]),
                    TM = np.array([[1.0]]),
                ) as simulator:
                    movie, trajectories = simulator.sim(n_tracks, n_frames)
                    tifffile.imsave(movie_path, movie)
                    trajectories.to_csv(csv_path, index=False)
                c += 1
    
if __name__ == "__main__":
    simulate_variable_pulse_widths()
