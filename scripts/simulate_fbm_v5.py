import os, numpy as np, pandas as pd, tifffile
from sps import MOTION_TYPE_BROWNIAN, SPTSimulator

def simulate_fbm_v5():
    """ Make a set of Brownian motion simulations. These simulations are very
    simple and fall into two categories:
        1. Two-state Brownian motion, with an effect that selectively affects
        the diffusion coefficient of the slow state
        2. Three-state Brownian motion, with an effect that selectively affects
        the diffusion coefficient of the middle state

    Same intensity and density for all simulations; 16 replicates per condition."""
    # Relative path to output directory
    out_dir = "fbm_simulations_v5"

    # Frame interval in seconds
    frame_interval = 0.01

    # Dynamical models to simulate (first element is scaling coefficient,
    # second is Hurst parameter, third is state occupations)
    DYNAMICAL_MODELS = [
        # Model 0 (two-state Brownian motion)
        (
            np.array([0.1, 7.0]),
            np.array([0.2, 0.8]),
        ),
        # Model 1 (like model 0, but with different slow diff coef)
        (
            np.array([0.3, 7.0]),
            np.array([0.2, 0.8]),
        ),
        # Model 2 (three-state Brownian motion)
        (
            np.array([0.05, 1.0, 10.0]),
            np.array([0.25, 0.25, 0.5])
        ),
        # Model 3 (like model 2, but with a different middle diff coef)
        (
            np.array([0.05, 2.0, 10.0]),
            np.array([0.25, 0.25, 0.5])
        ),
    ]

    # Total number of trajectories to simulate
    n_tracks = 4000

    # Mean number of photons per emitter per laser pulse
    psf_intensity = 125.0

    # Number of replicates to do for each dynamical model
    n_replicates = 16
    
    # Number of frames per simulation
    n_frames = 100

    # Index of the GPU to use
    gpu = None

    # Simulation to start at 
    start_idx = 0

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    def make_out_paths(model_idx: int, replicate: int) -> (str, str):
        movie_path = os.path.join(
            out_dir,
            f"model-{model_idx}_rep-{replicate}.tif"
        )
        csv_path = f"{os.path.splitext(movie_path)[0]}_trajectories.csv"
        return movie_path, csv_path

    # Total number of simulations to run, and current iteration index
    n_models = len(DYNAMICAL_MODELS)
    n_simulations = n_models * n_replicates
    print(f"{n_simulations} total simulations")

    c = 0
    for model_index, model in enumerate(DYNAMICAL_MODELS):
        diff_coefs, occs = model
        for replicate in range(n_replicates):
            if c < start_idx:
                print(f"Skipping simulation {c+1}/{n_simulations}")
                c += 1
                continue
            print(
                f"Simulation {c+1}/{n_simulations}: model={model_index}, replicate={replicate}"
            )
            movie_path, csv_path = make_out_paths(model_index, replicate)
            if os.path.isfile(movie_path):
                print(f"  output {movie_path} already exists; moving on")
                c += 1
                continue
            with SPTSimulator(
                motion_type=MOTION_TYPE_BROWNIAN,
                frame_interval=frame_interval,  # seconds
                pulse_width=0.002,              # seconds
                bleach_rate=0.2,                # Hz
                intensity=psf_intensity,            # mean photons/emitter
                wavelength=0.67,                # emission wavelength (microns)
                na=1.27,                        # numerical aperture
                ref_index=1.333,                # water
                pixel_size=0.1083,              # microns
                fov_size=(256, 256),            # height, width (pixels)
                spatial_bin_rate=4,             # per pixel
                temporal_bin_rate=20,           # per frame
                offset=100.0,                   # grayvalues
                gain=1.8,                       # grayvalues/photon
                read_noise=2.0,                 # grayvalues (stdev)
                gpu=gpu,
                diff_coefs=diff_coefs,
                TM=np.diag(occs),
            ) as simulator:
                movie, trajectories = simulator.sim(n_tracks, n_frames)
                tifffile.imsave(movie_path, movie)
                trajectories.to_csv(csv_path, index=False)
            c += 1

if __name__ == "__main__":
    simulate_fbm_v5()
