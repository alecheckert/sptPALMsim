import os, numpy as np, pandas as pd, tifffile
from sps import (
    MOTION_TYPE_FRACTIONAL_BROWNIAN,
    SPTSimulator,
)

def simulate_fbm_v4():
    """ Make a set of fractional Brownian motion simulations. The idea is
    to simulate a range of diffusion coefficients for each Hurst parameter,
    with the hope of training a weakly-supervised spot classifier to distinguish
    between different Hurst parameters regardless of the scaling coefficient.

    Also includes some simulations with multiple Hurst parameters.
    No state transitions.
    """
    # Relative path to output directory
    out_dir = "fbm_simulations_v4"

    # Frame interval in seconds
    frame_interval = 0.01

    # Densities to simulate (these are specified in terms of the number
    # of trajectories to simulate per movie)
    DENSITIES = [2000, 4000, 6000, 8000]

    # Intensities to simulate
    INTENSITIES = [75.0, 100.0, 150.0, 200.0]

    # Dynamical models to simulate (first element is scaling coefficient,
    # second is Hurst parameter, third is state occupations)
    DYNAMICAL_MODELS = [
        # Model 0 (very slow Brownian motion)
        (
            np.array([0.1]),
            np.array([0.5]),
            np.array([1.0])
        ),
        # Model 1 (slow Brownian motion)
        (
            np.array([1.0]),
            np.array([0.5]),
            np.array([1.0]),
        ),
        # Model 2 (fast Brownian motion)
        (
            np.array([10.0]),
            np.array([0.5]),
            np.array([1.0]),
        ),
        # Model 3 (really fast Brownian motion)
        (
            np.array([20.0]),
            np.array([0.5]),
            np.array([1.0])
        ),
        # Model 4 (equal mixture of slow and fast Brownian motion)
        (
            np.array([1.0, 10.0]),
            np.array([0.5, 0.5]),
            np.array([0.5, 0.5]),
        ),
        # Model 5 (mixture of very slow, slow, and fast Brownian motion)
        (
            np.array([0.1, 1.0, 10.0]),
            np.array([0.5, 0.5, 0.5]),
            np.array([0.2, 0.3, 0.5]),
        ),
        # Model 6 (4-part mixture of Brownian motions)
        (
            np.array([0.1, 1.0, 3.0, 20.0]),
            np.array([0.5, 0.5, 0.5, 0.5]),
            np.array([0.2, 0.2, 0.3, 0.3]),
        ),
        # Model 7 (large range of different diffusion coefficients with equal
        # occupations; all Brownian)
        (
            np.array([0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48]),
            np.full(11, 0.5, dtype=np.float64),
            np.ones(11, dtype=np.float64) / 11.0
        ),
        # Model 8 (subdiffusive vs. Brownian vs. directional motion with different scales)
        (
            np.tile([0.02, 0.08, 0.32, 1.28, 5.12, 10.24], 3),
            np.repeat([0.3, 0.5, 0.7], 6),
            np.full(6*3, 1.0/(6*3), dtype=np.float64)
        ),
        # Model 9 (subdiffusive vs. Brownian vs. directional motion, different scales, faster)
        (
            np.tile([0.04, 0.16, 0.64, 2.56, 10.24, 20.48], 3),
            np.repeat([0.3, 0.5, 0.7], 6),
            np.full(6*3, 1.0/(6*3), dtype=np.float64)
        ),
        # Model 10 (different degrees of directional motion)
        (
            np.tile([0.1, 0.5, 1.0, 2.0, 4.0, 8.0], 5),
            np.repeat([0.5, 0.6, 0.7, 0.8, 0.9], 6),
            np.full(6*3, 1.0/(6*3), dtype=np.float64)
        ),
        # Model 11 (all very directional at different scales) 
        (
            np.array([0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24]),
            np.array([0.9 for i in range(10)]),
            np.ones(10, dtype=np.float64) / 10
        ),
        # Model 12 (all subdiffusive at different scales)
        (
            np.array([0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24]),
            np.array([0.35 for i in range(10)]),
            np.ones(10, dtype=np.float64) / 10
        ),
        # Model 13 (mixture of Brownian and subdiffusive motion at different scales)
        (
            np.array([0.01, 1.0, 8.0, 4.0, 2.0]),
            np.array([0.5, 0.5, 0.5, 0.35, 0.3]),
            np.array([0.2, 0.2, 0.1, 0.25, 0.25]),
        ),
        # Model 14 (wide mixture of different diffusion coefficients and Hurst
        # parameters, all with equal occupation)
        (
            np.array([0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24] * 15),
            np.array([0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9] * 10),
            np.ones(10*15, dtype=np.float64) / (10*15)
        ),
        # Model 15 (wide mixture of different diffusion coefficients and Hurst
        # parameters, without the fastest ones from model 14)
        (
            np.array([0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56] * 15),
            np.array([0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9] * 8),
            np.ones(8*15, dtype=np.float64) / (8*15)
        )
    ]

    # Number of replicates to do for each density, brightness, and dynamical model
    n_replicates = 4
    
    # Number of frames per simulation
    n_frames = 100

    # Index of the GPU to use
    gpu = None

    # Simulation to start at 
    start_idx = 0

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    def make_out_paths(
        n_tracks_idx: int,
        intensity_idx: int,
        dynamical_model_idx: int,
        replicate: int,
    ) -> (str, str):
        movie_path = os.path.join(
            out_dir,
            f"density-{n_tracks_idx}_intensity-{intensity_idx}_model-{dynamical_model_idx}_rep-{replicate}.tif"
        )
        csv_path = f"{os.path.splitext(movie_path)[0]}_trajectories.csv"
        return movie_path, csv_path

    # Total number of simulations to run, and current iteration index
    n_densities = len(DENSITIES)
    n_intensities = len(INTENSITIES)
    n_models = len(DYNAMICAL_MODELS)
    n_simulations = n_densities * n_intensities * n_models * n_replicates
    print(f"{n_simulations} total simulations")

    c = 0
    for density_idx, density in enumerate(DENSITIES):
        for intensity_idx, intensity in enumerate(INTENSITIES):
            for model_index, model in enumerate(DYNAMICAL_MODELS):
                diff_coefs, hurst_pars, occs = model
                mod_diff_coefs = [D / np.power(frame_interval, 2 * H - 1) for D, H in zip(diff_coefs, hurst_pars)]
                for replicate in range(n_replicates):
                    if c < start_idx:
                        print(f"Skipping simulation {c+1}/{n_simulations}")
                        c += 1
                        continue
                    print(
                        f"Simulation {c+1}/{n_simulations}: density={density}, " \
                        f"intensity={intensity}, model={model_index}, replicate={replicate}"
                    )
                    movie_path, csv_path = make_out_paths(density_idx, intensity_idx, model_index, replicate)
                    if os.path.isfile(movie_path):
                        print(f"  output {movie_path} already exists; moving on")
                        c += 1
                        continue
                    with SPTSimulator(
                        motion_type=MOTION_TYPE_FRACTIONAL_BROWNIAN,
                        frame_interval=frame_interval,  # seconds
                        pulse_width=0.002,              # seconds
                        bleach_rate=0.2,                # Hz
                        intensity=intensity,            # mean photons/emitter
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
                        diff_coefs=mod_diff_coefs,
                        hurst_pars=hurst_pars,
                        state_occs=occs,
                    ) as simulator:
                        movie, trajectories = simulator.sim(density, n_frames)
                        tifffile.imsave(movie_path, movie)
                        trajectories.to_csv(csv_path, index=False)
                    c += 1
    
if __name__ == "__main__":
    simulate_fbm_v4()
