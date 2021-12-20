import os, numpy as np, pandas as pd, scipy.ndimage as ndi
from typing import Tuple, Callable
from functools import cached_property 
from .constants import MOTION_TYPE_BROWNIAN
from .camera import Camera
from .motion import Motion, make_motion
from .optics import Optics
from .constants import (
    MOTION_TYPE_BROWNIAN,
    DEFAULT_FRAME_INTERVAL,
    DEFAULT_PULSE_WIDTH,
    DEFAULT_BLEACH_RATE,
    DEFAULT_INTENSITY,
    DEFAULT_WAVELENGTH,
    DEFAULT_NA,
    DEFAULT_REF_INDEX,
    DEFAULT_PIXEL_SIZE,
    DEFAULT_SPATIAL_BIN_RATE,
    DEFAULT_TEMPORAL_BIN_RATE,
    DEFAULT_Z_LEVELS,
    DEFAULT_FOV_SIZE,
    DEFAULT_OFFSET,
    DEFAULT_GAIN,
    DEFAULT_READ_NOISE,
)

class SPTSimulator:
    def __init__(self, optics: Optics, camera: Camera, motion: Motion):
        self.optics = optics 
        self.camera = camera 
        self.motion = motion

    def __init__(
        self,
        motion_type:       str=MOTION_TYPE_BROWNIAN,
        frame_interval:    float=DEFAULT_FRAME_INTERVAL,
        pulse_width:       float=DEFAULT_PULSE_WIDTH,
        bleach_rate:       float=DEFAULT_BLEACH_RATE,
        intensity:         float=DEFAULT_INTENSITY,
        wavelength:        float=DEFAULT_WAVELENGTH,
        na:                float=DEFAULT_NA,
        ref_index:         float=DEFAULT_REF_INDEX,
        pixel_size:        float=DEFAULT_PIXEL_SIZE,
        fov_size:          Tuple[int]=DEFAULT_FOV_SIZE,
        spatial_bin_rate:  int=DEFAULT_SPATIAL_BIN_RATE,
        temporal_bin_rate: float=DEFAULT_TEMPORAL_BIN_RATE,
        z_levels:          np.ndarray=DEFAULT_Z_LEVELS,
        pupil_func:        Callable=None,
        offset:            float=DEFAULT_OFFSET,
        gain:              float=DEFAULT_GAIN,
        read_noise:        float=DEFAULT_READ_NOISE,
        gpu:               int=None,
        verbose:           bool=True,
        **kwargs
    ):
        """ Synthesize an SPTSimulator from a set of keyword arguments, instantiating the 
        underlying Optics, Camera, and Motion from scratch.

        Parameters
        ----------
            motion_type         :   type of motion to simulate (see motions.make_motion)

            frame_interval      :   frame interval in seconds

            pulse_width         :   excitation pulse width in seconds

            wavelength          :   emission wavelength in microns

            na                  :   numerical aperture

            ref_index           :   refractive index of immersion medium

            pixel_size          :   size of camera pixels in microns

            fov_size            :   2-tuple of int, YX dimensions of FOV (camera pixels)

            spatial_bin_rate    :   number of simulated "optics pixels" per camera pixel.
                                    Higher values make a better simulation but are more
                                    costly;

            z_levels            :   1D numpy.ndarray of shape (nz,), z-levels relative to
                                    the focus in microns

            pupil_func          :   Callable to generate pupil function (see Optics)

            offset              :   pixel offset (single value or 2D numpy.ndarray of 
                                    shape *fov_size*), grayvalues

            gain                :   pixel gain (single value or 2D numpy.ndarray of shape 
                                    *fov_size*), grayvalues / photon

            read_noise          :   read noise root variance (single value or 2D numpy.ndarray
                                    of shape *fov_size*), grayvalues

            gpu                 :   index of GPU (if using) or None (if using CPU)

            verbose             :   show indicators of progress

            kwargs              :   additional kwargs to make_motion()

        Returns
        -------
            new instance of SPTSimulator
        """
        self.frame_interval = frame_interval 
        self.pulse_width = pulse_width 
        self.bleach_rate = bleach_rate
        self.intensity = intensity
        self.spatial_bin_rate = spatial_bin_rate
        self.temporal_bin_rate = temporal_bin_rate
        self.fov_size = fov_size 
        self.verbose = verbose 

        # Spatial discretization (YX)
        self.grid_pixel_size = pixel_size / spatial_bin_rate
        self.grid_fov_size = (fov_size[0] * spatial_bin_rate, fov_size[1] * spatial_bin_rate)

        # Temporal discretization (T)
        self.dt = frame_interval / temporal_bin_rate

        # Optics simulator
        self.optics = Optics(
            wavelength=wavelength,
            na=na,
            ref_index=ref_index,
            pixel_size=self.grid_pixel_size,
            fov_size=self.grid_fov_size,
            pupil_func=pupil_func,
            z_levels=z_levels,
            gpu=gpu,
            mode="noninterference",
            verbose=verbose,
        )

        # Camera noise simulator
        self.camera = Camera(
            offset=offset,
            gain=gain,
            read_noise=read_noise,
            fov_size=fov_size,
        )

        # Motion simulator
        self.motion = make_motion(
            motion_type=motion_type,
            dt=self.dt,
            **kwargs
        )

    def aggregate(self, grid_image: np.ndarray) -> np.ndarray:
        assert grid_image.shape == self.grid_fov_size 
        result = np.zeros(self.fov_size, dtype=grid_image.dtype)
        for i in range(self.spatial_bin_rate):
            for j in range(self.spatial_bin_rate):
                result += grid_image[i::self.spatial_bin_rate, j::self.spatial_bin_rate]
        return result

    def sim(self, n_tracks: int, n_frames: int) -> (np.ndarray, pd.DataFrame):
        """ Simulate an SPT movie with accompanying ground truth trajectories.

        Parameters
        ----------
            n_tracks    :   number of trajectories to simulate
            n_frames    :   number of frames to simulate

        Returns
        -------
            (
                3D numpy.ndarray of shape (n_frames, *self.fov_size), the simulated
                    movie (uint16);
                pandas.DataFrame, trajectory ground truth
            )
        """
        n_dt = n_frames * self.temporal_bin_rate 
        T = np.arange(n_dt) * self.dt 
        in_pulse = (T % self.frame_interval) < self.pulse_width
        pulses = ndi.label(in_pulse)[0]
        n_pulses = pulses.max()

        # Simulate trajectories
        tracks, states = self.motion(n_tracks, n_dt, bleach_rate=self.bleach_rate)

        # Simulate SPT movie
        movie = np.zeros((n_frames, *self.fov_size), dtype=np.uint16)
        for p in range(1, n_pulses+1):
            match_pulse = pulses == p 
            point_sources = tracks[:,match_pulse,:].reshape((match_pulse.sum() * n_tracks, 3))
            movie[p-1,:,:] = self.camera(self.aggregate(self.optics.image_point_sources(
                point_sources,
                intensity=self.intensity/match_pulse.sum()
            ))).astype(np.uint16)
            if self.verbose:
                print(f"finished with pulse {p}/{n_pulses}")

        # Format ground truth trajectory DataFrame
        n_points = n_tracks * n_dt 
        tracks_df = pd.DataFrame(
            index=np.arange(n_points),
            columns=["trajectory", "microframe", "t", "z", "y", "x"],
            dtype=object,
        )
        tracks_df["trajectory"] = (np.arange(n_points) // n_dt).astype(np.int64)
        tracks_df["microframe"] = (np.arange(n_points) % n_dt).astype(np.int64)
        tracks_df["pulse_idx"] = np.tile(pulses, n_tracks)
        tracks_df["t"] = tracks_df["microframe"] * self.dt 
        tracks_df["z"] = tracks[:,:,0].ravel()
        tracks_df["y"] = tracks[:,:,1].ravel()
        tracks_df["x"] = tracks[:,:,2].ravel()

        # Format summary trajectories (akin to what would be seen for a typical
        # tracking movie, averaging over positions in each pulse)
        tracks_sum = tracks_df[tracks_df["pulse_idx"] > 0]
        tracks_sum = tracks_sum.groupby(by=["trajectory", "pulse_idx"], as_index=False).mean()

        # Alias frame to pulse - 1
        tracks_sum["frame"] = tracks_sum["pulse_idx"] - 1

        # Add background profile
        # if not bg_movie_path is None:
        #     movie = self.add_bg_movie(movie, bg_movie_path)

        return movie, tracks_sum





