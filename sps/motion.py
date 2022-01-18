import numpy as np
from functools import cached_property
from abc import ABC, abstractmethod 
from typing import Tuple
from .mc import MarkovChain
from .constants import (
    DEFAULT_FOV_SIZE,
    DEFAULT_PIXEL_SIZE,
    DEFAULT_FRAME_INTERVAL,
    DEFAULT_SEED_REGION,
    MOTION_TYPE_BROWNIAN,
    MOTION_TYPE_FRACTIONAL_BROWNIAN,
)

class Motion(ABC):
    """ Base class for trajectory simulators.

    All child classes should define the *self.__init__* and 
    *self.sim* methods. *self.sim* should have the same signature
    as Motion.__call__.

    init
    ----
        dt          :   time interval for the simulation (seconds)
        seed_region :   tuple of 2-tuple, the ZYX seed region
    """
    def __init__(self, dt: float, seed_region: Tuple[Tuple[int]]=DEFAULT_SEED_REGION):
        assert len(seed_region) == 3 and \
            all(map(lambda i: len(i) == 2, seed_region))
        self.dt = dt 
        self.seed_region = seed_region

    def __enter__(self):
        return self 

    def __exit__(self, etype, evalue, traceback):
        return etype is None

    @abstractmethod
    def sim(self, n_tracks: int, n_frames: int, **kwargs) -> (np.ndarray, np.ndarray):
        """ Simulate some trajectories with some options, returning a 3D numpy.ndarray
        with shape (n_tracks, n_frames, 3) giving the ZYX coordinates for each 
        trajectory at each time point and a 1D numpy.ndarray of shape (n_tracks, n_frames)
        with the states indices of each trajectory at each time point """

    def __call__(self, n_tracks: int, n_frames: int, bleach_rate: float=0.0,
        **kwargs) -> (np.ndarray, np.ndarray):
        """
        Make some trajectories according to this simulator.

        Parameters
        ----------
            n_tracks    :   number of trajectories to simulate

            n_frames    :   number of frames to simulate

            bleach_rate :   bleach rate in Hz

            kwargs      :   to self.sim

        Returns
        -------
            (
                3D ndarray of shape (n_tracks, n_frames, 3),
                    ZYX positions of each trajectory at each timepoint
                    in microns. All trajectories start at 0;

                2D ndarray of shape (n_tracks, n_frames), 
                    the state assignments for each trajectory 
                    at each frame
            )
        """
        tracks, states = self.sim(n_tracks, n_frames, **kwargs)
        if bleach_rate > 0.0:
            tracks = self.photobleach(tracks, bleach_rate=bleach_rate, mode="Hz")
        start_pos = self.seed(n_tracks)
        for t in range(n_frames):
            tracks[:,t,:] += start_pos
        return tracks, states 

    def photobleach(self, tracks: np.ndarray, bleach_rate: float=0.0, mode: str="Hz"):
        """ Simulate first-order photobleaching. 

        Parameters
        ----------
            tracks      :   3D np.ndarray of shape (n_tracks, n_frames, 3),
                            the positions of multiple trajectories; for 
                            instance, the output of __call__

            bleach_rate :   float, bleaching rate in Hz or probability per 
                            frame, as given by *mode*

            mode        :   str, either "Hz" or "probability"

        Returns
        -------
            version of *tracks* in which the positions of bleached 
                particles have been replaced with np.nan
        """
        if mode == "Hz":
            bleach_prob = bleach_rate * self.dt 
        else:
            bleach_prob = bleach_rate 

        n_tracks, n_frames, n_dim = tracks.shape 

        # Start with all fluorophores active
        bleached = np.zeros(n_tracks, dtype=np.bool)

        # At each frame, if a particle is bleached, it stays bleached.
        # Otherwise, it bleaches with probability *bleach_prob*.
        for frame in range(1, n_frames):
            bleached = np.logical_or(
                bleached,
                np.random.random(size=n_tracks)<bleach_prob
            )
            for d in range(3):
                tracks[bleached,frame,d] = np.nan 
        return tracks 

    def seed(self, n_points: int) -> np.ndarray:
        """ Randomly choose some 3D points (ZYX) such that the Z-position
        is in the range (-depth/2, depth/2) and the Y/X-positions are in 
        the range (0, fov_size[0]) and (0, fov_size[1]), respectively.

        Parameters
        ----------
            n_points    :   number of points to draw

        Returns
        -------
            2D numpy.ndarray of shape (point_idx, ZYX), coordinates of new
                points in microns
        """
        return np.asarray([np.random.uniform(t[0], t[1], size=n_points) \
            for t in self.seed_region]).T

class BrownianMotion(Motion):
    """ Simulator for multi-state Brownian motion with state transitions.
    The trajectories are seeded in a box with edges defined by *roi_limits*.

    init
    ----
        diff_coefs      :   1D ndarray, diffusion coefficients for each 
                            state in squared microns per second

        TM              :   2D ndarray, transition rate matrix between states
                            in Hz so that TM[i,j] is the rate of transition
                            from state i to state j

        dt              :   float, simulation time interval in seconds
    """
    def __init__(self, diff_coefs: np.ndarray, TM: np.ndarray, dt: float, 
        seed_region: Tuple[Tuple[int]]=DEFAULT_SEED_REGION):
        super().__init__(dt=dt, seed_region=seed_region)
        self.diff_coefs = diff_coefs
        self.TM = TM 
        self.MC = MarkovChain(self.TM, dt=dt)

    def sim(self, n_tracks: int, n_frames: int) -> (np.ndarray, np.ndarray):
        """ Simulate multiple trajectories, starting from the origin.

        Parameters
        ----------
            n_tracks    :   int, number of tracks to simulate
            n_frames    :   int, number of frames to simulate

        Returns
        -------
            (
                3D ndarray of shape (n_tracks, n_frames, 3),
                    the ZYX positions of each trajectory at
                    each timepoint in microns. All trajectories
                    start at 0;
                2D ndarray of shape (n_tracks, n_frames),
                    the state assignments for each trajectory
                    at each frame
            )
        """
        # Spatial root variances due to diffusion
        S = np.sqrt(2 * self.dt * self.diff_coefs)

        tracks = np.zeros((n_tracks, n_frames, 3), dtype=np.float64)
        states = np.zeros((n_tracks, n_frames), dtype=np.uint16)

        for i in range(n_tracks):

            # Simulate state history
            states[i,:] = self.MC(n_frames)

            # Simulate jumps
            for d in range(3):
                tracks[i,1:,d] = np.random.normal(scale=S[states[i,1:]])

        return np.cumsum(tracks, axis=1), states

class FractionalBrownianMotion(Motion):
    """ Simulator for multi-state fractional Brownian motion *without* state transitions.
    The trajectories are seeded in a box with edges defined by *roi_limits*.

    Each state is associated with a diffusion coefficient, Hurst parameter, and
    occupation. This definition of FBM uses a modified Diffusion coefficient:

        modified diff. coef. = (diff. coef.) * dt^(2 * Hurst parameter - 1)

    As a result, the covariance between the i^th and j^th jumps in a trajectory is

        (modified diff. coef.) * dt * ( |i - j + 1|^{2H} + |i - j - 1|^{2H} - 2 * |i - j|^{2H} )

    This definition of the diffusion coefficient has the advantage that it absorbs
    the contribution of the Hurst parameter to the jump variance. Without this, the 
    Hurst parameter exerts a strong influence on the jump variance.

    init
    ----
        diff_coefs      :   1D ndarray, diffusion coefficients for each 
                            state in squared microns per second

        hurst_pars      :   1D ndarray, Hurst parameters for each state
                            (same number of elements as *diff_coefs*)

        state_occs      :   1D ndarray, the occupations of each state 
                            (same number of elements as *diff_coefs*)

        dt              :   float, simulation time interval in seconds
    """
    def __init__(
        self,
        diff_coefs: np.ndarray,
        hurst_pars: np.ndarray,
        state_occs: np.ndarray,
        dt: float,
        seed_region: Tuple[Tuple[int]]=DEFAULT_SEED_REGION
    ):
        super().__init__(dt=dt, seed_region=seed_region)
        self.diff_coefs = np.asarray(diff_coefs)
        self.hurst_pars = np.asarray(hurst_pars)
        self.state_occs = np.asarray(state_occs)
        if self.diff_coefs.shape != self.hurst_pars.shape:
            raise ValueError(f"incompatible shapes for diff_coefs and " \
                f"hurst_pars: {diff_coefs.shape}, {hurst_pars.shape}")
        elif self.diff_coefs.shape != self.hurst_pars.shape:
            raise ValueError(f"incompatible shapes for diff_coefs and " \
                f"state_occs: {diff_coefs.shape}, {state_occs.shape}")

    def sim(self, n_tracks: int, n_frames: int) -> (np.ndarray, np.ndarray):
        """ Simulate multiple trajectories, starting from the origin.

        Parameters
        ----------
            n_tracks    :   int, number of tracks to simulate
            n_frames    :   int, number of frames to simulate

        Returns
        -------
            (
                3D ndarray of shape (n_tracks, n_frames, 3),
                    the ZYX positions of each trajectory at
                    each timepoint in microns. All trajectories
                    start at 0;
                2D ndarray of shape (n_tracks, n_frames),
                    the state assignments for each trajectory
                    at each frame
            )
        """
        def make_covariance_matrix(diff_coef: float, hurst_par: float) -> np.ndarray:
            """ Generate the covariance matrix for the increments of a fractional Brownian
            motion with diffusion coefficient *diff_coef* and Hurst parameter *hurst_par*.

            Returns 2D numpy.ndarray of shape (n_frames, n_frames) """
            diff_coef_mod = diff_coef / np.power(self.dt, 2 * hurst_par - 1)
            T, S = np.indices((n_frames, n_frames)) + 1
            C = diff_coef * self.dt * (
                np.power(np.abs(T - S + 1), 2 * hurst_par) + \
                np.power(np.abs(T - S - 1), 2 * hurst_par) - \
                2 * np.power(np.abs(T - S), 2 * hurst_par)
            )
            return C

        tracks, states = [], []
        n_states = self.diff_coefs.shape[0]
        tracks_per_state = np.random.multinomial(n_tracks, self.state_occs)
        mean = np.zeros(n_frames, dtype=np.float64)

        for i, n in enumerate(tracks_per_state):
            T = np.zeros((n, n_frames, 3), dtype=np.float64)
            cov = make_covariance_matrix(self.diff_coefs[i], self.hurst_pars[i])
            for d in range(3):
                T[:,:,d] = np.random.multivariate_normal(mean, cov, size=tracks_per_state[i])
            tracks.append(T)
            states.append(np.full((n, n_frames), i, dtype=np.uint16))

        tracks = np.concatenate(tracks, axis=0)
        states = np.concatenate(states, axis=0)

        return np.cumsum(tracks, axis=1), states

MOTION_TYPES = {
    MOTION_TYPE_BROWNIAN: BrownianMotion,
    MOTION_TYPE_FRACTIONAL_BROWNIAN: FractionalBrownianMotion,
}

def make_motion(motion_type: str, dt: float, **kwargs):
    """ Factory method for Motion subclasses """
    if motion_type not in MOTION_TYPES:
        raise KeyError(f"motion type {motion_type} not recognized")
    return MOTION_TYPES[motion_type](dt=dt, **kwargs)
