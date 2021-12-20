import numpy as np 
from typing import Tuple
from .constants import DEFAULT_FOV_SIZE

class Camera:
    """ Simple camera noise model. Assumes two sources of noise: shot noise (Poisson)
    and read noise (Gaussian). 

    init
    ----
        offset          :   pixel offset in grayvalues. Either a single value (for all
                            pixels) or a 2D numpy.ndarray (for each pixel)

        gain            :   pixel gain in grayvalues/photon. Either a single value (for
                            all pixels) or a 2D numpy.ndarray (for each pixel)

        read_noise      :   pixel read noise (root variance in grayvalues). Either a 
                            single value (for all pixels) or a 2D numpy.ndarray (for
                            each pixel)

        fov_size        :   2-tuple of int, FOV shape in pixels (YX)

    usage
    -----
        >>> fov_size = (128, 128)

        # EMCCD-like
        >>> camera = Camera(470.0, 109.0, 1.0, fov_size=fov_size)

        # Some kind of probability distribution over photon emissions
        >>> photons = np.random.normal(size=fov_size)

        # Add camera noise
        >>> grayvalues = camera(photons)
    """
    def __init__(self, offset, gain, read_noise, fov_size: Tuple[int]=DEFAULT_FOV_SIZE):
        self.fov_size = fov_size
        for attr, val in zip(["offset", "gain", "read_noise"], [offset, gain, read_noise]):
            self._process_attr(attr, val)

    def _process_attr(self, attr, value):
        if isinstance(value, float):
            setattr(self, attr, np.ones(self.fov_size) * value)
        elif isinstance(value, np.ndarray):
            assert value.shape == self.fov_size
            setattr(self, attr, value)
        else:
            raise ValueError(f"unrecognized argument for {attr}: {value}")

    def __call__(self, I: np.ndarray) -> np.ndarray:
        assert I.shape == self.fov_size 
        I[I<0] = 0
        AU = np.random.poisson(self.gain * I) + self.offset + \
            np.random.normal(loc=0, scale=self.read_noise)
        AU[AU<0] = 0
        return AU 

    def __enter__(self):
        return self 

    def __exit__(self, etype, evalue, traceback):
        return etype is None