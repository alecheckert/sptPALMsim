import os, gc, warnings, numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from functools import cached_property
from typing import Tuple, Callable
from .constants import (
    DEFAULT_WAVELENGTH,
    DEFAULT_NA,
    DEFAULT_REF_INDEX,
    DEFAULT_PIXEL_SIZE,
    DEFAULT_FOV_SIZE,
    DEFAULT_Z_LEVELS,
    DEFAULT_INTENSITY,
)

try:
    import cupy as cp 
except ImportError:
    cp = None

class Optics:
    """ Approximation to an optical system. """
    def __init__(
        self,
        wavelength: float=DEFAULT_WAVELENGTH,
        na: float=DEFAULT_NA,
        ref_index: float=DEFAULT_REF_INDEX,
        pixel_size: float=DEFAULT_PIXEL_SIZE,
        fov_size: Tuple[int]=DEFAULT_FOV_SIZE,
        pupil_func: Callable=None, 
        z_levels: np.ndarray=DEFAULT_Z_LEVELS,
        gpu: int=None,
        mode: str="noninterference",
        verbose: bool=True,
    ):
        assert len(fov_size) == 2
        assert mode in ["noninterference", "intensity", "amplitude"]
        self.wavelength = wavelength 
        self.na = na 
        self.ref_index = ref_index 
        self.pixel_size = pixel_size 
        self.fov_size = fov_size 
        self.z_levels = np.asarray(z_levels)
        self.nz = self.z_levels.shape[0]
        self.gpu = gpu
        self.mode = mode 
        self.verbose = verbose

        # Binning scheme
        self.yxbins = np.arange(self.fov_size[0] + 1) * self.pixel_size 

        # Maximum spatial frequency accepted by the objective in inverse microns
        self.k_na = 2 * np.pi * self.na / self.wavelength 

        # Spatial frequencies (inverse microns) in yx directions
        KY, KX = np.indices(self.fov_size)
        self.KY = (KY - (self.fov_size[0]-1)/2) * 2 * np.pi / (self.pixel_size * self.fov_size[0])
        self.KX = (KX - (self.fov_size[1]-1)/2) * 2 * np.pi / (self.pixel_size * self.fov_size[1])

        # Aperture function (defining the range of permissible frequencies
        # given this numerical aperture)
        self.aperture = (self.KY ** 2 + self.KX ** 2) <= (self.k_na ** 2)

        # Check for signs that the field of view is too small for accurate
        # diffraction simulations, given this numerical aperture
        if not self.aperture.any():
            warnings.warn("zero virtual pixels are within aperture. Try " \
                "increasing fov_size")
        elif self.aperture.sum() == 1:
            warnings.warn("only a single virtual pixel lies within the aperture. " \
                "Try increasing fov_size")

        # Pupil function. If not given, assume a perfect pupil (zero phase and 
        # unit transmission over the entire aperture)
        if pupil_func is None:
            self.pupil = self.aperture.astype(np.complex128)
        else:
            self.pupil = self.pupil_func(self.KY, self.KX)

        # Phase curvature (parabolic approximation)
        self.phase_parabola = np.sqrt((2 * np.pi * self.ref_index / self.wavelength) ** 2 - \
                self.KX ** 2 - self.KY ** 2 + 0j)

    def __enter__(self):
        return self 

    def __exit__(self, etype, evalue, traceback):
        return etype is None

    def make_defocus_kernel(self, z: float) -> np.ndarray:
        """ Return the defocus kernel at a particular axial offset.

        Parameters
        ----------
            z: offset from focus in microns

        Returns
        -------
            2D ndarray, dtype complex128, shape self.fov_size
        """
        result = self.pupil.copy()
        result[self.aperture] = result[self.aperture] * \
            np.exp(-1.0j * z * self.phase_parabola[self.aperture])
        return result 

    def psf(self, z: float) -> np.ndarray:
        """ Generate the point spread function at a particular axial offset.

        Parameters
        ----------
            z: offset from focus in microns

        Returns
        -------
            2D ndarray, dtype complex128, shape fov_size

        """
        return fftshift(fft2(ifftshift(self.make_defocus_kernel(z))))

    def psf_int(self, z: float) -> np.ndarray:
        """ Generate the intensity point spread function, proportional to the 
        squared modulus of the complex-valued point spread function.

        Parameters
        ----------
            z: offset from focus in microns

        Returns
        -------
            2D ndarray, dtype float64, shape fov_size
        """
        return np.abs(self.psf(z)) ** 2

    @property 
    def input_shape(self) -> Tuple[int]:
        """ Shape of numpy.ndarrays expected to be passed to Optics.image """
        return (self.nz, *self.fov_size)

    @property
    def use_gpu(self) -> bool:
        return (self.gpu is not None) and (cp is not None)

    @property 
    def xp(self):
        """ Polymorphic handle for either numpy or cupy, depending on whether
        we want to use a GPU """
        if self.use_gpu:
            xp = cp
            cp.cuda.Device(self.gpu).use()
        else:
            xp = np 
        return xp

    @cached_property 
    def z_bins(self) -> np.ndarray:
        """ A set of bins that evenly span the z-levels. Each element of 
        *self.z_levels* is the midpoint of a bin. Assumes that *z_levels*
        are evenly spaced and monotonic increasing.

        Returns
        -------
            1D numpy.ndarray of shape (self.nz+1,), microns
        """
        binsize = self.z_levels[1] - self.z_levels[0]
        return np.linspace(
            self.z_levels[0] - binsize * 0.5,
            self.z_levels[-1] + binsize * 0.5,
            self.nz + 1
        )

    @cached_property
    def defocus_kernels(self) -> np.ndarray:
        """
        Returns
        -------
            3D numpy.ndarray of shape (nz, ny, nx), frequency domain 
                defocus kernels. The last dimension (nx) will depend on the
                imaging mode
        """
        xp = self.xp
        xfft = xp.fft

        # Show cache size
        if self.verbose:
            if self.mode == "noninterference": 
                total_pixels = self.nz * self.fov_size[0] * (self.fov_size[1]//2+1)
            else:
                total_pixels = self.nz * self.fov_size[0] * self.fov_size[1]
            mbytes = (total_pixels * 16) * 1.0e-6
            print("\nTotal cache size: {:.2} MB".format(mbytes))

        # Calculate defocus kernels
        if self.mode == "noninterference": 
            out = xp.zeros((self.nz, self.fov_size[0], self.fov_size[1]//2+1), dtype=xp.complex128)
            for i, z in enumerate(self.z_levels):
                kernel = xp.abs(xfft.fftshift(xfft.ifft2(xp.asarray(self.make_defocus_kernel(z)))))**2
                out[i,:,:] = xfft.rfft2(kernel/kernel.sum())     

        else:
            out = xp.zeros((self.nz, *self.fov_size), dtype=xp.complex128)
            for i, z in enumerate(self.z_levels):
                out[i,:,:] = xp.asarray(self.make_defocus_kernel(z))

        return out

    def image(self, src: np.ndarray) -> np.ndarray:
        """ Image a 3D distribution of emitters to a 2D plane.

        Parameters
        ----------
            src     :   3D numpy.ndarray of shape (nz, *self.fov_size), the 
                        distribution of emitters. *nz* must be equal to the 
                        number of elements in *self._cache_z_levels*

        Returns
        -------
            2D numpy.ndarray of shape *self.fov_size*, the resulting image 
        """
        assert src.shape == self.input_shape, f"expected {self.input_shape}; got {src.shape}"
        xp = self.xp
        xfft = xp.fft

        src = xp.asarray(src)
        dtype = xp.complex128 if (self.mode == "amplitude") else xp.float64
        result = xp.zeros(self.fov_size, dtype=dtype)

        if self.mode == "noninterference":
            result = xfft.fftshift(xfft.irfft2(
                xfft.rfft2(src, axes=(1,2)) * self.defocus_kernels, s=self.fov_size, axes=(1,2)
            ), axes=(1,2)).sum(axis=0)
        elif self.mode == "intensity":
            result = (np.abs(xfft.ifft2(
                xfft.fft2(src, axes=(1,2)) * self.defocus_kernels, s=self.fov_size, axes=(1,2)
            ))**2).sum(axis=0)
        elif self.mode == "amplitude":
            result = xfft.ifft2(
                xfft.fft2(src, axes=(1,2)) * self.defocus_kernels, s=self.fov_size, axes=(1,2)
            ).sum(axis=0)

        if xp is cp:
            return cp.asnumpy(result)
        else:
            return result

    def image_point_sources(self, points: np.ndarray, intensity: float=DEFAULT_INTENSITY) -> np.ndarray:
        """
        Parameters
        ----------
            points      :   2D numpy.ndarray of shape (n_points, ZYX), the ZYX
                            coordinates of each particle in microns
            intensity   :   mean number of points per particle per frame

        Returns
        -------
            2D numpy.ndarray of shape *self.fov_size*, synthesized image
        """
        if points.shape[0] == 0:
            return np.zeros(self.fov_size, dtype=np.float64)

        # Histogram of particle positions
        H = np.histogramdd(points, bins=(self.z_bins, self.yxbins, self.yxbins)
            )[0].astype(np.float64) * intensity

        # Generate image
        return self.image(H)

