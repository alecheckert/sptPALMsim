import numpy as np 

# Default FOV size/shape in pixels (YX)
DEFAULT_FOV_SIZE = (128, 128)

# Default pixel size in microns
DEFAULT_PIXEL_SIZE = 0.16

# Default frame interval in seconds
DEFAULT_FRAME_INTERVAL = 0.00748

# Default excitation pulse width
DEFAULT_PULSE_WIDTH = 0.002

# Default bleach rate in Hz
DEFAULT_BLEACH_RATE = 1.0

# Size of the ZYX region in which to seed trajectories in microns
DEFAULT_SEED_REGION = (
    (-2.5, 2.5),
    (-2.0, 25.0),
    (-2.0, 25.0),
)

# Default emission wavelength in microns
DEFAULT_WAVELENGTH = 0.59

# Default numerical aperture
DEFAULT_NA = 1.49

# Default immersion media refractive index
DEFAULT_REF_INDEX = 1.515

# Discrete z-planes into which to divide the simulation (in microns)
DEFAULT_Z_LEVELS = np.linspace(-2.5, 2.5, 201)

# Mean number of photons emitted per particle per frame
DEFAULT_INTENSITY = 200.0

# Motion simulator handles
MOTION_TYPE_BROWNIAN = "brownian"

# Default camera parameters; similar to Andor iXon 897 EMCCD
DEFAULT_OFFSET = 470.0
DEFAULT_GAIN = 109.0
DEFAULT_READ_NOISE = 1.0

# Spatial and temporal discretization, per camera pixel or frame interval
# respectively
DEFAULT_SPATIAL_BIN_RATE = 4
DEFAULT_TEMPORAL_BIN_RATE = 60