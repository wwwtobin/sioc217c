# Physical constants
SIGMA = 5.67e-8  # Stefan-Boltzmann constant (W m^-2 K^-4)
G0 = 9.81  # gravitational acceleration (m s^-2)
C_P = 1004.0  # specific heat of dry air at constant pressure (J kg^-1 K^-1)
L_V = 2.5e6  # latent heat of vaporization (J kg^-1)
R_V = 461.0  # gas constant for water vapor (J kg^-1 K^-1)
EPSILON = 0.622  # ratio of water vapor to dry air molecular weight
E_SAT_REF = 611.0  # saturation vapor pressure at T_REF (Pa)
T_REF = 273.0  # reference temperature (K)

# Boundary layer parameters
GAMMA_D = 0.00977  # dry adiabatic lapse rate (K m^-1)
C_SH = 0.001  # sensible heat exchange coefficient
C_LH = 0.001  # latent heat exchange coefficient
RHO_AIR = 1.2  # surface air density (kg m^-3)

# Slab ocean parameters
H_OCEAN = 1.0  # slab ocean depth (m)
C_WATER = 4184.0  # specific heat of liquid water (J kg^-1 K^-1)
RHO_WATER = 1000.0  # density of liquid water (kg m^-3)

SECONDS_PER_DAY = 60.0 * 60.0 * 24.0