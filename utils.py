import copy

import numpy as np
from constants import *


def transmission_to_one_level(
    tau_levels: np.ndarray, single_tau: float, r: float
) -> np.ndarray:
    """Compute transmission between a single level and all other levels."""
    return np.exp(-r * abs(tau_levels - single_tau))


def transmission_to_all_levels(tau_levels: np.ndarray, r: float) -> np.ndarray:
    """Compute the full level-to-level transmission matrix."""
    return np.exp(-r * np.abs(tau_levels[:, None] - tau_levels[None, :]))


def surface_flux_to_all_levels(Ts: float, trans_levels: np.ndarray) -> np.ndarray:
    """Compute upward longwave flux from the surface to all levels."""
    Bs = SIGMA * Ts**4
    Fs_levels = Bs * trans_levels
    return Fs_levels


def upward_flux_to_all_levels(
    T_layers: np.ndarray,
    emissivity_layers: np.ndarray,
    trans_levels_to_levels: np.ndarray,
) -> np.ndarray:
    """Compute net upward atmospheric longwave flux at all levels."""
    num_levels = T_layers.size + 1
    Fup_levels = np.zeros(num_levels)
    for l in range(1, num_levels):
        Fup_levels[l] = np.sum(
            SIGMA
            * T_layers[:l] ** 4
            * emissivity_layers[:l]
            * trans_levels_to_levels[l, 1 : l + 1]
        )
    return Fup_levels


def downward_flux_to_all_levels(
    T_layers: np.ndarray,
    emissivity_layers: np.ndarray,
    trans_levels_to_levels: np.ndarray,
) -> np.ndarray:
    """Compute net downward atmospheric longwave flux at all levels."""
    num_levels = T_layers.size + 1
    Fdown_levels = np.zeros(num_levels)
    for l in range(0, num_levels - 1):
        Fdown_levels[l] = -np.sum(
            SIGMA
            * T_layers[l:] ** 4
            * emissivity_layers[l:]
            * trans_levels_to_levels[l, l:-1]
        )
    return Fdown_levels


def net_terrestrial_flux_at_all_levels(
    Ts: float,
    T_layers: np.ndarray,
    emissivity_layers: np.ndarray,
    trans_levels_to_levels: np.ndarray,
) -> np.ndarray:
    """Compute the net longwave flux at all levels (surface + upward + downward)."""
    Fs_levels = surface_flux_to_all_levels(Ts, trans_levels_to_levels[0, :])
    Fup_levels = upward_flux_to_all_levels(
        T_layers, emissivity_layers, trans_levels_to_levels
    )
    Fdown_levels = downward_flux_to_all_levels(
        T_layers, emissivity_layers, trans_levels_to_levels
    )
    F_LW_levels = Fs_levels + Fup_levels + Fdown_levels
    return F_LW_levels


def flux_divergence(F_levels: np.ndarray, del_p: float) -> np.ndarray:
    """Compute flux divergence between levels in pressure coordinates.

    Sign convention: [:-1] minus [1:] because pressure decreases with altitude,
    so level i has larger pressure than level i+1.
    """
    F_div_levels = -(F_levels[:-1] - F_levels[1:]) / del_p
    return F_div_levels


def dry_static_energy(T_layers: np.ndarray, z_layers: np.ndarray) -> np.ndarray:
    """Compute dry static energy (J kg^-1) for each layer."""
    s_layers = G0 * z_layers + C_P * T_layers
    return s_layers


def rate_of_energy_change(F_div_layers: np.ndarray) -> np.ndarray:
    """Convert pressure-coordinate flux divergence to energy tendency (J kg^-1 day^-1)."""
    dEdt = -F_div_layers * G0 * SECONDS_PER_DAY
    return dEdt


def T_from_dry_static_energy(s_layers: np.ndarray, z_layers: np.ndarray) -> np.ndarray:
    """Recover layer temperatures from dry static energy and geopotential height."""
    T_layers = (s_layers - G0 * z_layers) / C_P
    return T_layers


def rate_of_surface_temperature_change(F_surface: float) -> float:
    """Compute slab ocean surface temperature tendency (K day^-1) from net surface flux."""
    dTsdt = -F_surface / (RHO_WATER * H_OCEAN * C_WATER) * SECONDS_PER_DAY
    return dTsdt


def radiative_equilibrium_temperature(
    Ts: float,
    T_layers: np.ndarray,
    del_t: float,
    del_p: float,
    emissivity_layers: np.ndarray,
    trans_levels_to_levels: np.ndarray,
    z_layers: np.ndarray,
    F_SW: float,
) -> tuple[float, np.ndarray]:
    """Iterate toward radiative equilibrium.

    Returns the equilibrium surface temperature and layer temperatures once the
    net flux imbalance at the surface and TOA both fall below 0.1 W m^-2.
    """
    T_layers_copy = copy.deepcopy(T_layers)
    Ts_copy = copy.deepcopy(Ts)
    s_layers = dry_static_energy(T_layers_copy, z_layers)
    max_F = 1.0
    iteration_threshold = 0.1
    while max_F > iteration_threshold:
        F_LW_levels = net_terrestrial_flux_at_all_levels(
            Ts_copy,
            T_layers_copy,
            emissivity_layers,
            trans_levels_to_levels,
        )
        F_div_layers = flux_divergence(F_LW_levels, del_p)
        dsdt_layers = rate_of_energy_change(F_div_layers)
        ds_layers = dsdt_layers * del_t
        s_layers += ds_layers
        T_layers_copy = T_from_dry_static_energy(s_layers, z_layers)
        dTsdt = rate_of_surface_temperature_change(F_SW + F_LW_levels[0])
        dTs = dTsdt * del_t
        Ts_copy += dTs
        max_F = np.amax(
            [
                np.abs(F_SW + F_LW_levels[0]),
                np.abs(F_SW + F_LW_levels[-1]),
            ]
        )
    print("Done!")
    return Ts_copy, T_layers_copy


def temperature_at_2m(T1: float, z1: float) -> float:
    """Extrapolate air temperature to 2 m using the dry adiabatic lapse rate."""
    T2m = T1 + GAMMA_D * (z1 - 2.0)
    return T2m


def sensible_heat_flux(Ts: float, T2m: float, V0: float) -> float:
    """Compute surface sensible heat flux (W m^-2)."""
    F_SH = C_SH * C_P * RHO_AIR * V0 * (Ts - T2m)
    return F_SH


def dry_convective_buoyancy_up(
    s_layers: np.ndarray,
    F_SH: float,
    del_t: float,
    del_p: float,
) -> np.ndarray:
    """Apply dry convective adjustment, distributing sensible heat over unstable bottom layers."""
    s_layers_dry = copy.deepcopy(s_layers)
    dSH = F_SH * SECONDS_PER_DAY * del_t
    dSH_single = dSH * G0 / del_p
    N = 1
    unstable = True
    while unstable:
        dSH_N = dSH_single / N
        s_bottom_layers = np.average(s_layers[0:N])
        s_layers_dry[0:N] = s_bottom_layers + dSH_N
        s_layers_dry[N:] = s_layers[N:]
        s_layers_diff = np.diff(s_layers_dry)
        if np.any(s_layers_diff < 0):
            N += 1
        else:
            unstable = False
    return s_layers_dry


def radiative_dry_convective_equilibrium_temperature(
    Ts: float,
    T_layers: np.ndarray,
    del_t: float,
    del_p: float,
    emissivity_layers: np.ndarray,
    trans_levels_to_levels: np.ndarray,
    F_SW: float,
    z_layers: np.ndarray,
    V0: float,
) -> tuple[float, np.ndarray]:
    """Iterate toward radiative-dry convective equilibrium.

    Returns equilibrium (Ts, T_layers) once surface and TOA flux imbalances
    both fall below 0.1 W m^-2.
    """
    T_layers_copy = copy.deepcopy(T_layers)
    Ts_copy = copy.deepcopy(Ts)
    max_F = 1.0
    iteration_threshold = 0.1
    while max_F > iteration_threshold:
        F_LW_levels = net_terrestrial_flux_at_all_levels(
            Ts_copy,
            T_layers_copy,
            emissivity_layers,
            trans_levels_to_levels,
        )
        F_div_layers = flux_divergence(F_LW_levels, del_p)
        s_layers = dry_static_energy(T_layers_copy, z_layers)
        dsdt_layers = rate_of_energy_change(F_div_layers)
        ds_layers = dsdt_layers * del_t
        s_layers_rad = s_layers + ds_layers
        T2m = temperature_at_2m(T_layers_copy[0], z_layers[0])
        F_SH = sensible_heat_flux(Ts_copy, T2m, V0)
        s_layers_rad_dry = dry_convective_buoyancy_up(
            s_layers_rad,
            F_SH,
            del_t,
            del_p,
        )
        T_layers_copy = T_from_dry_static_energy(s_layers_rad_dry, z_layers)
        dTsdt = rate_of_surface_temperature_change(F_SW + F_LW_levels[0] + F_SH)
        dTs = dTsdt * del_t
        Ts_copy += dTs
        max_F = np.amax(
            [
                np.abs(F_SW + F_LW_levels[0] + F_SH),
                np.abs(F_SW + F_LW_levels[-1]),
            ]
        )
    print("Done!")
    return Ts_copy, T_layers_copy


def saturation_mixing_ratio(
    T: float | np.ndarray, p: float | np.ndarray
) -> float | np.ndarray:
    """Compute saturation mixing ratio via Clausius-Clapeyron.

    Clips esat to keep qsat < 1 and caps qsat at 0.0015 above 100 hPa
    to avoid unphysically large stratospheric values.
    """
    esat = E_SAT_REF * np.exp((L_V / R_V) * ((1.0 / T_REF) - (1 / T)))
    esat = np.clip(esat, 0.0, p / (1 + EPSILON))
    qsat = EPSILON * esat / (p - esat)
    if np.any(p < 10000.0):
        if np.ndim(p) == 0:
            qsat = 0.0015
        else:
            p_index = np.where(p < 10000.0)
            if np.size(p_index) > 1:
                qsat[p_index] = 0.0015
            else:
                qsat = 0.0015
    return qsat


def latent_heat_flux(Ts: float, T2m: float, p0: float, V0: float, RH0: float) -> float:
    """Compute surface latent heat flux (W m^-2)."""
    qs = saturation_mixing_ratio(Ts, p0)
    q2m = RH0 * saturation_mixing_ratio(T2m, p0)
    F_LH = C_LH * L_V * RHO_AIR * V0 * (qs - q2m)
    return F_LH


def saturated_moist_static_energy(
    T_layers: np.ndarray, p_layers: np.ndarray, z_layers: np.ndarray
) -> np.ndarray:
    """Compute saturated moist static energy (J kg^-1) for each layer."""
    qsat_layers = saturation_mixing_ratio(T_layers, p_layers)
    hsat_layers = G0 * z_layers + C_P * T_layers + L_V * qsat_layers
    return hsat_layers


def T_from_saturated_moist_static_energy(
    hsat_layers: np.ndarray, p_layers: np.ndarray, z_layers: np.ndarray
) -> np.ndarray:
    """Recover layer temperatures from saturated moist static energy via Newton iteration."""
    T_layers = T_from_dry_static_energy(hsat_layers, z_layers)
    for l in range(0, hsat_layers.size):
        dT = 1.0
        iteration_threshold = 0.0001
        while np.abs(dT) > iteration_threshold:
            new_hsat = saturated_moist_static_energy(
                T_layers[l], p_layers[l], z_layers[l]
            )
            dT = (hsat_layers[l] - new_hsat) / hsat_layers[l]
            T_layers[l] += dT
    return T_layers


def moist_convective_buoyancy_up(
    hsat_layers: np.ndarray,
    F_SH_LH: float,
    del_t: float,
    del_p: float,
) -> np.ndarray:
    """Apply moist convective adjustment, distributing sensible + latent heat over unstable bottom layers."""
    hsat_layers_conv = copy.deepcopy(hsat_layers)
    dSH_LH = F_SH_LH * SECONDS_PER_DAY * del_t
    dSH_LH_single = dSH_LH * G0 / del_p
    N = 1
    unstable = True
    while unstable:
        dSH_LH_N = dSH_LH_single / N
        hsat_bottom_layers = np.average(hsat_layers[0:N])
        hsat_layers_conv[0:N] = hsat_bottom_layers + dSH_LH_N
        hsat_layers_conv[N:] = hsat_layers[N:]
        hsat_layers_diff = np.diff(hsat_layers_conv)
        if np.any(hsat_layers_diff < 0):
            N += 1
        else:
            unstable = False
    return hsat_layers_conv


def radiative_convective_equilibrium_temperature(
    Ts: float,
    T_layers: np.ndarray,
    del_t: float,
    del_p: float,
    emissivity_layers: np.ndarray,
    trans_levels_to_levels: np.ndarray,
    F_SW: float,
    p_layers: np.ndarray,
    z_layers: np.ndarray,
    p0: float,
    V0: float,
    RH0: float,
) -> tuple[float, np.ndarray]:
    """Iterate toward radiative-moist convective equilibrium.

    Returns equilibrium (Ts, T_layers) once surface and TOA flux imbalances
    both fall below 0.5 W m^-2.
    """
    T_layers_copy = copy.deepcopy(T_layers)
    Ts_copy = copy.deepcopy(Ts)
    dF = 1.0
    iteration_threshold = 0.5
    while dF > iteration_threshold:
        F_LW_levels = net_terrestrial_flux_at_all_levels(
            Ts_copy,
            T_layers_copy,
            emissivity_layers,
            trans_levels_to_levels,
        )
        F_div_layers = flux_divergence(F_LW_levels, del_p)
        hsat_layers = saturated_moist_static_energy(T_layers_copy, p_layers, z_layers)
        dhsatdt_layers = rate_of_energy_change(F_div_layers)
        dhsat_layers = dhsatdt_layers * del_t
        hsat_layers_rad = hsat_layers + dhsat_layers
        T2m = temperature_at_2m(T_layers_copy[0], z_layers[0])
        F_SH = sensible_heat_flux(Ts_copy, T2m, V0)
        F_LH = latent_heat_flux(Ts_copy, T2m, p0, V0, RH0)
        F_SH_LH = F_SH + F_LH
        hsat_layers_rad_conv = moist_convective_buoyancy_up(
            hsat_layers_rad,
            F_SH_LH,
            del_t,
            del_p,
        )
        T_layers_copy = T_from_saturated_moist_static_energy(
            hsat_layers_rad_conv,
            p_layers,
            z_layers,
        )
        dTsdt = rate_of_surface_temperature_change(F_SW + F_LW_levels[0] + F_SH_LH)
        dTs = dTsdt * del_t
        Ts_copy += dTs
        dF = np.amax(
            [
                np.abs(F_SW + F_LW_levels[0] + F_SH_LH),
                np.abs(F_SW + F_LW_levels[-1]),
            ]
        )
    print("Done!")
    return Ts_copy, T_layers_copy
