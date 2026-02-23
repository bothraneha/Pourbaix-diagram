#!/usr/bin/env python3
"""
Pourbaix diagram generator for IrO2 surfaces (101, 110, 111).

This script computes surface free energies (Δγ) vs. potential (RHE)
for multiple coverages on three facets and plots them together.

Key outputs:
- corrected_H2O_included.pdf
- corrected_H2O_included.png
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


# -------------------------
# Thermodynamic parameters
# -------------------------
@dataclass(frozen=True)
class GasEnergies:
    h2: float
    h2o: float
    zpe_h2o: float
    zpe_h2: float
    cv_h2o: float
    cv_h2: float
    ts_h2o: float
    ts_h2: float


@dataclass(frozen=True)
class AdsorbateCorrections:
    zpe_oh: float
    zpe_o: float
    zpe_ooh: float
    cv_oh: float
    cv_o: float
    cv_ooh: float
    ts_oh: float
    ts_o: float
    ts_ooh: float


@dataclass(frozen=True)
class SurfaceSet:
    name: str
    # entries: [energy, nH, nO, nOH, (optional) nH2O, surface_id]
    entries: Sequence[Sequence[float]]
    area_2A: float


@dataclass(frozen=True)
class PlotConfig:
    u_min: float
    u_max: float
    u_step: float
    fig_size: Tuple[float, float]
    dpi: int


# -------------------------
# Input data (from VASP)
# -------------------------
GAS = GasEnergies(
    h2=-6.77149190,
    h2o=-14.23091949,
    zpe_h2o=0.560,
    zpe_h2=0.268,
    cv_h2o=0.103,
    cv_h2=0.0905,
    ts_h2o=0.675,
    ts_h2=0.408,
)

ADS = AdsorbateCorrections(
    zpe_oh=0.389,
    zpe_o=0.098,
    zpe_ooh=0.478,
    cv_oh=0.032,
    cv_o=0.017,
    cv_ooh=0.066,
    ts_oh=0.047,
    ts_o=0.023,
    ts_ooh=0.112,
)

# Energy, H O OH surface
SURF_101 = SurfaceSet(
    name="101",
    entries=[
        [-338.15785856, 0, 0, 0, 101],
        [-326.92290810, 0, 0, 0, 101],
        [-354.33466082, 0, 3, 1, 101],
        [-349.01078002, 0, 4, 0, 101],
        [-389.61526835, 4, 0, 4, 101],
        [-300.77828477, 0, -4, 0, 101],
    ],
    area_2A=50.34650000 * 2,
)

# Energy, H O OH surface H2O
SURF_110 = SurfaceSet(
    name="110",
    entries=[
        [-338.15785856, 0, 0, 0, 110, 0],
        [-330.48130082, 0, 0, 0, 110, 0],
        [-369.85616722, 2, 0, 0, 110, 2],
        [-366.66921928, 2, 0, 1, 110, 1],
        [-361.82300717, 2, 0, 2, 110, 0],
        [-356.66353027, 1, 1, 2, 110, 0],
        [-352.86492181, 0, 0, 2, 110, 0],
        [-347.57272733, 0, 1, 1, 110, 0],
        [-342.49476218, 0, 2, 0, 110, 0],
    ],
    area_2A=40.95766000 * 2,
)

# Energy, H O OH surface
SURF_111 = SurfaceSet(
    name="111",
    entries=[
        [-422.69732320, 0, 0, 0, 111],
        [-405.67370298, 0, 0, 0, 111],
        [-432.14909533, 0, 4, 0, 111],
    ],
    area_2A=57.90115977 * 2,
)


# -------------------------
# Thermodynamic helpers
# -------------------------
KB_T = 0.0256
CONST = KB_T * math.log(10.0)


def gas_gibbs_corrections(gas: GasEnergies) -> Tuple[float, float]:
    """Return (ΔG_H2O, ΔG_H2) corrections for gas molecules."""
    dg_h2o = gas.zpe_h2o + gas.cv_h2o - gas.ts_h2o
    dg_h2 = gas.zpe_h2 + gas.cv_h2 - gas.ts_h2
    return dg_h2o, dg_h2


def adsorbate_corrections(ads: AdsorbateCorrections, dg_h2o: float, dg_h2: float) -> Tuple[float, float, float, float]:
    """Return (ΔG_O, ΔG_OH, ΔG_OOH, ΔG_H) corrections for adsorbates."""
    dso = ads.zpe_o + ads.cv_o - ads.ts_o - (dg_h2o - dg_h2)
    dsoh = ads.zpe_oh + ads.cv_oh - ads.ts_oh - (dg_h2o - 0.5 * dg_h2)
    dsooh = ads.zpe_ooh + ads.cv_ooh - ads.ts_ooh - (2 * dg_h2o - 1.5 * dg_h2)
    dsh = dsoh - dso
    return dso, dsoh, dsooh, dsh


def add_o(pH: float, u: float, dso: float) -> float:
    return -(GAS.h2o - GAS.h2) - 2 * (u + pH * CONST) + dso


def add_oh(pH: float, u: float, dsoh: float) -> float:
    return -(GAS.h2o - 0.5 * GAS.h2) - (u + pH * CONST) + dsoh


def add_ooh(pH: float, u: float, dsooh: float) -> float:
    return -(2 * GAS.h2o - 1.5 * GAS.h2) - 3 * (u + pH * CONST) + dsooh


def add_h2o() -> float:
    return -GAS.h2o - (GAS.zpe_h2o - GAS.ts_h2o)


def add_h(pH: float, u: float, dsh: float) -> float:
    return -0.5 * GAS.h2 + (u + pH * CONST) + dsh


def surface_free_energy_101(i: int, pH: float, u: float, dso: float, dsoh: float, dsh: float) -> float:
    entry = SURF_101.entries[i]
    return (
        entry[0]
        - SURF_101.entries[0][0]
        + entry[1] * add_h(pH, u, dsh)
        + entry[2] * add_o(pH, u, dso)
        + entry[3] * add_oh(pH, u, dsoh)
        - 0.01
    ) / SURF_101.area_2A


def surface_free_energy_110(i: int, pH: float, u: float, dso: float, dsoh: float, dsh: float) -> float:
    entry = SURF_110.entries[i]
    return (
        entry[0]
        - SURF_110.entries[0][0]
        + entry[1] * add_h(pH, u, dsh)
        + entry[2] * add_o(pH, u, dso)
        + entry[3] * add_oh(pH, u, dsoh)
        + entry[5] * add_h2o()
        - 0.003
    ) / SURF_110.area_2A


def surface_free_energy_111(i: int, pH: float, u: float, dso: float, dsoh: float, dsh: float) -> float:
    entry = SURF_111.entries[i]
    return (
        entry[0]
        - SURF_111.entries[0][0]
        + entry[1] * add_h(pH, u, dsh)
        + entry[2] * add_o(pH, u, dso)
        + entry[3] * add_oh(pH, u, dsoh)
        - 0.013
    ) / SURF_111.area_2A


# -------------------------
# Analysis utilities
# -------------------------

def find_lowest_surfaces(u_values: Iterable[float], pH: float, dso: float, dsoh: float, dsh: float) -> List[int]:
    lowest = []
    for u in u_values:
        values: List[float] = []
        for k in range(len(SURF_101.entries)):
            values.append(surface_free_energy_101(k, pH, u, dso, dsoh, dsh))
        for k in range(len(SURF_110.entries)):
            values.append(surface_free_energy_110(k, pH, u, dso, dsoh, dsh))
        for k in range(len(SURF_111.entries)):
            values.append(surface_free_energy_111(k, pH, u, dso, dsoh, dsh))
        lowest.append(int(np.argmin(values)))
    return lowest


def print_min_values(u_targets: Sequence[float], pH: float, dso: float, dsoh: float, dsh: float) -> None:
    print("\nLowest dg1[101], dg2[110], and dg3[111] at Specific Potentials:")
    print("-" * 40)
    print("Potential (V)\tMin dg1 (101)\tMin dg2 (110)\tMin dg3 (111)")
    for u in u_targets:
        dg1_values = [surface_free_energy_101(k, pH, u, dso, dsoh, dsh) for k in range(1, len(SURF_101.entries))]
        dg2_values = [surface_free_energy_110(k, pH, u, dso, dsoh, dsh) for k in range(1, len(SURF_110.entries))]
        dg3_values = [surface_free_energy_111(k, pH, u, dso, dsoh, dsh) for k in range(1, len(SURF_111.entries))]
        print(f"{u:.2f}\t\t{min(dg1_values):.4f}\t\t\t{min(dg2_values):.4f}\t\t\t{min(dg3_values):.4f}")


# -------------------------
# Plotting
# -------------------------

def plot_surfaces(pH: float, config: PlotConfig, dso: float, dsoh: float, dsh: float) -> None:
    u_vals = np.arange(config.u_min, config.u_max + config.u_step, config.u_step)

    color_101 = ["maroon", "red", "gray", "sienna", "darksalmon", "wheat", "rosybrown", "lightsalmon", "peru"]
    color_110 = ["darkgreen", "aqua", "limegreen", "gold", "peru", "yellowgreen", "deepskyblue", "teal", "blue", "darkorchid"]
    color_111 = ["navy", "blue", "fuchsia", "mediumpurple", "darkviolet", "violet"]

    fig = plt.figure(figsize=config.fig_size, dpi=config.dpi)
    ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])

    ax.set_xlim(0.0, 2.00)
    ax.set_ylim(-0.02, 0.26)

    ymin, ymax = ax.get_ylim()
    ax.set_yticks(np.arange(ymin, ymax, 0.02))

    xmin, xmax = ax.get_xlim()
    ax.set_xticks(np.arange(xmin, xmax, 0.2))

    ax.set_xlabel("RHE (V)", fontsize=9)
    ax.set_ylabel(r"$\Delta \gamma$ (eV/A2)", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    xx = np.arange(0.0, 2.2, 0.05)

    for k in range(len(SURF_101.entries)):
        entry = SURF_101.entries[k]
        label = rf"S$_{{{k}}}$(H: {int(entry[1])} O: {int(entry[2])} OH: {int(entry[3])} surf: {int(entry[4])})"
        ax.plot(xx, [surface_free_energy_101(k, pH, u, dso, dsoh, dsh) for u in xx], "-", lw=1, c=color_101[k], label=label)

    for k in range(len(SURF_110.entries)):
        entry = SURF_110.entries[k]
        label = rf"S$_{{{k}}}$(H: {int(entry[1])} O: {int(entry[2])} OH: {int(entry[3])} H2O: {int(entry[5])} surf: {int(entry[4])})"
        ax.plot(xx, [surface_free_energy_110(k, pH, u, dso, dsoh, dsh) for u in xx], "--", lw=1, c=color_110[k], label=label)

    for k in range(len(SURF_111.entries)):
        entry = SURF_111.entries[k]
        label = rf"S$_{{{k}}}$(H: {int(entry[1])} O: {int(entry[2])} OH: {int(entry[3])} surf: {int(entry[4])})"
        ax.plot(xx, [surface_free_energy_111(k, pH, u, dso, dsoh, dsh) for u in xx], "-.", lw=1, c=color_111[k], label=label)

    ax.legend(bbox_to_anchor=(0.00, 1.16), loc=2, borderaxespad=0.0, ncol=3, fancybox=True, shadow=True, fontsize=3.5, handlelength=2)

    fig.savefig("corrected_H2O_included.pdf", bbox_inches="tight")
    fig.savefig("corrected_H2O_included.png", bbox_inches="tight")
    plt.show()


# -------------------------
# Main execution
# -------------------------

def main() -> None:
    dg_h2o, dg_h2 = gas_gibbs_corrections(GAS)
    dso, dsoh, dsooh, dsh = adsorbate_corrections(ADS, dg_h2o, dg_h2)

    print(f"Adsorbate corrections: dsoh={dsoh:.4f}, dso={dso:.4f}, dsh={dsh:.4f}")

    u_min = -0.6
    u_max = 2.2
    pH_value = 0.0

    u_targets = [0.06, 1.2, 1.5, 1.6]
    print_min_values(u_targets, pH_value, dso, dsoh, dsh)

    config = PlotConfig(u_min=u_min, u_max=u_max, u_step=0.01, fig_size=(6, 6), dpi=300)
    plot_surfaces(pH_value, config, dso, dsoh, dsh)


if __name__ == "__main__":
    main()
