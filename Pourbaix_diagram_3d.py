#!/usr/bin/env python3
"""
Pd(111) surface Pourbaix (2D pH–U map with phase regions).

Outputs:
- pd111_pourbaix_map.png
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# ============================================================
# GRID (pH, U)
# ============================================================
pH = np.linspace(0, 14, 2000)
U = np.linspace(-2.0, 2.5, 2000)

# ============================================================
# CONSTANTS & REFERENCES
# ============================================================
AREA_Pd111 = 27.049508  # Å² (two-sided Pd(111))

h2 = -6.77149190
h2o = -14.23091949

zpeh2o, zpeh2 = 0.560, 0.268
cvh2o, cvh2 = 0.103, 0.0905
tsh2o, tsh2 = 0.675, 0.408
zpeoh, zpeo, zpeooh = 0.389, 0.098, 0.478
cvoh, cvo, cvooh = 0.032, 0.017, 0.066
tsoh, tso, tsooh = 0.047, 0.023, 0.112

kbt = 0.0256
const = kbt * np.log(10)

# Gas corrections (CHE)
dgh2o = zpeh2o + cvh2o - tsh2o
dgh2 = zpeh2 + cvh2 - tsh2

dso = zpeo + cvo - tso - (dgh2o - dgh2)
dsoh = zpeoh + cvoh - tsoh - (dgh2o - 0.5 * dgh2)
dsooh = zpeooh + cvooh - tsooh - (2.0 * dgh2o - 1.5 * dgh2)
dsh = dsoh - dso

# ============================================================
# Pd(111) DFT REFERENCE ENERGIES (eV)
# ============================================================
CleanSurface_E0 = -78.9704

H2O1_0_E, H2O0_75_E, H2O0_5_E, H2O0_25_E = -137.5586, -122.5659, -108.0114, -93.3539
OH1_0_E, OH0_75_E, OH0_5_E, OH0_25_E = -118.2369, -109.3291, -99.1065, -89.1591
O1_0_E, O0_75_E, O0_5_E, O0_25_E = -99.8942, -95.9180, -90.9981, -85.2925
H1_0_E, H0_75_E, H0_5_E, H0_25_E = -94.6313, -90.7658, -86.85247, -82.9175
sub_H1_0_E, sub_H0_75_E, sub_H0_5_E, sub_H0_25_E = -106.1184, -102.8691, -99.5978, -96.0189

# ============================================================
# BULK + SLAB ENERGIES (PdH, PdO)
# ============================================================
PdH_bulk_E = -34.9140
PdH111_slab_E = -136.11070
n_PdH_fu_slab = 4
AREA_PdH111 = 17.141256

PdO_bulk_E = -40.7576
PdO111_slab_E = -163.0304008
n_PdO_fu_slab = 4
AREA_PdO111 = 20.36401

# ============================================================
# Pd DISSOLUTION LINE (vacancy referenced to bulk Pd)
# ============================================================
Pd_BULK = -20.8849
n_Pd_BULK = 4
Pd111_clean_big = -315.87957
Pd111_1Pdvac_big = -309.95170

E_Pd_bulk_atom = Pd_BULK / n_Pd_BULK
E_vac_form = Pd111_1Pdvac_big + E_Pd_bulk_atom - Pd111_clean_big

E0_Pd_Pd2_SHE = 0.987  # V

# ============================================================
# CHEMICAL POTENTIALS (RHE convention)
# ============================================================

def mu_H(x, y):
    return -0.5 * h2 + (y + x * const) + dsh


def mu_O(x, y):
    return -(h2o - h2) - 2.0 * (y + x * const) + dso


def mu_OH(x, y):
    return -(h2o - 0.5 * h2) - (y + x * const) + dsoh


def mu_H2O(_x, _y):
    return -h2o + dgh2o


# ============================================================
# SURFACE FREE ENERGIES (eV/Å²)
# ============================================================

def e0(_x, _y):
    return 0.0


def e_H2O0_25(x, y):
    return (H2O0_25_E - CleanSurface_E0 + 1 * mu_H2O(x, y)) / AREA_Pd111


def e_H2O0_5(x, y):
    return (H2O0_5_E - CleanSurface_E0 + 2 * mu_H2O(x, y)) / AREA_Pd111


def e_H2O0_75(x, y):
    return (H2O0_75_E - CleanSurface_E0 + 3 * mu_H2O(x, y)) / AREA_Pd111


def e_H2O1_0(x, y):
    return (H2O1_0_E - CleanSurface_E0 + 4 * mu_H2O(x, y)) / AREA_Pd111


def e_OH0_25(x, y):
    return (OH0_25_E - CleanSurface_E0 + 1 * mu_OH(x, y)) / AREA_Pd111


def e_OH0_5(x, y):
    return (OH0_5_E - CleanSurface_E0 + 2 * mu_OH(x, y)) / AREA_Pd111


def e_OH0_75(x, y):
    return (OH0_75_E - CleanSurface_E0 + 3 * mu_OH(x, y)) / AREA_Pd111


def e_OH1_0(x, y):
    return (OH1_0_E - CleanSurface_E0 + 4 * mu_OH(x, y)) / AREA_Pd111


def e_O0_25(x, y):
    return (O0_25_E - CleanSurface_E0 + 1 * mu_O(x, y)) / AREA_Pd111


def e_O0_5(x, y):
    return (O0_5_E - CleanSurface_E0 + 2 * mu_O(x, y)) / AREA_Pd111


def e_O0_75(x, y):
    return (O0_75_E - CleanSurface_E0 + 3 * mu_O(x, y)) / AREA_Pd111


def e_O1_0(x, y):
    return (O1_0_E - CleanSurface_E0 + 4 * mu_O(x, y)) / AREA_Pd111


def e_H0_25(x, y):
    return (H0_25_E - CleanSurface_E0 + 1 * mu_H(x, y)) / AREA_Pd111


def e_H0_5(x, y):
    return (H0_5_E - CleanSurface_E0 + 2 * mu_H(x, y)) / AREA_Pd111


def e_H0_75(x, y):
    return (H0_75_E - CleanSurface_E0 + 3 * mu_H(x, y)) / AREA_Pd111


def e_H1_0(x, y):
    return (H1_0_E - CleanSurface_E0 + 4 * mu_H(x, y)) / AREA_Pd111


def e_sub_H0_25(x, y):
    return (sub_H0_25_E - CleanSurface_E0 + 5 * mu_H(x, y)) / AREA_Pd111


def e_sub_H0_5(x, y):
    return (sub_H0_5_E - CleanSurface_E0 + 6 * mu_H(x, y)) / AREA_Pd111


def e_sub_H0_75(x, y):
    return (sub_H0_75_E - CleanSurface_E0 + 7 * mu_H(x, y)) / AREA_Pd111


def e_sub_H1_0(x, y):
    return (sub_H1_0_E - CleanSurface_E0 + 8 * mu_H(x, y)) / AREA_Pd111


def e_PdH111(_x, _y):
    return (PdH111_slab_E - n_PdH_fu_slab * PdH_bulk_E) / AREA_PdH111


def e_PdO111(_x, _y):
    return (PdO111_slab_E - n_PdO_fu_slab * PdO_bulk_E) / AREA_PdO111


# ============================================================
# DISSOLUTION LINE (boundary)
# ============================================================

def U_dissolution(pH_array):
    return E0_Pd_Pd2_SHE + const * pH_array + 0.5 * E_vac_form


# ============================================================
# MAIN ENERGY MINIMIZATION
# ============================================================

def phase_index(x, y):
    values = [
        e0(x, y),
        e_OH1_0(x, y),
        e_OH0_75(x, y),
        e_OH0_5(x, y),
        e_OH0_25(x, y),
        e_O1_0(x, y),
        e_O0_75(x, y),
        e_O0_5(x, y),
        e_O0_25(x, y),
        e_H1_0(x, y),
        e_H0_75(x, y),
        e_H0_5(x, y),
        e_H0_25(x, y),
        e_PdH111(x, y),
        e_PdO111(x, y),
        e_sub_H0_25(x, y),
        e_sub_H0_5(x, y),
        e_sub_H0_75(x, y),
        e_sub_H1_0(x, y),
    ]
    return int(np.argmin(values))


# ============================================================
# PLOT
# ============================================================

def plot_pourbaix():
    X, Y = np.meshgrid(pH, U)
    Z = np.zeros_like(X, dtype=int)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = phase_index(X[i, j], Y[i, j])

    colors = [
        "#f8f4e3",  # 0  Clean
        "#e7d5a4",  # 1  OH1.0
        "#d6b77c",  # 2  OH0.75
        "#b6935c",  # 3  OH0.5
        "#9a7448",  # 4  OH0.25
        "#c8d7f2",  # 5  O1.0
        "#91b6f2",  # 6  O0.75
        "#5a8ef0",  # 7  O0.5
        "#3842d0",  # 8  O0.25
        "#b7e3c0",  # 9  H1.0
        "#64c47b",  # 10 H0.75
        "#228b22",  # 11 H0.5
        "#145214",  # 12 H0.25
        "#ff7f50",  # 13 PdH(111)
        "#8b0000",  # 14 PdO(111)
        "#f6d6b8",  # 15 1.00 + 0.25 sub
        "#f0c49a",  # 16 1.00 + 0.50 sub
        "#e8ae78",  # 17 1.00 + 0.75 sub
        "#de9758",  # 18 1.00 + 1.00 sub
    ]

    cmap = ListedColormap(colors)
    bounds = np.arange(len(colors) + 1) - 0.5
    norm = BoundaryNorm(bounds, len(colors))

    phase_labels = {
        0: "Clean",
        1: r"OH$_{1.0}$",
        2: r"OH$_{0.75}$",
        3: r"OH$_{0.5}$",
        4: r"OH$_{0.25}$",
        5: r"O$_{1.0}$",
        6: r"O$_{0.75}$",
        7: r"O$_{0.5}$",
        8: r"O$_{0.25}$",
        9: r"H$_{1.0}$",
        10: r"H$_{0.75}$",
        11: r"H$_{0.5}$",
        12: r"H$_{0.25}$",
        13: r"PdH",
        14: r"PdO",
        15: r"sub-H$_{0.25}$",
        16: r"sub-H$_{0.50}$",
        17: r"sub-H$_{0.75}$",
        18: r"sub-H$_{1.0}$",
    }

    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
    mesh = ax.pcolormesh(pH, U, Z, cmap=cmap, norm=norm, shading="auto")

    levels = np.arange(len(colors) + 1) - 0.5
    ax.contour(pH, U, Z, levels=levels, colors="k", linewidths=0.4)

    # Water stability window (RHE)
    ax.plot(pH, 1.65 - pH * const, "--", color="black", lw=1, dashes=(3, 1))

    # Pd dissolution boundary
    ax.plot(pH, U_dissolution(pH), "-", color="red", lw=0.0)

    ax.set_xlabel("pH", fontsize=11)
    ax.set_ylabel(r"$U_\mathrm{SHE}$ (V)", fontsize=11)
    ax.set_xlim(0, 14)
    ax.set_ylim(-2, 2.5)

    cbar = plt.colorbar(mesh, ax=ax, ticks=np.arange(len(colors)))
    cbar.ax.set_yticklabels([phase_labels[i] for i in range(len(colors))])
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig("pd111_pourbaix_map.png", bbox_inches="tight", dpi=300)
    plt.show()


# ============================================================
# PRINT TABLE
# ============================================================

def print_surface_energies():
    pH_values = [0, 7, 14]
    U_values = [-2, -1, 0, 1, 2, 2.4]
    energy_funcs = [
        ("Clean", e0),
        ("OH1.0", e_OH1_0),
        ("OH0.75", e_OH0_75),
        ("OH0.5", e_OH0_5),
        ("OH0.25", e_OH0_25),
        ("O1.0", e_O1_0),
        ("O0.75", e_O0_75),
        ("O0.5", e_O0_5),
        ("O0.25", e_O0_25),
        ("H1.0", e_H1_0),
        ("H0.75", e_H0_75),
        ("H0.5", e_H0_5),
        ("H0.25", e_H0_25),
        ("PdH(111)", e_PdH111),
        ("PdO(111)", e_PdO111),
        ("1.00+sub0.25", e_sub_H0_25),
        ("1.00+sub0.50", e_sub_H0_5),
        ("1.00+sub0.75", e_sub_H0_75),
        ("1.00+sub1.00", e_sub_H1_0),
    ]

    print("\nVacancy formation energy (E_vac_form, eV) =", f"{E_vac_form:.4f}")
    for ph in pH_values:
        Udiss = U_dissolution(ph)
        print(f"Pd dissolution line at pH={ph:4.1f}: U_RHE = {Udiss:.3f} V")

    print("\n================ Surface Free Energies (γ, eV/Å²) ================")
    for ph in pH_values:
        print(f"\nAt pH = {ph:4.1f}")
        print(f"{'Phase':<15}" + "".join([f"{U:>10.2f}" for U in U_values]))
        for label, func in energy_funcs:
            gammas = [float(func(ph, Uv)) for Uv in U_values]
            print(f"{label:<15}" + "".join([f"{g:>10.4f}" for g in gammas]))
    print("=================================================================")


if __name__ == "__main__":
    print_surface_energies()
    plot_pourbaix()
