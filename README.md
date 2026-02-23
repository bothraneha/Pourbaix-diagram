# Pourbaix_Diagram

This repository contains a cleaned, documented script for generating combined surface Pourbaix diagrams for IrO2 facets (101, 110, 111). The script computes surface free energies (Δγ) as a function of SHE potential and plots all coverages together.

## What the script does

1. **Defines thermodynamic constants** for gas-phase H2 and H2O and for adsorbate corrections (O, OH, OOH) using VASP-PBE values.
2. **Computes Gibbs corrections** for gas molecules (ΔG) and adsorbates (ΔG_O, ΔG_OH, ΔG_OOH, ΔG_H).
3. **Builds surface free energies** for the three facets using the DFT total energies and coverage counts.
4. **Finds minimum Δγ values** at selected potentials to help identify stable coverages.
5. **Plots Δγ vs. SHE** for all surfaces and saves the figure as PDF and PNG.

## Files

pourbaix_diagram.py

## Outputs

Running the script generates:
corrected_H2O_included.pdf
corrected_H2O_included.png

## How to run

1. Ensure you have Python 3 and the dependencies installed:
numpy, matplotlib

2. Run:
python3 pourbaix_diagram.py
