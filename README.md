# De novo design of EGFR T790M inhibitors: GraphVAE vs pocket-conditioned TGVAE

MSc Applied Artificial Intelligence dissertation, WMG, University of Warwick (2026).
Author: Amelia Burton · Supervisor: Omid Chatrabgoun

A target-directed generative deep learning pipeline for proposing novel inhibitors of
mutant EGFR in non small cell lung cancer (NSCLC). The project compares two generative
strategies on identical data, screening and docking protocols:

| | Baseline | Proposed |
|---|---|---|
| Model | GraphVAE + LatentGAN | Pocket-conditioned transformer graph VAE (TGVAE) |
| Encoder | 3x GINEConv + global mean pool | 4x TransformerConv (graph attention) with residuals, LayerNorm, JK projection |
| Decoder | One-shot MLP (nodes, adjacency, bonds) | Autoregressive SMILES transformer |
| Latent | 64-d | 64-d, expanded to a memory and concatenated with a 128-d pocket embedding |

Generated molecules are filtered for EGFR viability and drug likeness, ranked by a
GNN potency regressor, and the leading candidates are docked against EGFR (PDB 6LUD,
L858R/T790M/C797S).


## Repository structure

Run the notebooks in numerical order. Each depends on outputs from the previous ones.

## Data

All data are public secondary data; no human participants or personal data are involved.

- EGFR bioactivity records from ChEMBL, BindingDB and PDB, standardised to pIC50
  (retained range 3.0 to 12.0), deduplicated on canonical SMILES.
- 19,846 molecular graphs retained from 20,861 initial records.
- Pocket embedding from EGFR structures 1M17, 2ITY, 3IKA, 3W2S and 6LUD
  (binding-site atoms within 6 Å of the co-crystallised ligand).


## Installation

Python <version>. Core dependencies: PyTorch, PyTorch Geometric, RDKit, scikit-learn,
SciPy, Biopython, NumPy, pandas, matplotlib, seaborn.


A GPU is recommended for training the TGVAE.

## Reproducibility

- Random seeds are fixed throughout, including conformer generation (ETKDG).
- The GNN is evaluated on a scaffold-disjoint (Bemis-Murcko) split. This gives a
  deliberately stringent, out-of-distribution estimate, so scores are lower than with a
  random split.
- Both generators use the same data, screening, filters and docking protocol.
- Docking was run through the SwissDock 2024 web service (AutoDock Vina, box centred at
  (-57, -6, -24), 20 Å cube, exhaustiveness 15), so that step is not scripted in the
  notebooks. 

## Limitations

- Bioactivity labels are aggregated across assays and carry inter-assay noise.
- Predicted pIC50 values for highly novel molecules are extrapolations (nearest-neighbour
  Tanimoto < 0.4). Use them as rankings, not absolute potencies.
- TGVAE raw validity (0.61) is modest, though controllable through sampling temperature.
- Docking sets are small (6 and 7 molecules) and rigid-receptor.
- The pocket embedding is a distance-signature proxy, not explicit 3D coordinates.


