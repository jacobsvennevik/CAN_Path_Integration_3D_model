This is where the analysis of Gong and Yu (2021) happens from theire own repository.

[preclustering_analysis.ipynb:] analyses before clustering the spikes. Its purpose is to characterize the statisitcal properties of grid structure after three metrics.
1. Spatail information and Sparsity index - this is to see if we have grid fields
2. Strcuture scores, FCC, HCP, COL and random - this is to check for the global lattics structure
3. MRA - answers if the repeating pattern survive at long distances, or only nearby

- Spatial information and sparsity index
- Structure scores
- Modified radial autocorrelation

[clustering.ipynb:] clustering analysis - This clustering step is the methodological bridge between raw spikes and the structural analysis it's prerequisite for structure scores and IFD.

Inter-field distance distribution
[3D_onto_2D.ipynb:] mode switching (Figure 5 in the paper, Gong and Yu, 2021). Simulates the trajectories and mode switching of the paper. Simulates trajectories in a half-flat, half-tilted arena

Run:
1. preclustering
2. clustering
3. 3d_onto_2d
