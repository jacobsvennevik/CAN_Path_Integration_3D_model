# CAN_Path_Integration_Plane_estimation_3D_model


SUMMARY:
This project investigates how grid cells active during three-dimensional navigation can be locally ordered yet globally unordered. We propose that this loss of global order arises because path integration is performed relative to different noisy estimates of reference planes rather than a single global lattice. To test this hypothesis, we will extend the grid cell continuous attractor network (CAN) model of Fernández-León et al. (2022) into a plane-based “2.5D” model following Gong and Yu (2021). In this extension, velocity inputs are integrated relative to a reference plane, while place cells provide anchoring to limit drift accumulation. Changes in plane estimates are expected to keep the local grid structure while gradually disrupting the global grid organization. Experiments will be conducted using a simulated drone agent in three environments: a 2D baseline arena, a fully volumetric 3D environment, and a plane switching manifold. 


The neuron grid model proposed by Burak and Fiete (2008) is used. 

For information about the implementasion of the project, both what I am currently working on and the implemented changes. Go to [docs/implementation/]
The folder [spatial_nav/] contain the models, the plane estimation and path integration.


REFERENCES:
- Fernandez-Leon, J.A., Uysal, A.K. & Ji, D. (2022) Place cells dynamically refine grid cell activities to reduce error accumulation during path integration in a continuous attractor model. Sci Rep 12, 21443. https://doi.org/10.1038/s41598-022-25863-2
- Burak, Y., & Fiete, I. R. (2009). Accurate Path Integration in Continuous Attractor Network Models of Grid Cells. PLoS Computational Biology, 5(2), e1000291. https://doi.org/10.1371/journal.pcbi.1000291
- Gong, Z., & Yu, F. (2021). A Plane-Dependent Model of 3D Grid Cells for Representing Both 2D and 3D Spaces Under Various Navigation Modes. Frontiers in Computational Neuroscience, 15. https://doi.org/10.3389/fncom.2021.739515



## Acknowledgements
The class CAN-model is based on the following implementasion, and was used a starting point for our extension https://github.com/changmin-yu/grid-cell-models-python/blob/main/burak_fiete_2009.py
The initial controller is from Fernandez-Leon and forked from his repository.
The plane estimation code is heaoliy inspired from Gong And Yu´s code and repository:
https://github.com/gongziyida/GridCells3D
