Copyright (C) 2025-present - Perpetual licence

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

MAIN PAPERS (USE THEM FOR REFERENCE):

- Fernandez-Leon, J.A., Uysal, A.K. & Ji, D. (2022) Place cells dynamically refine grid cell activities to reduce error accumulation during path integration in a continuous attractor model. Sci Rep 12, 21443. https://doi.org/10.1038/s41598-022-25863-2
- Fernandez-Leon, J.A., Sarramone L. (2023) Estimación robusta de posición durante navegación espacial mediante la coordinación de neuronas grilla. Reunión de trabajos en procesamiento de la información y control - RPIC

NOTES:

The neuron grid model proposed by Guanella (2005) is used. Everything related to this model is contained in the script "grid.py". At the beginning of the program, an instance of the network is created under the name "grid". At each time step, the activity of the neurons is updated using the "update" method, which takes the velocity in X and Y as parameters. The velocity is encapsulated as a complex number, where the X value forms the real part and Y the imaginary part.

Since this version has no velocity or acceleration sensors, velocity is calculated as the difference between the current position and the position from the previous step. However, any other method of obtaining the robot's velocity is valid, as long as it is decomposed into (X, Y) vectors.

The velocity values ​​accepted by the network are in the range [-0.1 and 0.1].

For position estimation, the script "estimator.py" is used, from which an "Estimator" object is created. At each step, the `new_estimation` method is executed, taking as parameters the current state of the grid neurons and the state of the grid neurons in a previous step. This method returns three values:
- `new_X`: Estimated position in X
- `new_Y`: Estimated position in Y
- `info`: Information about the state of the grid neuron network at each instant in time (ignored in this version)

METHODS:

Minimum code version requirements :
- Webots R2023b
- Python 3.7.0 
- Numpy 1.21.6

REFERENCES

-- Guanella, A., Kiper, D., & Verschure, P. (2007). A model of grid cells based on a twisted torus topology. International journal of neural systems, 17(04), 231-240.
