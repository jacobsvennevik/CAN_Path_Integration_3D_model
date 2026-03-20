Document describing what I am currently working on with the implementation.

- Understand the code and the model.
- Need to understand where I should inject the plane switching
- Need to understand how to decouple the simulation model DeepMind and the grid cell model.

- Need to implement the testing model - using gang and yu purely matemathical appraoch


CHECK:
- There isnt really any tracking of gaps or outliers something we have to check then
- CAN based burak and fiete CANs use continuous-valued rate units or Poisson spiking. the MeanShift bandwidth of 0.25 was for Gong & Yu's RNN spike density. Need to verify that this bandwidth is appropriate for your spike density, or the clustering will over- or under-segment fields. In the notebooks