"""The model: the attractor network, the plane estimator, and the coupler.

Three parts, in dependency order:

  network/            the T3 lattice -- kernel, connectivity, the six
                      velocity-driven CANs, the FFT recurrence, bump decoding
  plane_estimation    the recursive Bingham filter on S2
  path_integration    the only module importing both: filter -> n_hat -> rotate
                      velocity -> drive the network -> decode

Nothing here imports `config`. Parameters arrive as arguments, which is what
lets the sweep build a network without any experiment scaffolding. Keep it that
way; see ARCHITECTURE.md.
"""
