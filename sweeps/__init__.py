"""Parameter-sweep harness: calibration, diagnostics, and the Optuna objective.

Not part of the model and not an experiment. These drive the network directly
with R = I, which characterises the integrator with the plane filter out of the
loop -- see the scope note in sweep_minimal.py.
"""
