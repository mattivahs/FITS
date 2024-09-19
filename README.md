# Forward Invariance in Trajectory Spaces (FITS)
This repository contains the code of the paper *Forward Invariance in Trajectory Spaces for Safety-critical Control*.
We use two simulation environments, a custom navigation in a cluttered environment and the Quadcopter 2D environment in [safe-control-gym]([https://gym.openai.com](https://github.com/utiasDSL/safe-control-gym)).

```bibtex
@article{vahs2024forward,
  title={Forward Invariance in Trajectory Spaces for Safety-critical Control},
  author={Vahs, Matti and Muchacho, Rafael I Cabral and Pokorny, Florian T and Tumova, Jana},
  journal={arXiv preprint arXiv:2407.12624},
  year={2024}
}
```

## Run Experiments

Comparison with baseline approaches, namely Control Barrier Functions (CBFs) and Nonlinear Model Predictive Control (NMPC).

### 2D Quadrotor Geofencing Task

```bash
cd ./FITSExperiment/   # Navigate to the examples folder
./fits_experiment.sh
```
