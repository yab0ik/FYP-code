# FPGA-Based Hardware Acceleration of Options Pricing

This project validates a Python Black-Scholes reference implementation as the first stage
of an FPGA options-pricing pipeline. Later stages will train a neural-network approximation,
quantise it to fixed point, and deploy the inference pipeline to an FPGA for benchmarking.
Only the project skeleton and objective B1 are implemented at this stage.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

To install the test dependencies directly instead:

```powershell
python -m pip install -r requirements.txt
```

Run the tests with:

```powershell
pytest
```

## Baseline neural network (objective B2)

Train the baseline call/K MLP with the settings in `configs/default.yaml`:

```powershell
python src/optfpga/nn/train.py
python src/optfpga/nn/evaluate.py results/b2_baseline
```

Run the architecture/activation sweep with:

```powershell
python scripts/sweep_b2.py
```

The optional `py_vollib` comparison test can be enabled by installing the development
extra:

```powershell
python -m pip install -e ".[dev]"
```
