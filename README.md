# Legendre Membership Inference

Source package for the Legendre membership-inference pipeline.

## Usage

```bash
pip install -r requirements.txt
python run_pipeline.py --artifact-root /path/to/assets --output-root /path/to/results --devices 0,1
```

See `config/default.json` for the default configuration.

## Structure

- `run_models.py` - model training and output caching
- `run_attack.py` - attack readout fitting and scoring
- `run_results.py` - metric tables, ROC figures, validation
- `legendre_mia/attacks/` - feature construction and readout
- `legendre_mia/models/` - ResNet-18 and tabular architectures
- `legendre_mia/workflows/` - training and orchestration
