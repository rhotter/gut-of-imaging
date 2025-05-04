# Grand unified theory of imaging

## Using the Data Utilities

To use the data saving/loading functions in your notebooks or scripts:

1. First install the package in editable mode:
```bash
pip install -e .
```

2. Import and use the functions:
```python
import data_utils

# Save SVD results
data_utils.save_svd(singular_values, modality_name='fnirs_cw', params=optional_params)

# Load SVD results
singular_values, params = data_utils.load_svd(modality_name='fnirs_cw')
```

The data will be saved in a `results` directory at the root of the project. The directory structure will be:
```
gut-of-imaging/
├── results/
│   └── fnirs_cw_svd_spectrum.npz
├── data_utils/
├── fnirs/
└── ...
```

### Available Functions

- `save_svd(s, modality_name, params=None)`: Save singular values and optional parameters
  - `s`: numpy array of singular values
  - `modality_name`: str, e.g., 'fnirs_cw' or 'eeg'
  - `params`: optional dict of parameters to save alongside singular values

- `load_svd(modality_name)`: Load saved singular values and parameters
  - Returns: `(singular_values, parameters)` tuple
