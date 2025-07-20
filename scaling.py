# %%
try:
    import IPython

    IPython.get_ipython().run_line_magic("load_ext", "autoreload")
    IPython.get_ipython().run_line_magic("autoreload", "2")
except:
    pass

# %%
from guti.data_utils import list_svd_variants
import matplotlib.pyplot as plt
import numpy as np

modality_name = "fnirs_analytical_cw"

# %%
subdir = "grid_sweep"
param_key = "grid_resolution_mm"

variants = list_svd_variants(modality_name, subdir=subdir)
for k, v in variants.items():
    print(f"  {k}: {v['params']}")

# %%
# Set the parameter key you want to analyze

plt.figure(figsize=(10, 6))
sorted_variants = sorted(
    variants.items(), key=lambda x: getattr(x[1]["params"], param_key)
)
for k, v in sorted_variants:
    params = v["params"]
    s = v["s"]
    param_value = getattr(params, param_key)

    plt.plot(
        np.arange(1, len(s) + 1),
        s / s[0],
        label=f"{param_key}={param_value}",
    )
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Singular Value Index")
plt.ylabel("Singular Value")
plt.title(f"{param_key} Scaling - {modality_name}")
plt.ylim(1e-5, 1)
plt.show()

# %%
