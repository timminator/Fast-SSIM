<p align="center">
  <h1 align="center">Fast-SSIM</h1>
  <p align="center">
    Fast, multi-threaded SSIM and PSNR calculations!
    <br />
  </p>
</p>

<br>

## ℹ About

This fork wraps the Fast-SSIM project in an easy-to-use package that can be readily installed via PyPI.<br>
To further accelerate processing, this version enables multithreading and leverages AVX2/FMA hardware instructions. For older CPUs it falls back to a SSE implementation.<br>
This version achieves up to a 360x speedup for SSIM and 150x for PSNR on modern multi-core processors (tested on an AMD Ryzen 5 5600X 6-Core Processor) compared to standard implementations.

## Requirements

- Python 3.9 or higher

## How to Install

```
pip install Fast-SSIM
```

## Usage

The functionalities are explained in the following code snippet that is also provided in this repo:

```python
from skimage.io import imread

import fast_ssim

img1_path = r"test_images/0.jpg"
img2_path = r"test_images/1.jpg"

# Load the images into NumPy arrays
img1 = imread(img1_path)
img2 = imread(img2_path)

ssim_score = fast_ssim.ssim(
    img1, img2,
    data_range=255
)
psnr_score = fast_ssim.psnr(
    img1, img2,
    data_range=255
)

print(f"SSIM Score: {ssim_score}")
print(f"PSNR Score: {psnr_score}")
```

Output:
```
SSIM Score: 0.9886590838432312
PSNR Score: 31.11587142944336
```

## Diagnostics & Testing

You can query the C++ backend about what hardware acceleration is being used:

```python
import fast_ssim

status = fast_ssim.get_cpu_status()
if status == 0:
    print("Using AVX2 + FMA")
else:
    print(f"Running in compatibility mode (SSE Fallback). Status Code: {status}")
```

If you have a modern AVX2-capable CPU but want to test the slower SSE fallback path (for benchmarking or debugging), you can force the library to bypass AVX2 entirely.<br>
Just set the `SSIM_FORCE_SSE` environment variable before importing the library:

```python
import os
os.environ["SSIM_FORCE_SSE"] = "1"

import fast_ssim
print(fast_ssim.get_cpu_status()) # Will now output: 2 (Missing AVX2 / Forced SSE)
```

## Performance Comparison

Due to multithreading, your exact speedup will scale dynamically with your CPU's core count. The benchmarks below were recorded on an AMD Ryzen 5 5600X 6-Core Processor:

**Hardware Tested:** `AMD Ryzen 5 5600X 6-Core Processor`

| Resolution | Metric | scikit-image (Avg) | Fast-SSIM (Avg) | Speedup |
|------------|--------|--------------------|-----------------|---------|
| 1080p      | SSIM   | 0.5804s            | 0.0023s         | 251.0x  |
| 1080p      | PSNR   | 0.0426s            | 0.0002s         | 180.4x  |
| 4K         | SSIM   | 2.6128s            | 0.0071s         | 367.5x  |
| 4K         | PSNR   | 0.1782s            | 0.0012s         | 154.4x  |

## Notes

- The SSIM calculation uses sample covariance (and sample variance) for its statistics. This aligns with the default behavior of `scikit-image`'s `structural_similarity` function, where `use_sample_covariance` is True by default.

- This implementation does not offer an option for applying a Gaussian filter to the images or local windows prior to the SSIM/PSNR calculation.

## Original author

Chen Yu / [@Chen Yu](https://github.com/chinue)
