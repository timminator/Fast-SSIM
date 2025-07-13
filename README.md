<p align="center">
  <h1 align="center">Fast-SSIM</h1>
  <p align="center">
    Speed up your SSIM and PSNR calculations!
    <br />
  </p>
</p>

<br>

## ℹ About

This fork wraps the Fast-SSIM project in an easy to use package that can be readily installed via PyPI.
The Fast-SSIM package can accelerate your SSIM and PSNR calculations by up to 30x and 10x respectively.

## Requirements

- Python 3.9 or higher

## How to Install

```
pip install Fast-SSIM
```

## Usage

The functionalities are explained in the following code snippet that is also provided in this repo:

```python
import fast_ssim
from skimage.io import imread

img1_path = r"test_images\0.jpg"
img2_path = r"test_images\1.jpg"

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

## Original author

Chen Yu / [@Chen Yu](https://github.com/chinue)
