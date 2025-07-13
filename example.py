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
