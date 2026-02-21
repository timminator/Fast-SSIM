import platform
import time

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage
from skimage.metrics import structural_similarity as ssim_skimage

import fast_ssim


def get_cpu_name():
    try:
        if platform.system() == "Windows":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return name.strip()
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
        elif platform.system() == "Darwin":
            import subprocess
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"


def run_benchmark(name, width, height, iterations=10):
    print(f"Benchmarking {name} ({iterations} iterations)... please wait.")

    img1 = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

    # --- SSIM BENCHMARK ---
    start_sk = time.perf_counter()
    for _ in range(iterations):
        ssim_skimage(img1, img2, data_range=255, channel_axis=2, win_size=7)
    total_skimage_ssim = time.perf_counter() - start_sk

    start_fast = time.perf_counter()
    for _ in range(iterations):
        fast_ssim.ssim(img1, img2, data_range=255)
    total_fast_ssim = time.perf_counter() - start_fast

    avg_sk_ssim = total_skimage_ssim / iterations
    avg_fast_ssim = total_fast_ssim / iterations
    speedup_ssim = avg_sk_ssim / avg_fast_ssim

    # --- PSNR BENCHMARK ---
    start_sk = time.perf_counter()
    for _ in range(iterations):
        psnr_skimage(img1, img2, data_range=255)
    total_skimage_psnr = time.perf_counter() - start_sk

    start_fast = time.perf_counter()
    for _ in range(iterations):
        fast_ssim.psnr(img1, img2, data_range=255)
    total_fast_psnr = time.perf_counter() - start_fast

    avg_sk_psnr = total_skimage_psnr / iterations
    avg_fast_psnr = total_fast_psnr / iterations
    speedup_psnr = avg_sk_psnr / avg_fast_psnr

    return [
        (name, "SSIM", avg_sk_ssim, avg_fast_ssim, speedup_ssim),
        (name, "PSNR", avg_sk_psnr, avg_fast_psnr, speedup_psnr)
    ]


if __name__ == "__main__":
    print("Starting Benchmark Suite...\n")

    results = []
    results.extend(run_benchmark(name="1080p", width=1920, height=1080, iterations=10))
    results.extend(run_benchmark(name="4K", width=3840, height=2160, iterations=10))

    print("\nAll benchmarks finished!\n")

    # Print the Markdown Table with CPU Info
    cpu_name = get_cpu_name()
    print(f"**Hardware Tested:** `{cpu_name}`\n")
    print("| Resolution | Metric | scikit-image (Avg) | Fast-SSIM (Avg) | Speedup |")
    print("|------------|--------|--------------------|-----------------|---------|")
    for row in results:
        res, metric, sk_time, fast_time, speedup = row
        print(f"| {res:<10} | {metric:<6} | {sk_time:.4f}s            | {fast_time:.4f}s         | {speedup:.1f}x  |")
