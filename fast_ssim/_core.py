from __future__ import annotations

import ctypes
import os

import numpy as np

__version__ = '1.3.1'

ssim_dll_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources')
ssim_dll_name = 'ssim.dll' if (os.name == 'nt') else 'libssim.so'

Float1D = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')
Uint81D = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags='C_CONTIGUOUS')
c_int = ctypes.c_int
c_float = ctypes.c_float
c_double = ctypes.c_double


class SharedLibraryLoadError(ImportError):
    """Raised when the shared SSIM/PSNR library could not be loaded."""
    pass


class Loader:
    dll = None
    cpu_status = -1

    @classmethod
    def load(cls):
        if cls.dll is not None:
            return

        dll_path = os.path.join(ssim_dll_path, ssim_dll_name)
        if not os.path.exists(dll_path):
            raise SharedLibraryLoadError(f"Shared library not found at: {dll_path}")

        try:
            cls.dll = np.ctypeslib.load_library(ssim_dll_name, ssim_dll_path)

            check_func = cls.dll.CheckCpuSupport
            check_func.restype = c_int
            check_func.argtypes = []

            cls.cpu_status = check_func()
        except Exception as e:
            raise SharedLibraryLoadError(f"Failed to load shared library '{ssim_dll_name}': {e}") from e

    @classmethod
    def bind_func(cls, name, restype, argtypes):
        if cls.dll is None or not hasattr(cls.dll, name):
            return None
        func = getattr(cls.dll, name)
        func.restype = restype
        func.argtypes = argtypes
        return func


class DLL:
    _initialized = False
    PSNR_DISPATCH = {}
    SSIM_DISPATCH = {}
    SSIM_SLOW_DISPATCH = {}

    @classmethod
    def initialize(cls):
        if cls._initialized:
            return
        Loader.load()

        # float PSNR_Byte(Byte* pDataX, Byte* pDataY, int step, int width, int height, int maxVal);
        psnr_byte = Loader.bind_func('PSNR_Byte', c_float, [Uint81D, Uint81D, c_int, c_int, c_int, c_int])
        # float PSNR_Float(float* pDataX, float* pDataY, int step, int width, int height, double maxVal);
        psnr_float = Loader.bind_func('PSNR_Float', c_float, [Float1D, Float1D, c_int, c_int, c_int, c_double])
        # float SSIM_Byte(Byte* pDataX, Byte* pDataY, int step, int width, int height, int win_size, int maxVal);
        ssim_byte = Loader.bind_func('SSIM_Byte', c_float, [Uint81D, Uint81D, c_int, c_int, c_int, c_int, c_int])
        # float SSIM_Float(float* pDataX, float* pDataY, int step, int width, int height, int win_size, double maxVal);
        ssim_float = Loader.bind_func('SSIM_Float', c_float, [Float1D, Float1D, c_int, c_int, c_int, c_int, c_double])
        # float SSIM_Byte_Slow(Byte* pDataX, Byte* pDataY, int widthBytes, int width, int height, int win_size);
        ssim_byte_slow = Loader.bind_func('SSIM_Byte_Slow', c_float, [Uint81D, Uint81D, c_int, c_int, c_int, c_int])

        cls.PSNR_DISPATCH = {
            np.uint8: psnr_byte,
            np.float32: psnr_float
        }

        cls.SSIM_DISPATCH = {
            np.uint8: ssim_byte,
            np.float32: ssim_float
        }

        cls.SSIM_SLOW_DISPATCH = {
            np.uint8: ssim_byte_slow
        }

        cls._initialized = True


def get_cpu_status() -> int:
    """
    Retrieves the hardware acceleration status detected by the C++ backend.

    Returns:
        int: A status code representing the CPU support level:
            * 0: AVX2 + FMA fully supported and active.
            * 1: Missing OS XSAVE or AVX (Falling back to SSE).
            * 2: Missing AVX2 (Falling back to SSE).
            * 3: Missing FMA (Falling back to SSE).
            * -1: DLL failed to load or status was not checked.
    """
    if not DLL._initialized:
        DLL.initialize()
    return Loader.cpu_status


def _prepare_images(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """Shared helper to validate shapes, dtypes, and enforce memory contiguity."""
    if x.shape != y.shape:
        raise ValueError(f"Input images must have the same shape. Got {x.shape} and {y.shape}")

    if x.ndim == 2:
        h, w = x.shape
        c = 1
    elif x.ndim == 3:
        h, w, c = x.shape
    else:
        raise ValueError(f"Unsupported image dimensions: {x.shape}")

    if x.dtype == np.float64:
        x = x.astype(np.float32)
    if y.dtype == np.float64:
        y = y.astype(np.float32)

    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)

    return x, y, w, h, c


def psnr(x: np.ndarray, y: np.ndarray, data_range: int | float | None = None) -> float:
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        x (np.ndarray): The first image (e.g., original image). Can be 2D (grayscale) or 3D (color).
                        Supported dtypes are 'uint8' and 'float32'.
        y (np.ndarray): The second image (e.g., reconstructed or noisy image). Must have the same
                        dimensions and dtype as `x`.
        data_range (int | float | None, optional): The dynamic range of the pixel values (e.g., 255 for uint8
                                                    images, 1.0 for float images). If None, it defaults to 255
                                                    for 'uint8' and 255.0 for 'float32'.

    Returns:
        float: The PSNR value.

    Raises:
        ValueError: If the input images have unsupported dimensions or differing shapes.
        TypeError: If an unsupported dtype is provided.
    """
    x, y, w, h, c = _prepare_images(x, y)

    if not DLL._initialized:
        DLL.initialize()

    func = DLL.PSNR_DISPATCH.get(x.dtype.type)
    if func is None:
        raise TypeError(f"Unsupported dtype: {x.dtype}")

    maxval = 255 if x.dtype.type is np.uint8 else 255.0
    if data_range is not None:
        maxval = type(maxval)(data_range)

    return func(x.ravel(), y.ravel(), w * c, w, h, maxval)


def ssim(x: np.ndarray, y: np.ndarray, data_range: int | float | None = None, win_size: int = 7) -> float:
    """
    Calculates the Structural Similarity Index (SSIM) between two images.

    Args:
        x (np.ndarray): The first image (e.g., original image). Can be 2D (grayscale) or 3D (color).
                        Supported dtypes are 'uint8' and 'float32'.
        y (np.ndarray): The second image (e.g., reconstructed or noisy image). Must have the same
                        dimensions and dtype as `x`.
        data_range (int | float | None, optional): The dynamic range of the pixel values (e.g., 255 for uint8
                                                    images, 1.0 for float images). If None, it defaults to 255
                                                    for 'uint8' and 255.0 for 'float32'.
        win_size (int, optional): The size of the sliding window for SSIM calculation. Defaults to 7.

    Returns:
        float: The SSIM value.

    Raises:
        ValueError: If the input images have unsupported dimensions or differing shapes.
        TypeError: If an unsupported dtype is provided.
    """
    x, y, w, h, c = _prepare_images(x, y)

    if not DLL._initialized:
        DLL.initialize()

    func = DLL.SSIM_DISPATCH.get(x.dtype.type)
    if func is None:
        raise TypeError(f"Unsupported dtype: {x.dtype}")

    maxval = 255 if x.dtype.type is np.uint8 else 255.0
    if data_range is not None:
        maxval = type(maxval)(data_range)

    return func(x.ravel(), y.ravel(), w * c, w, h, win_size, maxval)


def ssim_slow(x: np.ndarray, y: np.ndarray, win_size: int = 7) -> float:
    """
    Calculates the Structural Similarity Index (SSIM) using the unoptimized, scalar C++ fallback.

    Args:
        x (np.ndarray): The first image (e.g., original image). Can be 2D (grayscale) or 3D (color).
                        Supported dtype is 'uint8' only.
        y (np.ndarray): The second image (e.g., reconstructed or noisy image). Must have the same
                        dimensions and dtype as `x`.
        win_size (int, optional): The size of the sliding window for SSIM calculation. Defaults to 7.

    Returns:
        float: The SSIM value.

    Raises:
        ValueError: If the input images have unsupported dimensions or differing shapes.
        NotImplementedError: If the input images are not of type 'uint8'.
    """
    x, y, w, h, c = _prepare_images(x, y)

    if not DLL._initialized:
        DLL.initialize()

    func = DLL.SSIM_SLOW_DISPATCH.get(x.dtype.type)
    if func is None:
        raise NotImplementedError("ssim_slow is only implemented for uint8 data types.")

    return func(x.ravel(), y.ravel(), w * c, w, h, win_size)
