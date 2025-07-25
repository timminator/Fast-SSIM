import ctypes
import os

import numpy as np

__version__ = '1.0.2'

ssim_dll_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources')
ssim_dll_name = 'ssim.dll' if (os.name == 'nt') else 'libssim.so'


class SharedLibraryLoadError(ImportError):
    """Raised when the shared SSIM/PSNR library could not be loaded."""
    pass


class Loader:
    dll = None

    try:
        dll_path = os.path.join(ssim_dll_path, ssim_dll_name)
        if os.path.exists(dll_path):
            dll = np.ctypeslib.load_library(ssim_dll_name, ssim_dll_path)
        else:
            raise SharedLibraryLoadError(f"Shared library not found at: {dll_path}")
    except Exception as e:
        raise SharedLibraryLoadError(f"Failed to load shared library '{ssim_dll_name}': {e}") from e

    type_dict = {'int': ctypes.c_int, 'float': ctypes.c_float, 'double': ctypes.c_double, 'void': None,
                 'int32': ctypes.c_int32, 'uint32': ctypes.c_uint32, 'int16': ctypes.c_int16, 'uint16': ctypes.c_uint16,
                 'int8': ctypes.c_int8, 'uint8': ctypes.c_uint8, 'byte': ctypes.c_uint8,
                 'char*': ctypes.c_char_p,
                 'float*': np.ctypeslib.ndpointer(dtype='float32', ndim=1, flags='CONTIGUOUS'),
                 'int*': np.ctypeslib.ndpointer(dtype='int32', ndim=1, flags='CONTIGUOUS'),
                 'byte*': np.ctypeslib.ndpointer(dtype='uint8', ndim=1, flags='CONTIGUOUS')}

    @staticmethod
    def get_function(res_type='float', func_name='PSNR_Byte', arg_types=None):
        if arg_types is None:
            arg_types = ['Byte*', 'int', 'int', 'int', 'Byte*']
        func = Loader.dll.__getattr__(func_name)
        func.restype = Loader.type_dict[res_type]
        func.argtypes = [Loader.type_dict[str.lower(x).replace(' ', '')] for x in arg_types]
        return func

    @staticmethod
    def had_member(name='dll'):
        return name in Loader.__dict__


class DLL:
    @staticmethod
    def had_function(name='PSNR_Byte'):
        return name in DLL.__dict__

    if Loader.had_member('dll'):
        # float PSNR_Byte(Byte* pDataX, Byte* pDataY, int step, int width, int height, int maxVal);
        PSNR_Byte = Loader.get_function('float', 'PSNR_Byte', ['Byte*', 'Byte*', 'int', 'int', 'int', 'int'])

        # float PSNR_Float(float* pDataX, float* pDataY, int step, int width, int height, double maxVal);
        PSNR_Float = Loader.get_function('float', 'PSNR_Float', ['float*', 'float*', 'int', 'int', 'int', 'double'])

        # float SSIM_Byte(Byte* pDataX, Byte* pDataY, int step, int width, int height, int win_size, int maxVal);
        SSIM_Byte = Loader.get_function('float', 'SSIM_Byte', ['Byte*', 'Byte*', 'int', 'int', 'int', 'int', 'int'])

        # float SSIM_Float(float* pDataX, float* pDataY, int step, int width, int height, int win_size, double maxVal);
        SSIM_Float = Loader.get_function('float', 'SSIM_Float', ['float*', 'float*', 'int', 'int', 'int', 'int', 'double'])


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
        ValueError: If the input images have unsupported dimensions.
    """
    if x.ndim == 2:
        h, w = x.shape
        c = 1
    elif x.ndim == 3:
        h, w, c = x.shape
    else:
        raise ValueError(f"Unsupported image dimensions: {x.shape}")

    x = x.astype('float32') if (x.dtype == 'float64') else x
    y = y.astype('float32') if (y.dtype == 'float64') else y
    if (DLL.had_function('PSNR_Byte') and x.dtype == 'uint8' and y.dtype == 'uint8'):
        return DLL.PSNR_Byte(x.reshape([-1]), y.reshape([-1]), w * c, w, h, 255 if (data_range is None) else int(data_range))
    if (DLL.had_function('PSNR_Float') and x.dtype == 'float32' and y.dtype == 'float32'):
        return DLL.PSNR_Float(x.reshape([-1]), y.reshape([-1]), w * c, w, h, 255.0 if (data_range is None) else float(data_range))


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
        ValueError: If the input images have unsupported dimensions.
    """
    if x.ndim == 2:
        h, w = x.shape
        c = 1
    elif x.ndim == 3:
        h, w, c = x.shape
    else:
        raise ValueError(f"Unsupported image dimensions: {x.shape}")

    x = x.astype('float32') if (x.dtype == 'float64') else x
    y = y.astype('float32') if (y.dtype == 'float64') else y
    if (DLL.had_function('SSIM_Byte') and x.dtype == 'uint8' and y.dtype == 'uint8'):
        return DLL.SSIM_Byte(x.reshape([-1]), y.reshape([-1]), w * c, w, h, win_size, 255 if (data_range is None) else int(data_range))
    if (DLL.had_function('SSIM_Float') and x.dtype == 'float32' and y.dtype == 'float32'):
        return DLL.SSIM_Float(x.reshape([-1]), y.reshape([-1]), w * c, w, h, win_size, 255.0 if (data_range is None) else float(data_range))
