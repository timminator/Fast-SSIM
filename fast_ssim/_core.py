from __future__ import annotations

import array
import ctypes
import os
from typing import Any

__version__ = '1.4.0'

ssim_dll_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources')
ssim_dll_name = 'ssim.dll' if (os.name == 'nt') else 'libssim.so'

c_int = ctypes.c_int
c_float = ctypes.c_float
c_double = ctypes.c_double
c_ubyte_p = ctypes.POINTER(ctypes.c_ubyte)
c_float_p = ctypes.POINTER(ctypes.c_float)


class SharedLibraryLoadError(ImportError):
    """Raised when the shared SSIM/PSNR library could not be loaded."""
    pass


def _get_ptr(data: Any, ctype: Any) -> tuple[Any, int, Any]:
    """
    Return (ctypes_pointer, item_count, keepalive). `keepalive` MUST
    stay referenced for as long as the pointer is used.
    """
    if type(data).__module__ == 'numpy' and hasattr(data, 'ctypes'):
        if hasattr(data, 'flags') and not data.flags.c_contiguous:
            raise ValueError(
                "Input numpy array is not C-contiguous. "
                "Please call .copy(order='C') on your array before passing it."
            )
        if str(getattr(data, 'dtype', '')) == 'float64':
            raise TypeError(
                "float64 numpy arrays are not supported. "
                "Please call .astype('float32') on your array before passing it."
            )

        return ctypes.cast(data.ctypes.data, ctypes.POINTER(ctype)), data.size, data

    mv = memoryview(data)
    if mv.itemsize != ctypes.sizeof(ctype):
        raise TypeError(
            f'buffer item size {mv.itemsize} does not match expected '
            f'{ctypes.sizeof(ctype)} bytes for {ctype.__name__} data'
        )
    n = len(mv)
    if mv.readonly:
        arr = (ctype * n).from_buffer_copy(mv)
    else:
        arr = (ctype * n).from_buffer(mv)

    return ctypes.cast(arr, ctypes.POINTER(ctype)), n, arr


def _as_float32_array(data: Any) -> Any:
    """array.array-based float64 to float32 caster."""
    if isinstance(data, array.array):
        if data.typecode == 'f':
            return data
        if data.typecode == 'd':
            return array.array('f', data)
        raise TypeError(f"expected array.array('f' or 'd', ...), got typecode {data.typecode!r}")
    return data


def _shape_of(data: Any, width: int | None, height: int | None, channels: int | None) -> tuple[int, int, int]:
    """Resolve (w, h, c): uses duck typing for NumPy arrays, else requires explicit args."""
    shape = getattr(data, 'shape', None)
    if shape is not None:
        if len(shape) == 2:
            return shape[1], shape[0], 1
        elif len(shape) == 3:
            return shape[1], shape[0], shape[2]
        else:
            raise ValueError(f'Unsupported image dimensions: {shape}')

    if width is None or height is None:
        raise ValueError(
            'width and height must be given explicitly for non-numpy '
            'inputs (bytes/bytearray/memoryview carry no shape info)'
        )
    return width, height, (channels if channels is not None else 1)


class Loader:
    dll: ctypes.CDLL | None = None
    cpu_status: int = -1

    @classmethod
    def load(cls) -> None:
        if cls.dll is not None:
            return

        dll_path = os.path.join(ssim_dll_path, ssim_dll_name)
        if not os.path.exists(dll_path):
            raise SharedLibraryLoadError(f"Shared library not found at: {dll_path}")

        try:
            cls.dll = ctypes.CDLL(dll_path)

            check_func = cls.dll.CheckCpuSupport
            check_func.restype = c_int
            check_func.argtypes = []

            cls.cpu_status = check_func()
        except Exception as e:
            raise SharedLibraryLoadError(f"Failed to load shared library '{ssim_dll_name}': {e}") from e

    @classmethod
    def bind_func(cls, name: str, restype: Any, argtypes: list[Any]) -> Any:
        if cls.dll is None or not hasattr(cls.dll, name):
            return None
        func = getattr(cls.dll, name)
        func.restype = restype
        func.argtypes = argtypes
        return func


class DLL:
    _initialized: bool = False
    PSNR_BYTE: Any = None
    PSNR_FLOAT: Any = None
    SSIM_BYTE: Any = None
    SSIM_FLOAT: Any = None
    SSIM_BYTE_SLOW: Any = None

    @classmethod
    def initialize(cls) -> None:
        if cls._initialized:
            return
        Loader.load()

        # float PSNR_Byte(Byte* pDataX, Byte* pDataY, int step, int width, int height, int maxVal);
        cls.PSNR_BYTE = Loader.bind_func('PSNR_Byte', c_float, [c_ubyte_p, c_ubyte_p, c_int, c_int, c_int, c_int])
        # float PSNR_Float(float* pDataX, float* pDataY, int step, int width, int height, double maxVal);
        cls.PSNR_FLOAT = Loader.bind_func('PSNR_Float', c_float, [c_float_p, c_float_p, c_int, c_int, c_int, c_double])
        # float SSIM_Byte(Byte* pDataX, Byte* pDataY, int step, int width, int height, int win_size, int maxVal);
        cls.SSIM_BYTE = Loader.bind_func('SSIM_Byte', c_float, [c_ubyte_p, c_ubyte_p, c_int, c_int, c_int, c_int, c_int])
        # float SSIM_Float(float* pDataX, float* pDataY, int step, int width, int height, int win_size, double maxVal);
        cls.SSIM_FLOAT = Loader.bind_func('SSIM_Float', c_float, [c_float_p, c_float_p, c_int, c_int, c_int, c_int, c_double])
        # float SSIM_Byte_Slow(Byte* pDataX, Byte* pDataY, int widthBytes, int width, int height, int win_size);
        cls.SSIM_BYTE_SLOW = Loader.bind_func('SSIM_Byte_Slow', c_float, [c_ubyte_p, c_ubyte_p, c_int, c_int, c_int, c_int])

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


def psnr(
    x: Any, y: Any,
    width: int | None = None,
    height: int | None = None,
    channels: int | None = None,
    data_range: int | float | None = None,
) -> float:
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        x: The first image (e.g., original image), which can be 2D (grayscale) or 3D (color).
           Supported types include numpy arrays, bytes, bytearray, array.array, memoryviews,
           or any object with .width and .height attributes.
        y: The second image (e.g., reconstructed or noisy image). Must have the same
           dimensions and dtype as `x`.
        width (int | None, optional): Explicit width required only for raw byte buffers.
        height (int | None, optional): Explicit height required only for raw byte buffers.
        channels (int | None, optional): Explicit channels required only for raw byte buffers. Defaults to 1 (grayscale) if omitted.
        data_range (int | float | None, optional): The dynamic range of the pixel values (e.g., 255 for uint8
                                                    images, 1.0 for float images). If None, it defaults to 255
                                                    for 'uint8' and 255.0 for 'float32'.

    Returns:
        float: The PSNR value.

    Raises:
        ValueError: If the input images have unsupported dimensions or differing lengths.
        TypeError: If an unsupported dtype or typecode is provided.
    """
    w, h, c = _shape_of(x, width, height, channels)
    is_float = isinstance(x, array.array) and x.typecode in ('f', 'd')
    is_float = is_float or (type(x).__module__ == 'numpy' and 'float' in str(getattr(x, 'dtype', '')))

    if not DLL._initialized:
        DLL.initialize()

    func: Any = None
    ctype: Any = None

    if is_float:
        x = _as_float32_array(x)
        y = _as_float32_array(y)
        func = DLL.PSNR_FLOAT
        ctype = ctypes.c_float
        maxval = 255.0 if data_range is None else float(data_range)
    else:
        func = DLL.PSNR_BYTE
        ctype = ctypes.c_ubyte
        maxval = 255 if data_range is None else int(data_range)

    if func is None:
        type_str = getattr(x, 'dtype', getattr(x, 'typecode', type(x).__name__))
        raise TypeError(f"Unsupported dtype or typecode: {type_str}")

    xptr, xn, _xkeep = _get_ptr(x, ctype)
    yptr, yn, _ykeep = _get_ptr(y, ctype)
    if xn != yn:
        raise ValueError(f'Input images must have the same length. Got {xn} and {yn}')

    return float(func(xptr, yptr, w * c, w, h, maxval))


def ssim(
    x: Any, y: Any,
    width: int | None = None,
    height: int | None = None,
    channels: int | None = None,
    data_range: int | float | None = None,
    win_size: int = 7,
) -> float:
    """
    Calculates the Structural Similarity Index (SSIM) between two images.

    Args:
        x: The first image (e.g., original image), which can be 2D (grayscale) or 3D (color).
           Supported types include numpy arrays, bytes, bytearray, array.array, memoryviews,
           or any object with .width and .height attributes.
        y: The second image (e.g., reconstructed or noisy image). Must have the same
           dimensions and dtype as `x`.
        width (int | None, optional): Explicit width required only for raw byte buffers.
        height (int | None, optional): Explicit height required only for raw byte buffers.
        channels (int | None, optional): Explicit channels required only for raw byte buffers. Defaults to 1 (grayscale) if omitted.
        data_range (int | float | None, optional): The dynamic range of the pixel values (e.g., 255 for uint8
                                                    images, 1.0 for float images). If None, it defaults to 255
                                                    for 'uint8' and 255.0 for 'float32'.
        win_size (int, optional): The size of the sliding window for SSIM calculation. Defaults to 7.

    Returns:
        float: The SSIM value.

    Raises:
        ValueError: If the input images have unsupported dimensions or differing lengths.
        TypeError: If an unsupported dtype or typecode is provided.
    """
    w, h, c = _shape_of(x, width, height, channels)
    is_float = isinstance(x, array.array) and x.typecode in ('f', 'd')
    is_float = is_float or (type(x).__module__ == 'numpy' and 'float' in str(getattr(x, 'dtype', '')))

    if not DLL._initialized:
        DLL.initialize()

    func: Any = None
    ctype: Any = None

    if is_float:
        x = _as_float32_array(x)
        y = _as_float32_array(y)
        func = DLL.SSIM_FLOAT
        ctype = ctypes.c_float
        maxval = 255.0 if data_range is None else float(data_range)
    else:
        func = DLL.SSIM_BYTE
        ctype = ctypes.c_ubyte
        maxval = 255 if data_range is None else int(data_range)

    if func is None:
        type_str = getattr(x, 'dtype', getattr(x, 'typecode', type(x).__name__))
        raise TypeError(f"Unsupported dtype or typecode: {type_str}")

    xptr, xn, _xkeep = _get_ptr(x, ctype)
    yptr, yn, _ykeep = _get_ptr(y, ctype)
    if xn != yn:
        raise ValueError(f'Input images must have the same length. Got {xn} and {yn}')

    return float(func(xptr, yptr, w * c, w, h, win_size, maxval))


def ssim_slow(
    x: Any, y: Any,
    width: int | None = None,
    height: int | None = None,
    channels: int | None = None,
    win_size: int = 7,
) -> float:
    """
    Calculates the Structural Similarity Index (SSIM) using the unoptimized, scalar C++ fallback.

    Args:
        x: The first image (e.g., original image). Can be 2D (grayscale) or 3D (color).
           Supported dtype is 'uint8' only.
        y: The second image (e.g., reconstructed or noisy image). Must have the same
           dimensions and dtype as `x`.
        width (int | None, optional): Explicit width required only for raw byte buffers.
        height (int | None, optional): Explicit height required only for raw byte buffers.
        channels (int | None, optional): Explicit channels required only for raw byte buffers. Defaults to 1 (grayscale) if omitted.
        win_size (int, optional): The size of the sliding window for SSIM calculation. Defaults to 7.

    Returns:
        float: The SSIM value.

    Raises:
        ValueError: If the input images have unsupported dimensions or differing lengths.
        NotImplementedError: If the input images are not of type 'uint8'.
    """
    w, h, c = _shape_of(x, width, height, channels)

    if not DLL._initialized:
        DLL.initialize()

    func: Any = DLL.SSIM_BYTE_SLOW
    if func is None:
        raise NotImplementedError('ssim_slow is only implemented for uint8 data types.')

    xptr, xn, _xkeep = _get_ptr(x, ctypes.c_ubyte)
    yptr, yn, _ykeep = _get_ptr(y, ctypes.c_ubyte)
    if xn != yn:
        raise ValueError(f'Input images must have the same length. Got {xn} and {yn}')

    return float(func(xptr, yptr, w * c, w, h, win_size))
