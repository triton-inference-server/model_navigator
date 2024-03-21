# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CUDA utils."""

import copy
import ctypes
import math
import os
import sys
from typing import Any, Mapping, Optional, Sequence, Tuple, Type, Union

import numpy as np

import model_navigator.utils.common as utils
from model_navigator.core.logger import LOGGER
from model_navigator.exceptions import ModelNavigatorError, ModelNavigatorUserInputError
from model_navigator.frameworks import is_torch_available
from model_navigator.frameworks.tensorrt import type_mapping as trt_type_mapping
from model_navigator.utils import module

torch = module.lazy_import("torch")
trt = module.lazy_import("tensorrt")


def void_ptr(val=None):
    """Returns a void pointer to the given value."""
    return ctypes.c_void_p(val)


class MemcpyKind:
    """Enumerates different kinds of copy operations."""

    HostToHost = ctypes.c_int(0)
    """Copies from host memory to host memory"""
    HostToDevice = ctypes.c_int(1)
    """Copies from host memory to device memory"""
    DeviceToHost = ctypes.c_int(2)
    """Copies from device memory to host memory"""
    DeviceToDevice = ctypes.c_int(3)
    """Copies from device memory to device memory"""
    Default = ctypes.c_int(4)


class StreamCaptureMode:
    """Enumerates different modes for capturing CUDA graphs."""

    Global = 0
    """Default mode: After stream capture is initiated, attempting to capture on a different stream will return an error."""
    ThreadLocal = 1
    """Thread-local mode: After stream capture is initiated, attempting to capture on a different stream will succeed and any captured work will go into a separate per-thread capture sequence."""
    Relaxed = 2
    """Relaxed mode: After stream capture is initiated, attempting to capture on a different stream will succeed and any captured work will go into the same capture sequence."""


class GraphInstantiateFlags:
    """Enumerates different flags for instantiating CUDA graphs."""

    NoneFlag = 0
    """No flags"""
    AutoFreeOnLaunch = 1
    """Auto-free mode: The graph exec object will be automatically freed after a successful launch."""


class Cuda:
    """Wrapper that exposes low-level CUDA functionality.

    NOTE: Do *not* construct this class manually.
    Instead, use the ``wrapper()`` function to get the global wrapper.
    """

    def __init__(self):
        """Initialize the CUDA wrapper."""
        self.handle = None

        fallback_lib = None
        if sys.platform.startswith("win"):
            cuda_paths = [os.environ.get("CUDA_PATH", "")]
            cuda_paths += os.environ.get("PATH", "").split(os.path.pathsep)
            lib_pat = "cudart64_*.dll"
        else:
            cuda_paths = [
                *os.environ.get("LD_LIBRARY_PATH", "").split(os.path.pathsep),
                os.path.join("/", "usr", "local", "cuda", "lib64"),
                os.path.join("/", "usr", "lib"),
                os.path.join("/", "lib"),
            ]
            lib_pat = "libcudart.so*"
            fallback_lib = "libcudart.so"

        cuda_paths = list(filter(lambda x: x, cuda_paths))  # Filter out empty paths (i.e. "")

        candidates = utils.find_in_dirs(lib_pat, cuda_paths)
        if not candidates:
            err_msg = f"Could not find the CUDA runtime library.\nNote: Paths searched were:\n{cuda_paths}"
            if fallback_lib is None:
                raise ModelNavigatorError(err_msg)
            else:
                LOGGER.warning(err_msg)

            lib = fallback_lib
            LOGGER.warning(f"Attempting to load: '{lib}' using default loader paths")
        else:
            LOGGER.info(f"Found candidate CUDA libraries: {candidates}")
            lib = candidates[0]

        self.handle = ctypes.CDLL(lib)

        if not self.handle:
            raise ModelNavigatorError("Could not load the CUDA runtime library. Is it on your loader path?")

    def check(self, status):
        """Check CUDA status and raise an exception if it is not 0."""
        if status != 0:
            raise ModelNavigatorError(
                f"CUDA Error: {status}. To figure out what this means, refer to https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3f51e3575c2178246db0a94a430e0038"
            )

    # extern __host__ ​cudaError_t cudaSetDevice(int  device);
    def set_device(self, device: int):
        """Set the current CUDA device.

        Args:
            device (int): The device index to set.
        """
        # Signature: int -> None
        self.check(self.handle.cudaSetDevice(ctypes.c_int(device)))

    # extern __host__ ​ __device__ ​cudaError_t cudaGetDevice(int* device)
    def get_device(self):
        """Get the current CUDA device.

        Returns:
            int: The current device index.
        """
        # Signature: None -> int
        device = ctypes.c_int()
        self.check(self.handle.cudaGetDevice(ctypes.byref(device)))
        return device.value

    # extern __host__ cudaError_t CUDARTAPI cudaStreamCreate(cudaStream_t *pStream);
    def stream_create(self):
        """Create CUDA stream."""
        # Signature: () -> int
        ptr = void_ptr()
        self.check(self.handle.cudaStreamCreate(ctypes.byref(ptr)))
        return ptr.value

    # extern __host__ cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream);
    def stream_synchronize(self, ptr):
        """Synchronize CUDA stream."""
        # Signature: int -> None
        self.check(self.handle.cudaStreamSynchronize(void_ptr(ptr)))

    # extern __host__ cudaError_t CUDARTAPI cudaStreamBeginCapture(cudaStream_t stream, enum cudaStreamCaptureMode mode);
    def stream_begin_capture(self, ptr: int, mode: int = StreamCaptureMode.Global):
        """Begin capturing a CUDA stream.

        Args:
            ptr (int): The handle to the stream to capture.
            mode (StreamCaptureMode): The mode to use for capturing.
        """
        # Signature: int, int -> None
        self.check(self.handle.cudaStreamBeginCapture(void_ptr(ptr), mode))

    # extern __host__ cudaError_t CUDARTAPI cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t *pGraph);
    def stream_end_capture(self, ptr: int):
        """End capturing a CUDA stream.

        Args:
            ptr (int): The handle to the stream to capture.

        Returns:
            int: The handle to the created graph.
        """
        # Signature: int, int -> None
        graph = void_ptr()
        self.check(self.handle.cudaStreamEndCapture(void_ptr(ptr), ctypes.byref(graph)))
        return graph.value

    # extern __host__ cudaError_t CUDARTAPI cudaGraphInstantiate(cudaGraphExec_t *pGraphExec, cudaGraph_t graph, unsigned long long flags __dv(0));
    def graph_instantiate(self, ptr: int, flags: int = GraphInstantiateFlags.NoneFlag):
        """Instantiate a CUDA graph.

        Args:
            ptr (int): The handle to the graph to instantiate.
            flags (GraphInstantiateWithFlags): The flags to use for instantiation.

        Returns:
            int: The handle to the created graph execution.
        """
        # Signature: int, int -> None
        graph_exec = void_ptr()
        self.check(self.handle.cudaGraphInstantiate(ctypes.byref(graph_exec), void_ptr(ptr), ctypes.c_int(flags)))
        return graph_exec.value

    # extern __host__ cudaError_t CUDARTAPI cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream);
    def graph_launch(self, ptr: int, stream: int):
        """Launch a CUDA graph.

        Args:
            ptr (int): The handle to the graph execution to launch.
            stream (int): The handle to the stream to launch the graph on.
        """
        # Signature: int, int -> None
        self.check(self.handle.cudaGraphLaunch(void_ptr(ptr), void_ptr(stream)))

    # extern __host__ cudaError_t CUDARTAPI cudaGraphExecDestroy(cudaGraphExec_t graphExec);
    def graph_exec_destroy(self, ptr: int):
        """Destroy CUDA graph execution."""
        # Signature: int -> None
        self.check(self.handle.cudaGraphExecDestroy(void_ptr(ptr)))

    # extern __host__ cudaError_t CUDARTAPI cudaGraphDestroy(cudaGraph_t graph);
    def graph_destroy(self, ptr: int):
        """Destroy CUDA graph."""
        # Signature: int -> None
        self.check(self.handle.cudaGraphDestroy(void_ptr(ptr)))

    # extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream);
    def stream_destroy(self, ptr):
        """Destroy CUDA stream."""
        # Signature: int -> None
        self.check(self.handle.cudaStreamDestroy(void_ptr(ptr)))

    # extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size);
    def malloc(self, nbytes: int) -> int:
        """Allocates memory on the GPU.

        Args:
            nbytes: The number of bytes to allocate.

        Returns:
            he memory address of the allocated region, i.e. a device pointer.
        """
        ptr = void_ptr()
        nbytes = ctypes.c_size_t(nbytes)  # Required to prevent overflow
        self.check(self.handle.cudaMalloc(ctypes.byref(ptr), nbytes))
        return ptr.value

    # extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size);
    def free(self, ptr: int):
        """Frees memory allocated on the GPU.

        Args:
            ptr (int): The memory address of the allocated region, i.e. a device pointer.
        """
        self.check(self.handle.cudaFree(void_ptr(ptr)))

    def memcpy(self, dst: int, src: int, nbytes: int, kind: MemcpyKind, stream_ptr: Optional[int] = None):
        """Copies data between host and device memory.

        Args:
            dst: The memory address of the destination, i.e. a pointer.
            src: The memory address of the source, i.e. a pointer.
            nbytes: The number of bytes to copy.
            kind: The kind of copy to perform.
            stream_ptr: The memory address of a CUDA stream, i.e. a pointer.
                    If this is not provided, a synchronous copy is performed.
        """
        nbytes = ctypes.c_size_t(nbytes)  # Required to prevent overflow
        if stream_ptr is not None:
            # extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
            self.check(self.handle.cudaMemcpyAsync(void_ptr(dst), void_ptr(src), nbytes, kind, void_ptr(stream_ptr)))
        else:
            # extern __host__ cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
            self.check(self.handle.cudaMemcpy(void_ptr(dst), void_ptr(src), nbytes, kind))


G_CUDA = None


def wrapper():
    """Returns the global Polygraphy CUDA wrapper.

    Returns:
        Cuda: The global CUDA wrapper.
    """
    global G_CUDA
    if G_CUDA is None:
        G_CUDA = Cuda()
    return G_CUDA


class CudaDeviceSelector:
    """A context manager that sets the current CUDA device to the specified device for current thread."""

    def __init__(self, device: int):
        """Initializes a CUDA device context.

        Args:
            device (int): The device index to set.
        """
        self.device = device

    def __enter__(self):
        """Sets the current CUDA device."""
        self.prev_device = wrapper().get_device()
        wrapper().set_device(self.device)

    def __exit__(self, exc_type, exc_value, traceback):
        """Resets the current CUDA device to the previous device."""
        wrapper().set_device(self.prev_device)


class GraphExec:
    """High-level wrapper for CUDA graph execution.

    GraphExec is a context manager, so it can be used with the `with` statement. When the context
    is exited, the underlying CUDA graph is freed. For example:

    .. code-block:: python

        with Graph.launch(graph) as graph_exec:
            graph_exec.launch(stream)
    GraphExec can also be used without the `with` statement, but in that case, the user is responsible
    for freeing the underlying CUDA graph.

    GraphExec is instantiated by calling `Graph.launch`.

    """

    def __init__(self, ptr):
        """Creates a new CUDA graph execution wrapper. Never call this directly - use `Graph.launch` instead.

        Args:
            ptr (int): The memory address of the CUDA graph execution, i.e. a pointer.
        """
        self.ptr = ptr

    def __enter__(self):
        """Returns this object."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Frees the underlying CUDA graph."""
        self.free()

    def free(self):
        """Frees the underlying CUDA graph."""
        wrapper().graph_exec_destroy(self.ptr)
        self.ptr = ctypes.c_void_p(None)

    def launch(self, stream):
        """Launches the CUDA graph.

        Args:
            stream (Stream): The CUDA stream to use for execution.
        """
        if not isinstance(stream, Stream):
            raise TypeError(f"Expected stream to be a Stream, but got: {type(stream)}")
        wrapper().graph_launch(self.ptr, stream.ptr)


class Graph:
    """High-level wrapper for CUDA graphs.

    Graph is instantiated by calling `Stream.end_capture`.
    """

    def __init__(self, ptr):
        """Creates a new CUDA graph wrapper. Never call this directly - use `Stream.end_capture` instead.

        Args:
            ptr (int): The memory address of the CUDA graph, i.e. a pointer.
        """
        self.ptr = ptr

    def __enter__(self):
        """Returns this object."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Frees the underlying CUDA graph."""
        self.free()

    def free(self):
        """Frees the underlying CUDA graph."""
        wrapper().graph_destroy(self.ptr)
        self.ptr = ctypes.c_void_p(None)

    def instantiate(self, flags=GraphInstantiateFlags.NoneFlag):
        """Instantiates the CUDA exec graph.

        Returns:
            GraphExec: A handle to the instantiated CUDA graph.
        """
        return GraphExec(wrapper().graph_instantiate(self.ptr, flags))


class Stream:
    """High-level wrapper for a CUDA stream."""

    def __init__(self):
        """Initializes CUDA stream."""
        self.ptr = wrapper().stream_create()
        """int: The memory address of the underlying CUDA stream"""

    def __enter__(self):
        """Enters CUDA context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Frees the underlying CUDA stream."""
        self.free()

    def free(self):
        """Frees the underlying CUDA stream.

        You can also use a context manager to manage the stream lifetime.
        """
        wrapper().stream_destroy(self.ptr)
        self.ptr = ctypes.c_void_p(None)

    def synchronize(self):
        """Synchronizes the stream."""
        wrapper().stream_synchronize(self.ptr)

    def begin_capture(self, mode=StreamCaptureMode.Global):
        """Begins capturing graph on the stream.

        Args:
            mode (StreamCaptureMode): The capture mode to use.
        """
        wrapper().stream_begin_capture(self.ptr, mode)

    def end_capture(self):
        """Ends capturing graph on the stream.

        Returns:
            Graph: A handle to the captured CUDA graph.
        """
        return Graph(wrapper().stream_end_capture(self.ptr))


class DeviceView:
    """A read-only view of a GPU memory region."""

    def __init__(self, ptr: int, shape: Tuple[int], dtype: Union[np.dtype, Type[np.dtype]]):
        """Initializes a device view.

        Args:
            ptr: A pointer to the region of memory.

            shape: The shape of the region.
            dtype: The data type of the region.
        """
        self.ptr = int(ptr)
        """int: The memory address of the underlying GPU memory"""
        self.shape = shape
        """Tuple[int]: The shape of the device buffer"""
        self.itemsize = None
        self.dtype = dtype
        """np.dtype: The data type of the device buffer"""

        self._torch_array = None

    def torch(self):
        """Returns a torch array of the device buffer."""
        if self._torch_array is None:
            self._torch_array = self._torch()
        return self._torch_array

    @property
    def dtype(self):
        """Get the data type of the device buffer."""
        return self._dtype

    @dtype.setter
    def dtype(self, new):
        """Set the data type of the device buffer."""
        self._dtype = new

        if isinstance(new, trt.tensorrt.DataType):
            self.itemsize = self._dtype.itemsize
        elif is_torch_available() and isinstance(new, torch.dtype):
            np_dtype = utils.torch_to_numpy_dtype(new)
            self.itemsize = np_dtype().itemsize
        else:
            self.itemsize = np.dtype(new).itemsize

    @property
    def nbytes(self):
        """The number of bytes in the memory region."""
        return utils.volume(self.shape) * self.itemsize

    @property
    def __class__(self):
        """Returns the class of the device buffer."""
        if is_torch_available():
            return torch.Tensor
        return type(self)

    def __getattr__(self, __name: str) -> Any:
        """Returns the attribute of the torch array."""
        return getattr(self.torch(), __name)

    def __getitem__(self, key: Any) -> Any:
        """Returns the item of the torch array."""
        return self.torch()[key]

    def __str__(self):
        """Returns a string representation of the device buffer."""
        return f"DeviceView[(dtype={np.dtype(self.dtype).name}, shape={self.shape}), ptr={hex(self.ptr)}]"

    def __repr__(self):
        """Returns a string representation of the device buffer."""
        return _make_repr("DeviceView", ptr=self.ptr, shape=self.shape, dtype=self.dtype)[0]

    def copy_to(self, host_buffer: np.ndarray, stream: Optional[Stream] = None) -> np.ndarray:
        """Copies from this device buffer to the provided host buffer.

        Args:
            host_buffer: The host buffer to copy into. The buffer must be contiguous in
                    memory (see np.ascontiguousarray) and large enough to accommodate
                    the device buffer.
            stream: A Stream instance. Performs a synchronous copy if no stream is provided.

        Returns:
            The host buffer
        """
        if not self.nbytes:
            return host_buffer

        self._check_host_buffer(host_buffer, copying_from=False)
        wrapper().memcpy(
            dst=host_buffer.ctypes.data,
            src=self.ptr,
            nbytes=self.nbytes,
            kind=MemcpyKind.DeviceToHost,
            stream_ptr=try_get_stream_handle(stream),
        )
        return host_buffer

    def copy_to_device(self, device_buffer: "DeviceView", stream: Optional[Stream] = None) -> "DeviceView":
        """Copies from this device buffer to the provided device buffer.

        Args:
            device_buffer: The device buffer to copy into.
            stream: A Stream instance. Performs a synchronous copy if no stream is provided.

        Returns:
            The host buffer
        """
        wrapper().memcpy(
            dst=device_buffer.ptr,
            src=self.ptr,
            nbytes=self.nbytes,
            kind=MemcpyKind.DeviceToDevice,
            stream_ptr=try_get_stream_handle(stream),
        )
        return device_buffer

    def numpy(self) -> np.ndarray:
        """Create a new NumPy array containing the contents of this device buffer.

        Returns:
            The newly created NumPy array.
        """
        arr = np.empty(self.shape, dtype=self.dtype)
        self.copy_to(arr)
        return arr

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Dispatches torch functions to the underlying torch array."""
        if kwargs is None:
            kwargs = {}

        args = cls._to_torch(args)
        kwargs = cls._to_torch(kwargs)
        return func(*args, **kwargs)

    @classmethod
    def _to_torch(cls, obj):
        if isinstance(obj, cls):
            return obj.torch()
        if isinstance(obj, Mapping):
            return {k: cls._to_torch(v) for k, v in obj.items()}
        if isinstance(obj, Sequence):
            return [cls._to_torch(v) for v in obj]
        return obj

    def _torch(self, stream=None):
        LOGGER.debug(
            f"Creating torch array from device buffer.\n"
            f"Note: dtype={self.dtype}, shape={self.shape}, nbytes={self.nbytes}, ptr={hex(self.ptr)}"
        )

        if isinstance(self.dtype, trt.tensorrt.DataType):
            dtype = trt_type_mapping.trt_to_torch_dtype(self.dtype)
        elif is_torch_available() and not isinstance(self.dtype, torch.dtype):
            dtype = utils.numpy_to_torch_dtype(self.dtype)
        else:
            dtype = self.dtype

        arr = torch.empty(self.shape, dtype=dtype, device="cuda")
        wrapper().memcpy(
            dst=arr.data_ptr(),
            src=self.ptr,
            nbytes=self.nbytes,
            kind=MemcpyKind.DeviceToDevice,
            stream_ptr=try_get_stream_handle(stream),
        )

        return arr

    def _check_host_buffer(self, host_buffer, copying_from):  # pytype: disable=attribute-error
        if host_buffer.dtype != self.dtype:
            raise ModelNavigatorUserInputError(
                f"Host buffer type: {host_buffer.dtype} does not match the type of this device buffer: {self.dtype}. This may cause CUDA errors!"
            )

        if not utils.is_contiguous(host_buffer):
            raise ModelNavigatorUserInputError(
                "Provided host buffer is not contiguous in memory.\n"
                "Hint: Use `util.make_contiguous()` or `np.ascontiguousarray()` to make the array contiguous in memory."
            )

        # If the host buffer is an input, the device buffer should be large enough to accommodate it.
        # Otherwise, the host buffer needs to be large enough to accommodate the device buffer.
        if copying_from:
            if host_buffer.nbytes > self.nbytes:
                raise ModelNavigatorUserInputError(
                    f"Provided host buffer is larger than device buffer.\n"
                    f"Note: host buffer is {host_buffer.nbytes} bytes but device buffer is only {self.nbytes} bytes.\n"
                    f"Hint: Use `resize()` to resize the device buffer to the correct shape."
                )
        else:
            if host_buffer.nbytes < self.nbytes:
                raise ModelNavigatorUserInputError(
                    f"Provided host buffer is smaller than device buffer.\n"
                    f"Note: host buffer is only {host_buffer.nbytes} bytes but device buffer is {self.nbytes} bytes.\n"
                    f"Hint: Use `util.resize_buffer()` to resize the host buffer to the correct shape."
                )


def try_get_stream_handle(stream):
    """Returns the stream handle if the stream is not None, otherwise returns None."""
    if stream is None:
        return None
    return stream.ptr


class DeviceArray(DeviceView):
    """An array on the GPU."""

    def __init__(self, shape: Optional[Tuple[int]] = None, dtype: Optional[np.dtype] = None):
        """Initializes a device array.

        Args:
            shape: The initial shape of the buffer.
            dtype: The data type of the buffer.
        """
        super().__init__(ptr=0, shape=utils.default(shape, ()), dtype=utils.default(dtype, np.float32))
        self.allocated_nbytes = 0
        self.resize(self.shape)

    def __enter__(self):
        """Enter DeviceArray context."""
        return self

    @staticmethod
    def raw(shape: Tuple) -> "DeviceArray":
        """Creates an untyped device array of the specified shape.

        Args:
            shape: The initial shape of the buffer, in units of bytes.
                For example, a shape of ``(4, 4)`` would allocate a 16 byte array.

        Returns:
            The raw device array.
        """
        return DeviceArray(shape=shape, dtype=np.byte)

    def resize(self, shape: Tuple[int]):
        """Resizes or reshapes the array to the specified shape.

        If the allocated memory region is already large enough,
        no reallocation is performed.

        Args:
            shape: The new shape.
        """
        nbytes = utils.volume(shape) * self.itemsize
        if nbytes > self.allocated_nbytes:
            self.free()
            self.ptr = wrapper().malloc(nbytes)
            self.allocated_nbytes = nbytes
        self.shape = shape

    def __exit__(self, exc_type, exc_value, traceback):
        """Frees the underlying memory of this DeviceArray."""
        self.free()

    def free(self):
        """Frees the GPU memory associated with this array.

        You can also use a context manager to ensure that memory is freed.
        """
        wrapper().free(self.ptr)
        self.shape = ()
        self.allocated_nbytes = 0
        self.ptr = 0

    def copy_from(self, host_buffer: np.ndarray, stream: Optional[Stream] = None) -> "DeviceArray":
        """Copies from the provided host buffer into this device buffer.

        Args:
            host_buffer: The host buffer to copy from. The buffer must be contiguous in
                    memory (see np.ascontiguousarray) and not larger than this device buffer.
            stream: A Stream instance. Performs a synchronous copy if no stream is provided.

        Returns:
            self
        """
        if not host_buffer.nbytes:
            return self

        self._check_host_buffer(host_buffer, copying_from=True)
        wrapper().memcpy(
            dst=self.ptr,
            src=host_buffer.ctypes.data,
            nbytes=host_buffer.nbytes,
            kind=MemcpyKind.HostToDevice,
            stream_ptr=try_get_stream_handle(stream),
        )
        return self

    def view(self, shape: Optional[Sequence[int]] = None, dtype: Optional[np.dtype] = None) -> DeviceView:
        """Creates a read-only DeviceView from this DeviceArray.

        Args:
            shape: The desired shape of the view.
                Defaults to the shape of this array or view.
            dtype: The desired data type of the view.
                Defaults to the data type of this array or view.

        Returns:
            A view of this arrays data on the device.
        """
        shape = utils.default(shape, self.shape)
        dtype = utils.default(dtype, self.dtype)
        view = DeviceView(self.ptr, shape, dtype)

        if view.nbytes > self.nbytes:
            raise ModelNavigatorError(
                "A view cannot exceed the number of bytes of the original array.\n"
                f"Note: Original array has shape: {self.shape} and dtype: {self.dtype}, which requires {self.nbytes} bytes, "
                f"while the view has shape: {shape} and dtype: {dtype}, which requires {view.nbytes} bytes, "
            )
        return view

    def __str__(self):
        """Returns a string representation of the device buffer."""
        return f"DeviceArray[(dtype={np.dtype(self.dtype).name}, shape={self.shape}), ptr={hex(self.ptr)}]"

    def __repr__(self):
        """Returns a string representation of the device buffer."""
        return _make_repr("DeviceArray", shape=self.shape, dtype=self.dtype)[0]


def is_nan(obj) -> bool:
    """Checks if passed object is NaN.

    Args:
        obj: Object to check

    Returns:
        bool: True if object is NaN, False otherwise
    """
    return isinstance(obj, float) and math.isnan(obj)


def is_inf(obj) -> bool:
    """Checks if passed object is inf.

    Args:
        obj: Object to check

    Returns:
        bool: True if object is inf, False otherwise
    """
    return isinstance(obj, float) and math.isinf(obj)


# Some objects don't have correct `repr` implementations, so we need to handle them specially.
# For other objects, we do nothing.
def _handle_special_repr(obj):
    # 1. Work around incorrect `repr` implementations

    # Use a special __repr__ override so that we can inline strings
    class InlineString(str):
        def __repr__(self) -> str:
            return self

    if is_nan(obj) or is_inf(obj):
        return InlineString(f"float('{obj}')")

    # 2. If this object is a collection, recursively apply this logic.
    # Note that we only handle the built-in collections here, since custom collections
    # may have special behavior that we don't know about.

    if type(obj) not in [tuple, list, dict, set]:
        return obj

    obj = copy.copy(obj)
    # Tuple needs special handling since it doesn't support assignment.
    if isinstance(obj, tuple):
        args = tuple(_handle_special_repr(elem) for elem in obj)
        obj = type(obj)(args)
    elif isinstance(obj, list):
        for index, elem in enumerate(obj):
            obj[index] = _handle_special_repr(elem)
    elif isinstance(obj, dict):
        new_items = {}
        for key, value in obj.items():
            new_items[_handle_special_repr(key)] = _handle_special_repr(value)
        obj.clear()
        obj.update(new_items)
    elif isinstance(obj, set):
        new_elems = set()
        for value in obj:
            new_elems.add(_handle_special_repr(value))
        obj.clear()
        obj.update(new_elems)

    # 3. Finally, return the modified version of the object
    return obj


def _apply_repr(obj):
    obj = _handle_special_repr(obj)
    return repr(obj)


def _make_repr(type_str, *args, **kwargs):
    """Creates a string suitable for use with ``__repr__`` for a given type with the provided arguments.

    Skips keyword arguments that are set to ``None``.

    For example, ``make_repr("Example", None, "string", w=None, x=2)``
    would return a string: ``"Example(None, 'string', x=2)"``

    Args:
        type_str (str):
                The name of the type to create a representation for.
        args: Arguments to represent
        kwargs: Keyword arguments to represent

    Returns:
        Tuple[str, bool, bool]:
                A tuple including the ``__repr__`` string and two booleans
                indicating whether all the positional and keyword arguments were default
                (i.e. None) respectively.
    """
    processed_args = list(map(_apply_repr, args))

    processed_kwargs = []
    for key, val in filter(lambda t: t[1] is not None, kwargs.items()):
        processed_kwargs.append(f"{key}={_apply_repr(val)}")

    repr_str = f"{type_str}({', '.join(processed_args + processed_kwargs)})"

    def all_default(arg_list):
        return all(arg == _apply_repr(None) for arg in arg_list)

    return repr_str, all_default(processed_args), all_default(processed_kwargs)
