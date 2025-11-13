import os
import sys
import ctypes
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
cuda_python_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
cuda_bindings_path = os.path.join(cuda_python_root, 'cuda_bindings')
cuda_core_path = os.path.join(cuda_python_root, 'cuda_core')
sys.path.insert(0, cuda_bindings_path)
sys.path.insert(0, cuda_core_path)

import cuda.bindings.driver as cuda
from cuda.bindings import cufile
from cuda.core.experimental._cufile._buffer_handle import BufferHandle
from cuda.core.experimental._cufile._file_handle import FileHandle
from cuda.core.experimental._cufile._driver_handle import DriverHandle
from cuda.core.experimental._cufile._io_params import IOParams
from cuda.core.experimental import Device


def main():
    """Example of asynchronously reading a file using cuFile with cuda.core memory resource."""
    
    size = 4096
    filename = "test-file-async.bin"

    test_data = np.arange(size, dtype=np.uint8)
    test_data.tofile(filename)
    os.sync()
    
    # Initialize CUDA using cuda.core Device
    dev = Device(0)
    dev.set_current()
    
    driver_handle = DriverHandle()
    
    # Allocate GPU buffer using cuda.core memory resource
    gpu_buffer = dev.allocate(size)
    
    buf_handle = BufferHandle(gpu_buffer, size, 0)
    
    file_handle = FileHandle(use_direct=True, flags=0, file_path=filename)
    
    io_params = IOParams(
        buffer=gpu_buffer,
        size=size,
        file_offset=0,
        buffer_offset=0
    )
    
    file_handle.read_async(io_params, stream=0)
    
    cuda.cuStreamSynchronize(0)
    
    bytes_read = io_params.get_bytes_done()
    assert bytes_read == size, f"Expected to read {size} bytes, got {bytes_read}"
    
    host_buffer = np.empty(size, dtype=np.uint8)
    err, = cuda.cuMemcpyDtoH(host_buffer.ctypes.data, int(gpu_buffer.handle), size)
    assert err == cuda.CUresult.CUDA_SUCCESS
    
    np.testing.assert_array_equal(test_data, host_buffer)

    # Cleanup - gpu_buffer is automatically freed when it goes out of scope!
    buf_handle.close()
    file_handle.close()
    driver_handle.close()
    
    if os.path.exists(filename):
        os.unlink(filename)


if __name__ == "__main__":
    main()


