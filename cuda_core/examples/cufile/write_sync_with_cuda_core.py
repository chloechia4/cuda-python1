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
from cuda.core.experimental import Device


def main():
    """Example of writing a file using cuFile with cuda.core memory resource."""
    
    size = 4096
    filename = "test-write.bin"
    
    test_data = np.arange(size, dtype=np.uint8)
    
    # Initialize CUDA using cuda.core Device
    dev = Device(0)
    dev.set_current()
    
    cufile.driver_open()
    
    # Allocate GPU buffer using cuda.core memory resource
    gpu_buffer = dev.allocate(size)
    
    # Copy data to GPU
    err, = cuda.cuMemcpyHtoD(int(gpu_buffer.handle), test_data.ctypes.data, size)
    assert err == cuda.CUresult.CUDA_SUCCESS
    
    buf_handle = BufferHandle(gpu_buffer, size, 0)
    
    file_handle = FileHandle(use_direct=True, flags=os.O_CREAT, file_path=filename)
    
    bytes_written = file_handle.write(
        buffer=gpu_buffer,
        size=size,
        file_offset=0,
        buf_offset=0
    )
    
    assert bytes_written == size, f"Expected to write {size} bytes, got {bytes_written}"
    
    written_data = np.fromfile(filename, dtype=np.uint8)
    np.testing.assert_array_equal(test_data, written_data)

    # Cleanup - gpu_buffer is automatically freed when it goes out of scope!
    buf_handle.close()
    file_handle.close()
    cufile.driver_close()
    
    if os.path.exists(filename):
        os.unlink(filename)


if __name__ == "__main__":
    main()

