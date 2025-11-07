import os
import sys
import ctypes

current_dir = os.path.dirname(os.path.abspath(__file__))
cuda_python_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
cuda_bindings_path = os.path.join(cuda_python_root, 'cuda_bindings')
cuda_core_path = os.path.join(cuda_python_root, 'cuda_core')
sys.path.insert(0, cuda_bindings_path)
sys.path.insert(0, cuda_core_path)

import cuda.bindings.driver as cuda
from cuda.bindings import cufile # This is for the Driver Open/Close
from cuda.core.experimental._cufile._buffer_handle import BufferHandle
from cuda.core.experimental._cufile._file_handle import FileHandle


def cufile_read_example():
    """Minimal example of reading a file using cuFile."""
    
    buf_size = 4 * 1024
    file_path = "example_data.bin"
    
    try:
        # Initialize CUDA
        (err,) = cuda.cuInit(0)
        assert err == cuda.CUresult.CUDA_SUCCESS
        
        err, device = cuda.cuDeviceGet(0)
        assert err == cuda.CUresult.CUDA_SUCCESS
        
        err, ctx = cuda.cuDevicePrimaryCtxRetain(device)
        assert err == cuda.CUresult.CUDA_SUCCESS
        
        (err,) = cuda.cuCtxSetCurrent(ctx)
        assert err == cuda.CUresult.CUDA_SUCCESS
        
        # Open cuFile driver
        cufile.driver_open()
        
        # Allocate GPU memory
        err, buf_ptr = cuda.cuMemAlloc(buf_size)
        assert err == cuda.CUresult.CUDA_SUCCESS
        buf_ptr_int = int(buf_ptr)
        
        # Register buffer with cuFile
        buf_handle = BufferHandle(buf_ptr_int, buf_size, 0)
        
        # Create test file
        test_data = b"Hello cuFile! This is sample data. " * (buf_size // 36 + 1)
        test_data = test_data[:buf_size]
        
        with open(file_path, 'wb') as f:
            f.write(test_data)
        
        # Open file handle for cuFile
        file_handle = FileHandle(use_direct=True, flags=0, file_path=file_path)
        
        # Perform the read operation using FileHandle.read()
        bytes_read = file_handle.read(
            buffer=buf_ptr_int,
            size=buf_size,
            file_offset=0,
            buf_offset=0
        )
        assert bytes_read == buf_size
        
        # Optional: Verify data
        host_buffer = (ctypes.c_byte * buf_size)()
        err, = cuda.cuMemcpyDtoH(host_buffer, buf_ptr, buf_size)
        assert err == cuda.CUresult.CUDA_SUCCESS
        
        read_data = bytes(host_buffer)
        assert read_data == test_data
        
        # Cleanup
        cuda.cuMemFree(buf_ptr)
        cufile.driver_close()
        cuda.cuDevicePrimaryCtxRelease(device)
        
        if os.path.exists(file_path):
            os.unlink(file_path)
        
    except Exception as e:
        if os.path.exists(file_path):
            os.unlink(file_path)
        raise


if __name__ == "__main__":
    cufile_read_example()

