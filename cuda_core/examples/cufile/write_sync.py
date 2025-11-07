import os
import sys
import time
import ctypes

current_dir = os.path.dirname(os.path.abspath(__file__))
cuda_python_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
cuda_bindings_path = os.path.join(cuda_python_root, 'cuda_bindings')
sys.path.insert(0, cuda_bindings_path)

import cuda.bindings.driver as cuda
from cuda.bindings import cufile
from cuda_handles import BufferHandle, FileHandle

def cufile_write_example():
    """Minimal example of writing a file using cuFile."""
    
    buf_size = 4 * 1024
    file_path = "example_output.bin"
    
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
        
        # Prepare test data
        test_data = b"Hello cuFile! This is sample data for write operations. " * (buf_size // 58 + 1)
        test_data = test_data[:buf_size]
        host_buf = ctypes.create_string_buffer(buf_size)
        ctypes.memmove(host_buf, test_data, buf_size)
        
        # Copy test data to GPU
        (err,) = cuda.cuMemcpyHtoDAsync(buf_ptr, host_buf, buf_size, 0)
        assert err == cuda.CUresult.CUDA_SUCCESS
        (err,) = cuda.cuStreamSynchronize(0)
        assert err == cuda.CUresult.CUDA_SUCCESS
        
        # Open file handle for cuFile
        fd = os.open(file_path, os.O_CREAT | os.O_RDWR | os.O_DIRECT, 0o600)
        file_handle = FileHandle(use_direct=True, flags=0, file_path=file_path)
        
        # Perform the write operation
        bytes_written = cufile.write(
            file_handle._handle,
            buf_ptr_int,
            buf_size,
            0,
            0
        )
        assert bytes_written == buf_size
        
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
    cufile_write_example()


