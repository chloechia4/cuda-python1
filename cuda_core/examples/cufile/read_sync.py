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


def main():
    """Example of reading a file using cuFile."""
    
    # Test parameters
    size = 4096
    filename = "test-file.bin"

    # Step 1: Create test file with known data
    test_data = np.arange(size, dtype=np.uint8)
    test_data.tofile(filename)
    os.sync()
    
    # Step 2: Initialize CUDA
    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS
    
    err, device = cuda.cuDeviceGet(0)
    assert err == cuda.CUresult.CUDA_SUCCESS
    
    err, ctx = cuda.cuDevicePrimaryCtxRetain(device)
    assert err == cuda.CUresult.CUDA_SUCCESS
    
    (err,) = cuda.cuCtxSetCurrent(ctx)
    assert err == cuda.CUresult.CUDA_SUCCESS
    
    # Step 3: Open cuFile driver
    cufile.driver_open()
    
    # Step 4: Allocate GPU buffer
    err, buf_ptr = cuda.cuMemAlloc(size)
    assert err == cuda.CUresult.CUDA_SUCCESS
    buf_ptr_int = int(buf_ptr)
    
    # Step 5: Register buffer with cuFile
    buf_handle = BufferHandle(buf_ptr_int, size, 0)
    
    # Step 6: Open file with cuFile
    file_handle = FileHandle(use_direct=True, flags=0, file_path=filename)
    
    # Step 7: Read file into GPU buffer (like KvikIO f.read(b))
    bytes_read = file_handle.read(
        buffer=buf_ptr_int,
        size=size,
        file_offset=0,
        buf_offset=0
    )
    
    # Assert like KvikIO: f.read(b) == b.nbytes
    assert bytes_read == size, f"Expected to read {size} bytes, got {bytes_read}"
    
    # Step 8: Copy back to CPU and verify (like KvikIO xp.testing.assert_array_equal)
    host_buffer = np.empty(size, dtype=np.uint8)
    err, = cuda.cuMemcpyDtoH(host_buffer.ctypes.data, buf_ptr, size)
    assert err == cuda.CUresult.CUDA_SUCCESS
    
    # Verify data matches
    np.testing.assert_array_equal(test_data, host_buffer)

    # 1. Close handles BEFORE closing driver
    buf_handle.close()
    file_handle.close()
    # 2. Free GPU memory
    cuda.cuMemFree(buf_ptr)
    # 3. Close cuFile driver
    cufile.driver_close()
    # 4. Release CUDA context
    cuda.cuDevicePrimaryCtxRelease(device)
    
    if os.path.exists(filename):
        os.unlink(filename)



if __name__ == "__main__":
    main()

