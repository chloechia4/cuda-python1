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


def main():
    """Example of asynchronously writing a file using cuFile."""
    
    size = 4096
    filename = "test-write-async.bin"
    
    test_data = np.arange(size, dtype=np.uint8)
    
    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS
    
    err, device = cuda.cuDeviceGet(0)
    assert err == cuda.CUresult.CUDA_SUCCESS
    
    err, ctx = cuda.cuDevicePrimaryCtxRetain(device)
    assert err == cuda.CUresult.CUDA_SUCCESS
    
    (err,) = cuda.cuCtxSetCurrent(ctx)
    assert err == cuda.CUresult.CUDA_SUCCESS
    
    driver_handle = DriverHandle()
    
    err, buf_ptr = cuda.cuMemAlloc(size)
    assert err == cuda.CUresult.CUDA_SUCCESS
    buf_ptr_int = int(buf_ptr)
    
    err, = cuda.cuMemcpyHtoD(buf_ptr, test_data.ctypes.data, size)
    assert err == cuda.CUresult.CUDA_SUCCESS
    
    buf_handle = BufferHandle(buf_ptr_int, size, 0)
    
    file_handle = FileHandle(use_direct=True, flags=os.O_CREAT, file_path=filename)
    
    io_params = IOParams(
        buffer=buf_ptr_int,
        size=size,
        file_offset=0,
        buffer_offset=0
    )
    
    file_handle.write_async(io_params, stream=0)
    
    cuda.cuStreamSynchronize(0)
    
    bytes_written = io_params.get_bytes_done()
    assert bytes_written == size, f"Expected to write {size} bytes, got {bytes_written}"
    
    buf_handle.close()
    file_handle.close()
    
    written_data = np.fromfile(filename, dtype=np.uint8)
    np.testing.assert_array_equal(test_data, written_data)
    
    cuda.cuMemFree(buf_ptr)
    driver_handle.close()
    cuda.cuDevicePrimaryCtxRelease(device)
    
    if os.path.exists(filename):
        os.unlink(filename)


if __name__ == "__main__":
    main()

