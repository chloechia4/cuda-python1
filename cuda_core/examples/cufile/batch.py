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
from cuda.core.experimental._cufile._batch_handle import (
    BatchHandle, 
    make_read_operation, 
    make_write_operation,
    BatchIOResult
)
from cuda.core.experimental import Device


def main():
    """Example of batch I/O operations using cuFile with cuda.core memory resource."""
    
    num_operations = 3
    chunk_size = 4096  # 4KB per chunk
    total_size = chunk_size * num_operations
    filename = "test-batch-file.bin"
    
    test_data = np.arange(total_size, dtype=np.uint8)
    test_data.tofile(filename)
    os.sync()
    
    # Initialize CUDA using cuda.core Device
    dev = Device(0)
    dev.set_current()
    
    driver_handle = DriverHandle()
    
    gpu_buffers = []
    buf_handles = []
    
    # Allocate GPU buffers using cuda.core memory resource
    for i in range(num_operations):
        gpu_buffer = dev.allocate(chunk_size)
        gpu_buffers.append(gpu_buffer)
        
        buf_handle = BufferHandle(gpu_buffer, chunk_size, 0)
        buf_handles.append(buf_handle)
        
    file_handle = FileHandle(use_direct=True, flags=0, file_path=filename)
    
    with BatchHandle(max_operations=num_operations) as batch:
        
        operations = []
        for i in range(num_operations):
            file_offset = i * chunk_size
            cookie = {"operation_id": i, "offset": file_offset}
            
            op = make_read_operation(
                file_handle=file_handle,
                buffer=gpu_buffers[i],
                size=chunk_size,
                file_offset=file_offset,
                buffer_offset=0,
                cookie=cookie
            )
            operations.append(op)
        
        
        batch.submit(operations)
        
        results = batch.execute_operations()
        
        for i, result in enumerate(results):
            if result.is_complete():
                print(f"  Operation {i} (cookie: {result.cookie}): ✓ {result.result} bytes")
            elif result.has_error():
                print(f"  Operation {i}: ✗ Error: {result.error}")
            else:
                print(f"  Operation {i}: Incomplete")
        
        assert all(r.is_complete() for r in results), "Not all operations completed successfully"
    
    
    for i in range(num_operations):
        host_buffer = np.empty(chunk_size, dtype=np.uint8)
        err, = cuda.cuMemcpyDtoH(host_buffer.ctypes.data, int(gpu_buffers[i].handle), chunk_size)
        assert err == cuda.CUresult.CUDA_SUCCESS
        
        expected_chunk = test_data[i * chunk_size:(i + 1) * chunk_size]
        np.testing.assert_array_equal(expected_chunk, host_buffer)
    
    # Cleanup - gpu_buffers are automatically freed when they go out of scope!
    for buf_handle in buf_handles:
        buf_handle.close()
    
    file_handle.close()
    driver_handle.close()
    
    if os.path.exists(filename):
        os.unlink(filename)


if __name__ == "__main__":
    main()
