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
    
    # Create test data in host memory
    test_data = np.arange(total_size, dtype=np.uint8)
    
    # Initialize CUDA using cuda.core Device
    dev = Device(0)
    dev.set_current()
    
    driver_handle = DriverHandle()
    
    gpu_buffers = []
    buf_handles = []
    
    # Allocate GPU buffers using cuda.core memory resource and copy test data to GPU
    for i in range(num_operations):
        gpu_buffer = dev.allocate(chunk_size)
        gpu_buffers.append(gpu_buffer)
        
        # Copy test data chunk to GPU buffer
        chunk_data = test_data[i * chunk_size:(i + 1) * chunk_size]
        err, = cuda.cuMemcpyHtoD(int(gpu_buffer.handle), chunk_data.ctypes.data, chunk_size)
        assert err == cuda.CUresult.CUDA_SUCCESS
        
        buf_handle = BufferHandle(gpu_buffer, chunk_size, 0)
        buf_handles.append(buf_handle)
        
    # Create empty file for writing
    with open(filename, 'wb') as f:
        f.write(b'\x00' * total_size)
    
    file_handle = FileHandle(use_direct=True, flags=0, file_path=filename)
    
    # ===== PHASE 1: Batch WRITE operations =====
    with BatchHandle(max_operations=num_operations) as batch:
        
        write_operations = []
        for i in range(num_operations):
            file_offset = i * chunk_size
            cookie = {"operation_id": i, "offset": file_offset, "type": "write"}
            
            op = make_write_operation(
                file_handle=file_handle,
                buffer=gpu_buffers[i],
                size=chunk_size,
                file_offset=file_offset,
                buffer_offset=0,
                cookie=cookie
            )
            write_operations.append(op)
        
        batch.submit(write_operations, flags=0)
        
        # Get status of submitted operations
        results = batch.get_status(min_completed=num_operations, timeout_ms=5000)
        
        # Check results
        for i, result in enumerate(results):
            if result.is_complete():
                pass
            elif result.has_error():
                pass
            else:
                pass
        
        assert all(r.is_complete() for r in results), "Not all write operations completed successfully"
    
    # ===== PHASE 2: Batch READ operations =====
    # Allocate new GPU buffers for reading
    gpu_read_buffers = []
    buf_read_handles = []
    
    for i in range(num_operations):
        gpu_read_buffer = dev.allocate(chunk_size)
        gpu_read_buffers.append(gpu_read_buffer)
        
        buf_read_handle = BufferHandle(gpu_read_buffer, chunk_size, 0)
        buf_read_handles.append(buf_read_handle)
    
    with BatchHandle(max_operations=num_operations) as batch:
        
        read_operations = []
        for i in range(num_operations):
            file_offset = i * chunk_size
            cookie = {"operation_id": i, "offset": file_offset, "type": "read"}
            
            op = make_read_operation(
                file_handle=file_handle,
                buffer=gpu_read_buffers[i],
                size=chunk_size,
                file_offset=file_offset,
                buffer_offset=0,
                cookie=cookie
            )
            read_operations.append(op)
        
        batch.submit(read_operations, flags=0)
        
        # Get status of submitted operations
        results = batch.get_status(min_completed=num_operations, timeout_ms=5000)
        
        # Check results
        for i, result in enumerate(results):
            if result.is_complete():
                pass
            elif result.has_error():
                pass
            else:
                pass
        
        assert all(r.is_complete() for r in results), "Not all read operations completed successfully"
    
    # Verify data by copying read buffers back to host and comparing
    for i in range(num_operations):
        host_buffer = np.empty(chunk_size, dtype=np.uint8)
        err, = cuda.cuMemcpyDtoH(host_buffer.ctypes.data, int(gpu_read_buffers[i].handle), chunk_size)
        assert err == cuda.CUresult.CUDA_SUCCESS
        
        expected_chunk = test_data[i * chunk_size:(i + 1) * chunk_size]
        np.testing.assert_array_equal(expected_chunk, host_buffer)
    
    # Cleanup read buffers
    for buf_read_handle in buf_read_handles:
        buf_read_handle.close()
    
    # Cleanup - gpu_buffers are automatically freed when they go out of scope!
    for buf_handle in buf_handles:
        buf_handle.close()
    
    file_handle.close()
    driver_handle.close()
    
    if os.path.exists(filename):
        os.unlink(filename)


if __name__ == "__main__":
    main()
