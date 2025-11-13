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
    make_write_operation,
    BatchIOResult
)
from cuda.core.experimental import Device


def main():
    """Example of batch I/O with mixed O_DIRECT and non-direct modes using cuda.core memory resource."""
    
    num_operations = 4
    chunk_size = 4096
    
    print("="*60)
    print("Batch I/O with Mixed File Opening Modes")
    print("="*60)
    
    # Initialize CUDA using cuda.core Device
    dev = Device(0)
    dev.set_current()
    
    driver_handle = DriverHandle()
    
    # Create buffers and file handles
    gpu_buffers = []
    buf_handles = []
    file_handles = []
    
    print(f"\nSetting up {num_operations} operations:")
    for i in range(num_operations):
        # Allocate GPU buffer using cuda.core memory resource
        gpu_buffer = dev.allocate(chunk_size)
        
        # Fill with test data
        test_value = 0xEF + i
        err, = cuda.cuMemsetD8(int(gpu_buffer.handle), test_value, chunk_size)
        assert err == cuda.CUresult.CUDA_SUCCESS
        
        gpu_buffers.append(gpu_buffer)
        
        # Register only even-indexed buffers (like the C++ example)
        if i % 2 == 0:
            buf_handle = BufferHandle(gpu_buffer, chunk_size, 0)
            buf_handles.append(buf_handle)
            print(f"  Operation {i}: Buffer REGISTERED, ", end="")
        else:
            buf_handles.append(None)
            print(f"  Operation {i}: Buffer NOT registered, ", end="")
        
        # Open files: even indices with O_DIRECT, odd indices without
        filename = f"test-batch-mixed-{i}.bin"
        if i % 2 == 0:
            # O_DIRECT mode
            file_handle = FileHandle(use_direct=True, flags=os.O_CREAT, file_path=filename)
            print("File opened with O_DIRECT")
        else:
            # Non-direct mode
            file_handle = FileHandle(use_direct=False, flags=os.O_CREAT, file_path=filename)
            print("File opened in NON-DIRECT mode")
        
        file_handles.append(file_handle)
    
    print("\n" + "="*60)
    print("Submitting Batch Write Operations")
    print("="*60)
    
    with BatchHandle(max_operations=num_operations) as batch:
        operations = []
        for i in range(num_operations):
            cookie = {
                "index": i,
                "mode": "O_DIRECT" if i % 2 == 0 else "NON_DIRECT",
                "registered": "YES" if i % 2 == 0 else "NO"
            }
            
            op = make_write_operation(
                file_handle=file_handles[i],
                buffer=gpu_buffers[i],
                size=chunk_size,
                file_offset=0,
                buffer_offset=0,
                cookie=cookie
            )
            operations.append(op)
        
        print(f"\nSubmitting {len(operations)} operations to batch...")
        batch.submit(operations)
        
        print("Executing batch operations...")
        results = batch.execute_operations()
        
        print("\n" + "="*60)
        print("Batch Operation Results:")
        print("="*60)
        for i, result in enumerate(results):
            cookie = result.cookie
            status_str = "✓ SUCCESS" if result.is_complete() else "✗ FAILED"
            print(f"  Op {cookie['index']}: {status_str} - {result.result} bytes")
            print(f"    Mode: {cookie['mode']}, Buffer Registered: {cookie['registered']}")
            if result.has_error():
                print(f"    Error: {result.error}")
        
        success_count = sum(1 for r in results if r.is_complete())
        print(f"\nCompleted: {success_count}/{len(results)} operations")
    
    print("\n" + "="*60)
    print("Verifying Written Data")
    print("="*60)
    
    # Verify the written data
    for i in range(num_operations):
        filename = f"test-batch-mixed-{i}.bin"
        if os.path.exists(filename):
            written_data = np.fromfile(filename, dtype=np.uint8)
            expected_value = 0xEF + i
            if len(written_data) == chunk_size and all(written_data == expected_value):
                print(f"  File {i}: ✓ Verified ({len(written_data)} bytes, value={hex(expected_value)})")
            else:
                print(f"  File {i}: ✗ Verification failed")
    
    print("\n" + "="*60)
    print("Cleanup")
    print("="*60)
    
    # Cleanup - gpu_buffers are automatically freed when they go out of scope!
    for i, buf_handle in enumerate(buf_handles):
        if buf_handle is not None:
            buf_handle.close()
            print(f"  Deregistered buffer {i}")
    
    for i, file_handle in enumerate(file_handles):
        file_handle.close()
    
    # Clean up test files
    for i in range(num_operations):
        filename = f"test-batch-mixed-{i}.bin"
        if os.path.exists(filename):
            os.unlink(filename)
    
    driver_handle.close()
    
    print("\n" + "="*60)
    print("Batch Mixed Mode Example Completed Successfully!")
    print("="*60)
    print("\nKey Takeaways:")
    print("  • Files can be opened with or without O_DIRECT")
    print("  • Buffers can be registered or unregistered")
    print("  • Batch operations work with mixed configurations")
    print("  • Even indices: O_DIRECT + registered buffers")
    print("  • Odd indices: Non-direct + unregistered buffers")


if __name__ == "__main__":
    main()

