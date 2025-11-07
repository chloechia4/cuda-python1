#!/usr/bin/env python3

import os
import sys
import tempfile

# Add paths for CUDA bindings
sys.path.insert(0, '/home/rladmin/cchia/cuda-python/cuda_bindings')
sys.path.insert(0, '/home/rladmin/cchia/cuda-python/cuda_core')

import cuda.bindings.driver as cuda
from cuda.bindings import cufile
from cuda.core.experimental._cufile._file_handle import FileHandle
from cuda.core.experimental._cufile._buffer_handle import BufferHandle


def test_file_handle_context_manager():
    """Test FileHandle with context manager syntax"""

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        test_file = tmp.name

    try:
        with FileHandle(use_direct=False, file_path=test_file) as fh:
            print(f"✓ FileHandle opened successfully: {fh._handle}")
            print(f"✓ File descriptor: {fh._fd}")

            # Store references to verify cleanup
            handle_before = fh._handle
            fd_before = fh._fd

            assert handle_before is not None, "Handle should be registered"
            assert fd_before is not None, "File descriptor should be open"

        assert fh._handle is None, "Handle should be deregistered after context exit"
        assert fh._fd is None, "File descriptor should be closed after context exit"

        print("✓ FileHandle context manager exited cleanly")
        print("✓ Handle cleanup verification passed")

    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.unlink(test_file)


def test_buffer_handle_context_manager():
    """Test BufferHandle with context manager syntax"""

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

    try:
        # Allocate CUDA memory
        buf_size = 4096
        err, buf_ptr = cuda.cuMemAlloc(buf_size)
        assert err == cuda.CUresult.CUDA_SUCCESS
        buf_ptr_int = int(buf_ptr)

        try:
            # Test context manager syntax
            with BufferHandle(buf_ptr_int, buf_size, 0) as bh:
                print(f"✓ BufferHandle opened successfully")
                print(f"✓ Buffer pointer: {bh.buffer}")
                print(f"✓ Buffer size: {bh._buffer_size}")

                # Verify buffer is registered
                assert bh._registered == True, "Buffer should be registered during context"
                buffer_before = bh.buffer
                assert buffer_before is not None, "Buffer pointer should be valid"

            # Verify cleanup happened after context manager exit
            assert bh._registered == False, "Buffer should be deregistered after context exit"

            print("✓ BufferHandle context manager exited cleanly")
            print("✓ Buffer cleanup verification passed")

        finally:
            # Free CUDA memory
            cuda.cuMemFree(buf_ptr)

    finally:
        # Cleanup CUDA
        cufile.driver_close()
        cuda.cuDevicePrimaryCtxRelease(device)


def main():
    """Run all context manager tests"""
    print("="*60)
    print("Testing Context Managers for FileHandle and BufferHandle")
    print("="*60)

    try:
        test_file_handle_context_manager()
        print()
        test_buffer_handle_context_manager()

        print("\n" + "="*60)
        print("✓ All context manager tests passed!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()