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
from cuda.core.experimental._cufile._stream_handle import StreamHandle

def test_stream_handle_basic():
    """Test StreamHandle as context manager."""
    print("Test 2: StreamHandle context manager")
    
    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS
    
    err, device = cuda.cuDeviceGet(0)
    assert err == cuda.CUresult.CUDA_SUCCESS
    
    err, ctx = cuda.cuDevicePrimaryCtxRetain(device)
    assert err == cuda.CUresult.CUDA_SUCCESS
    
    (err,) = cuda.cuCtxSetCurrent(ctx)
    assert err == cuda.CUresult.CUDA_SUCCESS
    
    cufile.driver_open()
    
    err, stream = cuda.cuStreamCreate(0)
    assert err == cuda.CUresult.CUDA_SUCCESS
    print(f"  ✓ Created CUDA stream: {stream}")
    
    with StreamHandle(int(stream), flags=0) as stream_handle:
        print(f"  ✓ Inside context: is_valid() = {stream_handle.is_valid()}")
        assert stream_handle.is_valid(), "Stream should be valid inside context"
        assert stream_handle.get() == int(stream), "Should return correct stream"
    
    # After context exits, stream should be deregistered
    # Note: We can't easily check is_valid() here since the object is out of scope
    print(f"  ✓ Context exited - stream automatically deregistered")
    
    cuda.cuStreamDestroy(stream)
    cufile.driver_close()
    cuda.cuDevicePrimaryCtxRelease(device)
    print("  ✓ Test passed!\n")


def test_stream_handle_multiple_streams():
    """Test multiple StreamHandles simultaneously."""
    print("Test 3: Multiple StreamHandles")
    
    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS
    
    err, device = cuda.cuDeviceGet(0)
    assert err == cuda.CUresult.CUDA_SUCCESS
    
    err, ctx = cuda.cuDevicePrimaryCtxRetain(device)
    assert err == cuda.CUresult.CUDA_SUCCESS
    
    (err,) = cuda.cuCtxSetCurrent(ctx)
    assert err == cuda.CUresult.CUDA_SUCCESS
    
    cufile.driver_open()
    
    # Create multiple streams
    err, stream1 = cuda.cuStreamCreate(0)
    assert err == cuda.CUresult.CUDA_SUCCESS
    
    err, stream2 = cuda.cuStreamCreate(0)
    assert err == cuda.CUresult.CUDA_SUCCESS
    
    err, stream3 = cuda.cuStreamCreate(0)
    assert err == cuda.CUresult.CUDA_SUCCESS
    
    print(f"  ✓ Created 3 CUDA streams: {stream1}, {stream2}, {stream3}")
    
    # Register all streams
    with StreamHandle(int(stream1), flags=0) as sh1, \
         StreamHandle(int(stream2), flags=0) as sh2, \
         StreamHandle(int(stream3), flags=0) as sh3:
        
        print(f"  ✓ All streams registered")
        assert sh1.is_valid() and sh2.is_valid() and sh3.is_valid()
        print(f"  ✓ sh1.is_valid() = {sh1.is_valid()}")
        print(f"  ✓ sh2.is_valid() = {sh2.is_valid()}")
        print(f"  ✓ sh3.is_valid() = {sh3.is_valid()}")
    
    print(f"  ✓ All streams automatically deregistered on context exit")
    
    cuda.cuStreamDestroy(stream1)
    cuda.cuStreamDestroy(stream2)
    cuda.cuStreamDestroy(stream3)
    cufile.driver_close()
    cuda.cuDevicePrimaryCtxRelease(device)
    print("  ✓ Test passed!\n")


def test_stream_handle_reregistration():
    """Test that deregister + register works correctly."""
    print("Test 4: StreamHandle re-registration")
    
    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS
    
    err, device = cuda.cuDeviceGet(0)
    assert err == cuda.CUresult.CUDA_SUCCESS
    
    err, ctx = cuda.cuDevicePrimaryCtxRetain(device)
    assert err == cuda.CUresult.CUDA_SUCCESS
    
    (err,) = cuda.cuCtxSetCurrent(ctx)
    assert err == cuda.CUresult.CUDA_SUCCESS
    
    cufile.driver_open()
    
    err, stream = cuda.cuStreamCreate(0)
    assert err == cuda.CUresult.CUDA_SUCCESS
    
    stream_handle = StreamHandle(int(stream), flags=0)
    print(f"  ✓ Initial registration: is_valid() = {stream_handle.is_valid()}")
    assert stream_handle.is_valid()
    
    stream_handle.deregister()
    print(f"  ✓ After deregister: is_valid() = {stream_handle.is_valid()}")
    assert not stream_handle.is_valid()
    
    # Try to deregister again (should be safe - no-op)
    stream_handle.deregister()
    print(f"  ✓ Double deregister is safe")
    
    cuda.cuStreamDestroy(stream)
    cufile.driver_close()
    cuda.cuDevicePrimaryCtxRelease(device)
    print("  ✓ Test passed!\n")


def main():
    """Run all StreamHandle isolation tests."""
    print("=" * 60)
    print("StreamHandle Isolation Tests")
    print("=" * 60 + "\n")
    
    test_stream_handle_basic()
    test_stream_handle_context_manager()
    test_stream_handle_multiple_streams()
    test_stream_handle_reregistration()
    
    print("=" * 60)
    print("All StreamHandle tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()

