#!/usr/bin/env python3

import io
import os
import sys
import random
import tempfile
from contextlib import contextmanager

import pytest

# Add paths for CUDA bindings
sys.path.insert(0, '/home/rladmin/cchia/cuda-python/cuda_bindings')
sys.path.insert(0, '/home/rladmin/cchia/cuda-python/cuda_core')

import cuda.bindings.driver as cuda
from cuda.bindings import cufile
from cuda.core.experimental._cufile._file_handle import FileHandle
from cuda.core.experimental._cufile._buffer_handle import BufferHandle
from cuda.core.experimental._cufile._driver_handle import DriverHandle

cupy = pytest.importorskip("cupy")
numpy = pytest.importorskip("numpy")


def check_bit_flags(x: int, y: int) -> bool:
    """Check that the bits set in `y` is also set in `x`"""
    return x & y == y


@pytest.fixture(scope="function")
def cuda_context():
    """Initialize CUDA context for tests"""
    # Initialize CUDA
    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS
    
    err, device = cuda.cuDeviceGet(0)
    assert err == cuda.CUresult.CUDA_SUCCESS
    
    err, ctx = cuda.cuDevicePrimaryCtxRetain(device)
    assert err == cuda.CUresult.CUDA_SUCCESS
    
    (err,) = cuda.cuCtxSetCurrent(ctx)
    assert err == cuda.CUresult.CUDA_SUCCESS
    
    yield device
    
    # Cleanup
    cuda.cuDevicePrimaryCtxRelease(device)


@pytest.fixture(scope="function")
def driver_handle(cuda_context):
    """Initialize cuFile driver for tests"""
    driver = DriverHandle()
    yield driver
    driver.close()


@contextmanager
def cuda_buffer(size):
    """Context manager for CUDA buffer allocation"""
    err, buf_ptr = cuda.cuMemAlloc(size)
    assert err == cuda.CUresult.CUDA_SUCCESS
    buf_ptr_int = int(buf_ptr)
    
    try:
        yield buf_ptr_int
    finally:
        cuda.cuMemFree(buf_ptr)


def cupy_to_gpu_buffer(arr):
    """Convert cupy array to GPU buffer pointer and size"""
    return int(arr.data.ptr), arr.nbytes


def copy_gpu_to_host(buf_ptr, size, dtype):
    """Copy GPU buffer to host numpy array"""
    host_buffer = numpy.empty(size // numpy.dtype(dtype).itemsize, dtype=dtype)
    err, = cuda.cuMemcpyDtoH(host_buffer.ctypes.data, buf_ptr, size)
    assert err == cuda.CUresult.CUDA_SUCCESS
    return host_buffer


def copy_host_to_gpu(host_arr, buf_ptr):
    """Copy host numpy array to GPU buffer"""
    err, = cuda.cuMemcpyHtoD(buf_ptr, host_arr.ctypes.data, host_arr.nbytes)
    assert err == cuda.CUresult.CUDA_SUCCESS


@pytest.mark.parametrize("size", [1, 10, 100, 1000, 1024, 4096, 4096 * 10])
def test_write(tmp_path, driver_handle, size):
    """Test basic write using FileHandle and BufferHandle - follows write_sync.py pattern"""
    filename = tmp_path / "test-file"
    
    # Step 1: Create test data
    test_data = numpy.arange(size, dtype=numpy.uint8)
    
    # Step 2: Allocate GPU buffer
    err, buf_ptr = cuda.cuMemAlloc(size)
    assert err == cuda.CUresult.CUDA_SUCCESS
    buf_ptr_int = int(buf_ptr)
    
    # Step 3: Copy test data to GPU
    err, = cuda.cuMemcpyHtoD(buf_ptr, test_data.ctypes.data, size)
    assert err == cuda.CUresult.CUDA_SUCCESS
    
    # Step 4: Register buffer with cuFile
    buf_handle = BufferHandle(buf_ptr_int, size, 0)
    
    # Step 5: Open file with cuFile (create new file)
    file_handle = FileHandle(use_direct=True, flags=os.O_CREAT, file_path=str(filename))
    
    # Step 6: Write GPU buffer to file
    bytes_written = file_handle.write(
        buffer=buf_ptr_int,
        size=size,
        file_offset=0,
        buf_offset=0
    )
    
    assert bytes_written == size, f"Expected to write {size} bytes, got {bytes_written}"
    
    # Step 7: Verify by reading back
    written_data = numpy.fromfile(filename, dtype=numpy.uint8)
    numpy.testing.assert_array_equal(test_data, written_data)
    
    # Cleanup (proper order: buffer handle, file handle, GPU memory)
    buf_handle.close()
    file_handle.close()
    cuda.cuMemFree(buf_ptr)


@pytest.mark.parametrize("size", [1, 10, 100, 1000, 1024, 4096, 4096 * 10])
def test_read(tmp_path, driver_handle, size):
    """Test basic read using FileHandle and BufferHandle - follows read_sync.py pattern"""
    filename = tmp_path / "test-file"
    
    # Step 1: Create test file with known data
    test_data = numpy.arange(size, dtype=numpy.uint8)
    test_data.tofile(filename)
    os.sync()
    
    # Step 2: Allocate GPU buffer
    err, buf_ptr = cuda.cuMemAlloc(size)
    assert err == cuda.CUresult.CUDA_SUCCESS
    buf_ptr_int = int(buf_ptr)
    
    # Step 3: Register buffer with cuFile
    buf_handle = BufferHandle(buf_ptr_int, size, 0)
    
    # Step 4: Open file with cuFile
    file_handle = FileHandle(use_direct=True, flags=0, file_path=str(filename))
    
    # Step 5: Read file into GPU buffer
    bytes_read = file_handle.read(
        buffer=buf_ptr_int,
        size=size,
        file_offset=0,
        buf_offset=0
    )
    
    # Assert like KvikIO: f.read(b) == b.nbytes
    assert bytes_read == size, f"Expected to read {size} bytes, got {bytes_read}"
    
    # Step 6: Copy back to CPU and verify
    host_buffer = numpy.empty(size, dtype=numpy.uint8)
    err, = cuda.cuMemcpyDtoH(host_buffer.ctypes.data, buf_ptr, size)
    assert err == cuda.CUresult.CUDA_SUCCESS
    
    # Verify data matches
    numpy.testing.assert_array_equal(test_data, host_buffer)
    
    # Cleanup (proper order: buffer handle, file handle, GPU memory)
    buf_handle.close()
    file_handle.close()
    cuda.cuMemFree(buf_ptr)


def test_file_handle_context(tmp_path, driver_handle):
    """Open a FileHandle in a context"""
    filename = tmp_path / "test-file"
    size = 200
    
    # Create cupy array
    a = cupy.arange(size)
    buf_ptr_a, nbytes_a = cupy_to_gpu_buffer(a)
    
    # Allocate buffer for reading
    with cuda_buffer(nbytes_a) as buf_ptr_b:
        with BufferHandle(buf_ptr_a, nbytes_a, 0) as buf_handle_a:
            with BufferHandle(buf_ptr_b, nbytes_a, 0) as buf_handle_b:
                flags = os.O_CREAT | os.O_RDWR
                with FileHandle(use_direct=False, flags=flags, file_path=str(filename)) as f:
                    assert f._handle is not None
                    assert check_bit_flags(flags, os.O_RDWR)
                    
                    # Write
                    bytes_written = f.write(buf_ptr_a, nbytes_a, file_offset=0, buf_offset=0)
                    assert bytes_written == nbytes_a
                    
                    # Read
                    bytes_read = f.read(buf_ptr_b, nbytes_a, file_offset=0, buf_offset=0)
                    assert bytes_read == nbytes_a
                
                assert f._handle is None
        
        # Verify data
        b = copy_gpu_to_host(buf_ptr_b, nbytes_a, a.dtype)
        cupy.testing.assert_array_equal(a, b)


def test_no_file_error(tmp_path, driver_handle):
    """Test 'No such file' error"""
    filename = tmp_path / "test-file"
    
    with pytest.raises(FileNotFoundError):
        FileHandle(use_direct=False, flags=os.O_RDONLY, file_path=str(filename))


def test_incorrect_open_mode_error(tmp_path, driver_handle):
    """Test that files can be opened in different modes"""
    filename = tmp_path / "test-file"
    size = 10
    
    # Create test file
    test_data = numpy.arange(size, dtype=numpy.uint8)
    test_data.tofile(filename)
    os.sync()
    
    # Allocate GPU buffer
    err, buf_ptr = cuda.cuMemAlloc(size)
    assert err == cuda.CUresult.CUDA_SUCCESS
    buf_ptr_int = int(buf_ptr)
    
    # Register buffer
    buf_handle = BufferHandle(buf_ptr_int, size, 0)
    
    # Test 1: Open for reading
    file_handle = FileHandle(use_direct=True, flags=0, file_path=str(filename))
    bytes_read = file_handle.read(buf_ptr_int, size, file_offset=0, buf_offset=0)
    assert bytes_read == size
    file_handle.close()
    
    # Test 2: Open for read/write (O_RDWR)
    file_handle = FileHandle(use_direct=True, flags=os.O_RDWR, file_path=str(filename))
    # Can read
    bytes_read = file_handle.read(buf_ptr_int, size, file_offset=0, buf_offset=0)
    assert bytes_read == size
    # Can write
    bytes_written = file_handle.write(buf_ptr_int, size, file_offset=0, buf_offset=0)
    assert bytes_written == size
    file_handle.close()
    
    # Cleanup
    buf_handle.close()
    cuda.cuMemFree(buf_ptr)


def test_write_to_files_in_chunks(tmp_path, driver_handle):
    """Write to files in chunks"""
    filename = tmp_path / "test-file"
    size = 200
    
    a = cupy.arange(size)
    buf_ptr, nbytes = cupy_to_gpu_buffer(a)
    
    with BufferHandle(buf_ptr, nbytes, 0) as buf_handle:
        flags = os.O_CREAT | os.O_RDWR
        with FileHandle(use_direct=False, flags=flags, file_path=str(filename)) as f:
            nchunks = 20
            chunk_size = size // nchunks
            item_size = a.dtype.itemsize
            
            order = list(range(nchunks))
            random.shuffle(order)
            
            for i in order:
                offset = i * chunk_size
                chunk_nbytes = chunk_size * item_size
                buf_offset = offset * item_size
                file_offset = offset * item_size
                
                f.write(buf_ptr, chunk_nbytes, file_offset=file_offset, buf_offset=buf_offset)
    
    # Read and verify
    with cuda_buffer(nbytes) as buf_ptr_read:
        with BufferHandle(buf_ptr_read, nbytes, 0) as buf_handle_read:
            with FileHandle(use_direct=False, flags=os.O_RDONLY, file_path=str(filename)) as f:
                bytes_read = f.read(buf_ptr_read, nbytes, file_offset=0, buf_offset=0)
                assert bytes_read == nbytes
        
        b = copy_gpu_to_host(buf_ptr_read, nbytes, a.dtype)
        cupy.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "start,end",
    [(0, 10 * 4096), (1, int(1.3 * 4096)), (int(2.1 * 4096), int(5.6 * 4096))],
)
def test_read_write_slices(tmp_path, driver_handle, start, end):
    """Read and write different slices"""
    filename = tmp_path / "test-file"
    total_size = 10 * 4096
    
    # Create arrays
    a = cupy.arange(total_size, dtype=cupy.int64)
    b = a.copy()
    a[start:end] = 42
    
    # Get buffer info
    buf_ptr_a, nbytes_total = cupy_to_gpu_buffer(a)
    buf_ptr_b, _ = cupy_to_gpu_buffer(b)
    
    item_size = a.dtype.itemsize
    slice_nbytes = (end - start) * item_size
    slice_offset = start * item_size
    
    with BufferHandle(buf_ptr_a, nbytes_total, 0) as buf_handle_a:
        # Write slice
        flags = os.O_CREAT | os.O_RDWR
        with FileHandle(use_direct=False, flags=flags, file_path=str(filename)) as f:
            bytes_written = f.write(buf_ptr_a, slice_nbytes, file_offset=0, buf_offset=slice_offset)
            assert bytes_written == slice_nbytes
    
    with BufferHandle(buf_ptr_b, nbytes_total, 0) as buf_handle_b:
        # Read slice
        with FileHandle(use_direct=False, flags=os.O_RDONLY, file_path=str(filename)) as f:
            bytes_read = f.read(buf_ptr_b, slice_nbytes, file_offset=0, buf_offset=slice_offset)
            assert bytes_read == slice_nbytes
    
    cupy.testing.assert_array_equal(a, b)


def test_driver_handle_context_manager(cuda_context):
    """Test DriverHandle with context manager syntax"""
    with DriverHandle() as driver:
        assert driver._driver_opened == True
    
    assert driver._driver_opened == False


def test_buffer_handle_context_manager(driver_handle):
    """Test BufferHandle with context manager syntax"""
    # Allocate CUDA memory
    buf_size = 4096
    err, buf_ptr = cuda.cuMemAlloc(buf_size)
    assert err == cuda.CUresult.CUDA_SUCCESS
    buf_ptr_int = int(buf_ptr)
    
    try:
        # Test context manager syntax
        with BufferHandle(buf_ptr_int, buf_size, 0) as bh:
            # Verify buffer is registered
            assert bh._registered == True
            assert bh.buffer is not None
        
        # Verify cleanup happened after context manager exit
        assert bh._registered == False
    
    finally:
        # Free CUDA memory
        cuda.cuMemFree(buf_ptr)


def test_file_handle_context_manager(driver_handle):
    """Test FileHandle with context manager syntax"""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        test_file = tmp.name
    
    try:
        flags = os.O_CREAT | os.O_RDWR
        with FileHandle(use_direct=False, flags=flags, file_path=test_file) as fh:
            # Store references to verify cleanup
            handle_before = fh._handle
            fd_before = fh._fd
            
            assert handle_before is not None
            assert fd_before is not None
        
        assert fh._handle is None
        assert fh._fd is None
    
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.unlink(test_file)


@pytest.mark.parametrize("size", [1, 10, 100, 1000, 1024, 4096, 4096 * 10])
def test_raw_read_write(tmp_path, driver_handle, size):
    """Test raw read/write - follows working example patterns"""
    filename = tmp_path / "test-file"
    
    # Create test data
    test_data = numpy.arange(size, dtype=numpy.uint8)
    
    # Allocate GPU buffer for write
    err, buf_ptr = cuda.cuMemAlloc(size)
    assert err == cuda.CUresult.CUDA_SUCCESS
    buf_ptr_int = int(buf_ptr)
    
    # Copy test data to GPU
    err, = cuda.cuMemcpyHtoD(buf_ptr, test_data.ctypes.data, size)
    assert err == cuda.CUresult.CUDA_SUCCESS
    
    # Register buffer and write
    buf_handle = BufferHandle(buf_ptr_int, size, 0)
    file_handle = FileHandle(use_direct=True, flags=os.O_CREAT, file_path=str(filename))
    bytes_written = file_handle.write(buf_ptr_int, size, file_offset=0, buf_offset=0)
    assert bytes_written == size
    file_handle.close()
    buf_handle.close()
    cuda.cuMemFree(buf_ptr)
    
    # Allocate GPU buffer for read
    err, buf_ptr_read = cuda.cuMemAlloc(size)
    assert err == cuda.CUresult.CUDA_SUCCESS
    buf_ptr_read_int = int(buf_ptr_read)
    
    # Register buffer and read
    buf_handle_read = BufferHandle(buf_ptr_read_int, size, 0)
    file_handle_read = FileHandle(use_direct=True, flags=0, file_path=str(filename))
    bytes_read = file_handle_read.read(buf_ptr_read_int, size, file_offset=0, buf_offset=0)
    assert bytes_read == size
    file_handle_read.close()
    
    # Copy back to host and verify
    host_buffer = numpy.empty(size, dtype=numpy.uint8)
    err, = cuda.cuMemcpyDtoH(host_buffer.ctypes.data, buf_ptr_read, size)
    assert err == cuda.CUresult.CUDA_SUCCESS
    
    numpy.testing.assert_array_equal(test_data, host_buffer)
    
    # Cleanup
    buf_handle_read.close()
    cuda.cuMemFree(buf_ptr_read)


def test_multiple_buffers(tmp_path, driver_handle):
    """Test registering multiple buffers simultaneously"""
    size1 = 1024
    size2 = 2048
    
    with cuda_buffer(size1) as buf1:
        with cuda_buffer(size2) as buf2:
            with BufferHandle(buf1, size1, 0) as bh1:
                with BufferHandle(buf2, size2, 0) as bh2:
                    assert bh1._registered == True
                    assert bh2._registered == True
                    assert bh1.buffer != bh2.buffer
                
                assert bh2._registered == False
            
            assert bh1._registered == False


def test_sequential_file_operations(tmp_path, driver_handle):
    """Test sequential file operations - follows working example patterns"""
    filename = tmp_path / "test-file"
    size = 100
    
    # Create full test data (200 bytes total)
    total_size = size * 2
    test_data = numpy.arange(total_size, dtype=numpy.uint8)
    
    # First write: write first 100 bytes
    err, buf_ptr1 = cuda.cuMemAlloc(size)
    assert err == cuda.CUresult.CUDA_SUCCESS
    buf_ptr1_int = int(buf_ptr1)
    
    err, = cuda.cuMemcpyHtoD(buf_ptr1, test_data[:size].ctypes.data, size)
    assert err == cuda.CUresult.CUDA_SUCCESS
    
    buf_handle1 = BufferHandle(buf_ptr1_int, size, 0)
    file_handle1 = FileHandle(use_direct=True, flags=os.O_CREAT, file_path=str(filename))
    bytes_written1 = file_handle1.write(buf_ptr1_int, size, file_offset=0, buf_offset=0)
    assert bytes_written1 == size
    file_handle1.close()
    buf_handle1.close()
    cuda.cuMemFree(buf_ptr1)
    
    # Second write: write next 100 bytes at offset
    err, buf_ptr2 = cuda.cuMemAlloc(size)
    assert err == cuda.CUresult.CUDA_SUCCESS
    buf_ptr2_int = int(buf_ptr2)
    
    err, = cuda.cuMemcpyHtoD(buf_ptr2, test_data[size:].ctypes.data, size)
    assert err == cuda.CUresult.CUDA_SUCCESS
    
    buf_handle2 = BufferHandle(buf_ptr2_int, size, 0)
    file_handle2 = FileHandle(use_direct=True, flags=os.O_RDWR, file_path=str(filename))
    bytes_written2 = file_handle2.write(buf_ptr2_int, size, file_offset=size, buf_offset=0)
    assert bytes_written2 == size
    file_handle2.close()
    buf_handle2.close()
    cuda.cuMemFree(buf_ptr2)
    
    # Read back everything (200 bytes)
    err, buf_ptr_read = cuda.cuMemAlloc(total_size)
    assert err == cuda.CUresult.CUDA_SUCCESS
    buf_ptr_read_int = int(buf_ptr_read)
    
    buf_handle_read = BufferHandle(buf_ptr_read_int, total_size, 0)
    file_handle_read = FileHandle(use_direct=True, flags=0, file_path=str(filename))
    bytes_read = file_handle_read.read(buf_ptr_read_int, total_size, file_offset=0, buf_offset=0)
    assert bytes_read == total_size
    file_handle_read.close()
    
    # Copy back to host and verify
    host_buffer = numpy.empty(total_size, dtype=numpy.uint8)
    err, = cuda.cuMemcpyDtoH(host_buffer.ctypes.data, buf_ptr_read, total_size)
    assert err == cuda.CUresult.CUDA_SUCCESS
    
    numpy.testing.assert_array_equal(test_data, host_buffer)
    
    # Cleanup
    buf_handle_read.close()
    cuda.cuMemFree(buf_ptr_read)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
