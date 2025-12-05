#!/usr/bin/env python3

import os
import sys
import time
import ctypes  # Needed for cuFile operations
import cupy as cp

# Add paths for CUDA bindings
sys.path.insert(0, '/home/rladmin/cchia/cuda-python/cuda_bindings')

import cuda.bindings.driver as cuda
from cuda.bindings import cufile

from cuda.core.experimental._cufile._buffer_handle import BufferHandle
from cuda.core.experimental._cufile._file_handle import FileHandle
from cuda.core.experimental._cufile._io_params import IOParams

import kvikio  # kvikIO
import kvikio.defaults

kvikio.defaults.set("gds_threshold", 0)
kvikio.defaults.set("compat_mode", kvikio.CompatMode.OFF)

def is_cufile_available():
    """Check if cuFile is available on the system."""
    try:
        version = cufile.get_version()
        print(f"cuFile version: {version}")
        return True
    except Exception as e:
        print(f"cuFile not available: {e}")
        return False


def is_kvikio_available():
    """Check if kvikIO is available and GPUDirect Storage is enabled."""

    try:
        # Force GDS mode (disable compatibility mode)
        kvikio.defaults.set("compat_mode", kvikio.CompatMode.OFF)

        # # Check if kvikIO is using GDS (not in compat mode)
        compat_mode = kvikio.defaults.get("compat_mode")
        
        if compat_mode == 0:
            print("GPUDirect Storage (GDS) is fully enabled")
            print(f"  - kvikIO compat_mode: {compat_mode} (OFF)")
            print(f"  - kvikIO version: {kvikio.__version__}")
            return True
        else:
            print("kvikIO is in compatibility mode")
            print(f"   compat_mode = {compat_mode}")
            print("   libcufile.so is not being used - falling back to POSIX I/O")
            return False
    except Exception as e:
        print(f"kvikIO not available: {e}")
        return False


def benchmark_cufile_async_operation(operation, buf_size, duration_seconds):
    """
    Benchmark cuFile ASYNC read or write operation.

    Args:
        operation: 'read' or 'write'
        buf_size: Buffer size in bytes
        duration_seconds: How long to run the test in seconds

    Returns:
        tuple: (throughput_gibs, avg_latency_ms, iterations_completed)
    """
    # Write to NVMe device that supports GDS
    file_path = f"/mnt/nvme0/test_cufile_async_{operation}_{buf_size}.bin"

    try:
        # Allocate CUDA memory
        err, buf_ptr = cuda.cuMemAlloc(buf_size)
        assert err == cuda.CUresult.CUDA_SUCCESS
        buf_ptr_int = int(buf_ptr)

        # Register buffer with cuFile using BufferHandle class
        buf_handle = BufferHandle(buf_ptr_int, buf_size, 0)

        # Create CUDA stream for async operations
        err, stream = cuda.cuStreamCreate(0)
        assert err == cuda.CUresult.CUDA_SUCCESS
        stream_int = int(stream)  # Convert CUstream to integer for cufile

        if operation == 'write':
            # Prepare test data
            test_string = b"Hello cuFile! This is test data for read/write operations. "
            test_string_len = len(test_string)
            repetitions = (buf_size // test_string_len) + 1
            test_data = (test_string * repetitions)[:buf_size]
            host_buf = ctypes.create_string_buffer(buf_size)
            ctypes.memmove(host_buf, test_data, buf_size)

            # Copy test data to GPU
            (err,) = cuda.cuMemcpyHtoDAsync(buf_ptr, host_buf, buf_size, stream_int)
            assert err == cuda.CUresult.CUDA_SUCCESS
            (err,) = cuda.cuStreamSynchronize(stream)
            assert err == cuda.CUresult.CUDA_SUCCESS

            fd = os.open(file_path, os.O_CREAT | os.O_RDWR | os.O_DIRECT, 0o600)
            file_handle = FileHandle(use_direct=True, flags=0, file_path=file_path)    

            # Create IOParams for async operations
            io_params = IOParams(
                buffer=buf_ptr_int,
                size=buf_size,
                file_offset=0,
                buffer_offset=0
            )

            # Benchmark ASYNC write operations for specified duration
            # Submit ALL operations first, then sync ONCE at the end
            start_time = time.time()
            iterations = 0

            while (time.time() - start_time) < duration_seconds:
                file_handle.write_async(io_params, stream=stream_int)
                iterations += 1

            # Sync only ONCE after all submissions
            (err,) = cuda.cuStreamSynchronize(stream)
            assert err == cuda.CUresult.CUDA_SUCCESS
            end_time = time.time()
            

        else:  # read
            # Pre-write test file
            test_string = b"Hello cuFile! This is test data for read/write operations. "
            test_string_len = len(test_string)
            repetitions = (buf_size // test_string_len) + 1
            test_data = (test_string * repetitions)[:buf_size]

            with open(file_path, 'wb') as f:
                f.write(test_data)

            fd = os.open(file_path, os.O_RDONLY | os.O_DIRECT)
            file_handle = FileHandle(use_direct=True, flags=0, file_path=file_path)
            
            # Create IOParams for async operations
            io_params = IOParams(
                buffer=buf_ptr_int,
                size=buf_size,
                file_offset=0,
                buffer_offset=0
            )
            
            # Benchmark ASYNC read operations for specified duration
            # Submit ALL operations first, then sync ONCE at the end
            start_time = time.time()
            iterations = 0

            while (time.time() - start_time) < duration_seconds:
                file_handle.read_async(io_params, stream=stream_int)
                iterations += 1

            # Sync only ONCE after all submissions
            (err,) = cuda.cuStreamSynchronize(stream)
            assert err == cuda.CUresult.CUDA_SUCCESS
            end_time = time.time()
        
        # Calculate metrics
        elapsed_time = end_time - start_time
        total_bytes = buf_size * iterations
        total_gib = total_bytes / (1024**3)
        throughput_gibs = total_gib / elapsed_time
        avg_latency_ms = (elapsed_time / iterations) * 1000

        # Cleanup
        cuda.cuStreamDestroy(stream)
        cuda.cuMemFree(buf_ptr)

        return throughput_gibs, avg_latency_ms, iterations

    finally:
        if os.path.exists(file_path):
            os.unlink(file_path)


def benchmark_kvikio_async_operation(operation, buf_size, duration_seconds):
    """
    Benchmark kvikIO ASYNC read or write operation.

    Args:
        operation: 'read' or 'write'
        buf_size: Buffer size in bytes
        duration_seconds: How long to run the test in seconds

    Returns:
        tuple: (throughput_gibs, avg_latency_ms, iterations_completed)
    """
    # Write to NVMe device that supports GDS
    file_path = f"/mnt/nvme0/test_kvikio_async_{operation}_{buf_size}.bin"

    try:
        # Create GPU buffer using CuPy (matching sync version pattern)
        write_buf = cp.arange(buf_size // 4, dtype=cp.uint32)  # 4 bytes per uint32
        read_buf = cp.zeros_like(write_buf)

        # Create a CuPy stream for async operations
        stream = cp.cuda.Stream()

        if operation == 'write':
            # Open file once before timing
            f = kvikio.CuFile(file_path, "w")

            
            # Benchmark ASYNC write operations for specified duration
            # Submit ALL operations first, then sync ONCE at the end
            start_time = time.time()
            iterations = 0
            futures = []

            while (time.time() - start_time) < duration_seconds:
                future = f.raw_write_async(write_buf, stream=stream.ptr, size=write_buf.nbytes, file_offset=0, dev_offset=0)  # ASYNC version
                futures.append(future)  # Keep future alive
                iterations += 1

            # Sync only ONCE after all submissions
            stream.synchronize()
            end_time = time.time()
            
            
            # Close file after timing
            f.close()

        else:  # read
            # Pre-write test file
            f_write = kvikio.CuFile(file_path, "w")
            f_write.raw_write(write_buf, size=write_buf.nbytes, file_offset=0, dev_offset=0)
            f_write.close()

            # Open file once before timing
            f = kvikio.CuFile(file_path, "r")
            
            
            # Benchmark ASYNC read operations for specified duration
            # Submit ALL operations first, then sync ONCE at the end
            start_time = time.time()
            iterations = 0
            futures = []

            while (time.time() - start_time) < duration_seconds:
                future = f.raw_read_async(read_buf, stream=stream.ptr, size=read_buf.nbytes, file_offset=0, dev_offset=0)  # ASYNC version
                futures.append(future)  # Keep future alive
                iterations += 1

            # Sync only ONCE after all submissions
            stream.synchronize()
            end_time = time.time()
            
            # Close file after timing
            f.close()

        # Calculate metrics
        elapsed_time = end_time - start_time
        total_bytes = buf_size * iterations
        total_gib = total_bytes / (1024**3)
        throughput_gibs = total_gib / elapsed_time
        avg_latency_ms = (elapsed_time / iterations) * 1000

        return throughput_gibs, avg_latency_ms, iterations

    finally:
        if os.path.exists(file_path):
            os.unlink(file_path)


def run_comprehensive_benchmark():
    """Run comprehensive ASYNC benchmark across different file sizes."""

    # Check availability first
    if not is_cufile_available():
        raise RuntimeError("cuFILE not available")

    if not is_kvikio_available():
        print(f"is_kvikio_available(): {is_kvikio_available()}")
        print(f"kvikio.defaults.compat_mode(): {kvikio.defaults.compat_mode()}")
        raise RuntimeError("kvikIO GDS not available")

    # Initialize CUDA
    print("Initializing CUDA...")
    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, device = cuda.cuDeviceGet(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    err, ctx = cuda.cuDevicePrimaryCtxRetain(device)
    assert err == cuda.CUresult.CUDA_SUCCESS

    (err,) = cuda.cuCtxSetCurrent(ctx)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Configuration
    file_sizes = {
        '4K': 4 * 1024,
        '8K': 8 * 1024,
        '64K': 64 * 1024,
        '512K': 512 * 1024,
        '1M': 1 * 1024 * 1024,
        '4M': 4 * 1024 * 1024,
        '16M': 16 * 1024 * 1024,
    }
    duration_seconds = 10  # Run each test for 10 seconds

    print(f"\nConfiguration:")
    print(f"  Duration per test: {duration_seconds} seconds")
    print(f"  File sizes: {', '.join(file_sizes.keys())}")
    print(f"  Operations: ASYNC - cuFile (write_async/read_async) & kvikIO (raw_write_async/raw_read_async)")
    print(f"  Metrics: Bandwidth (GiB/s) and Average Latency (ms)")
    print("="*70 + "\n")

    # Storage for results
    results = {}

    try:
        # Run benchmarks for each file size
        for size_name, buf_size in file_sizes.items():
            print(f"Benchmarking {size_name} ({buf_size:,} bytes)...")
            results[size_name] = {}

            # cuFile async operations
            print(f"  cuFile WRITE (async)...", end='', flush=True)
            cf_write_bw, cf_write_lat, cf_write_iters = benchmark_cufile_async_operation('write', buf_size, duration_seconds)
            results[size_name]['cufile_write_bw'] = cf_write_bw
            results[size_name]['cufile_write_lat'] = cf_write_lat
            results[size_name]['cufile_write_iters'] = cf_write_iters
            print(f" {cf_write_bw:.3f} GiB/s, {cf_write_lat:.3f} ms ({cf_write_iters:,} ops)")

            print(f"  cuFile READ (async)...", end='', flush=True)
            cf_read_bw, cf_read_lat, cf_read_iters = benchmark_cufile_async_operation('read', buf_size, duration_seconds)
            results[size_name]['cufile_read_bw'] = cf_read_bw
            results[size_name]['cufile_read_lat'] = cf_read_lat
            results[size_name]['cufile_read_iters'] = cf_read_iters
            print(f" {cf_read_bw:.3f} GiB/s, {cf_read_lat:.3f} ms ({cf_read_iters:,} ops)")

            # kvikIO async operations 
            print(f"  kvikIO WRITE (async)...", end='', flush=True)
            kv_write_bw, kv_write_lat, kv_write_iters = benchmark_kvikio_async_operation('write', buf_size, duration_seconds)
            results[size_name]['kvikio_write_bw'] = kv_write_bw
            results[size_name]['kvikio_write_lat'] = kv_write_lat
            results[size_name]['kvikio_write_iters'] = kv_write_iters
            print(f" {kv_write_bw:.3f} GiB/s, {kv_write_lat:.3f} ms ({kv_write_iters:,} ops)")
        
            # kvikIO READ
            print(f"  kvikIO READ (async)...", end='', flush=True)
            kv_read_bw, kv_read_lat, kv_read_iters = benchmark_kvikio_async_operation('read', buf_size, duration_seconds)
            results[size_name]['kvikio_read_bw'] = kv_read_bw
            results[size_name]['kvikio_read_lat'] = kv_read_lat
            results[size_name]['kvikio_read_iters'] = kv_read_iters
            print(f" {kv_read_bw:.3f} GiB/s, {kv_read_lat:.3f} ms ({kv_read_iters:,} ops)")

        # Display results
        print_results_tables(results, file_sizes, duration_seconds)

    finally:
        # Cleanup
        cuda.cuDevicePrimaryCtxRelease(device)


def print_results_tables(results, file_sizes, duration_seconds):
    """Print formatted results tables - ASYNC operations."""

    print("\n\n" + "=" * 90)
    print("ASYNC BANDWIDTH (GiB/s)")
    print("=" * 90)
    print(f"{'Size':<8} {'cuFile WRITE':<20} {'cuFile READ':<20} {'kvikIO WRITE':<20} {'kvikIO READ':<20}")
    print("-" * 90)

    for size_name in file_sizes.keys():
        cf_write = results[size_name]['cufile_write_bw']
        cf_read = results[size_name]['cufile_read_bw']
        kv_write = results[size_name]['kvikio_write_bw']
        kv_read = results[size_name]['kvikio_read_bw']
        print(f"{size_name:<8} {cf_write:<20.6f} {cf_read:<20.6f} {kv_write:<20.6f} {kv_read:<20.6f}")

    print("=" * 90)

    print("\n\n" + "=" * 90)
    print("ASYNC AVERAGE LATENCY (ms per operation)")
    print("=" * 90)
    print(f"{'Size':<8} {'cuFile WRITE':<20} {'cuFile READ':<20} {'kvikIO WRITE':<20} {'kvikIO READ':<20}")
    print("-" * 90)

    for size_name in file_sizes.keys():
        cf_write = results[size_name]['cufile_write_lat']
        cf_read = results[size_name]['cufile_read_lat']
        kv_write = results[size_name]['kvikio_write_lat']
        kv_read = results[size_name]['kvikio_read_lat']
        print(f"{size_name:<8} {cf_write:<20.6f} {cf_read:<20.6f} {kv_write:<20.6f} {kv_read:<20.6f}")

    print("=" * 90)
    
    print("\n\n" + "=" * 100)
    print("PERFORMANCE RATIOS (kvikIO / cuFile) - ASYNC")
    print("=" * 100)
    print(f"{'Size':<8} {'WRITE BW Ratio':<20} {'READ BW Ratio':<20} {'WRITE LAT Ratio':<20} {'READ LAT Ratio':<20}")
    print("-" * 100)

    for size_name in file_sizes.keys():
        cf_write_bw = results[size_name]['cufile_write_bw']
        cf_read_bw = results[size_name]['cufile_read_bw']
        kv_write_bw = results[size_name]['kvikio_write_bw']
        kv_read_bw = results[size_name]['kvikio_read_bw']

        cf_write_lat = results[size_name]['cufile_write_lat']
        cf_read_lat = results[size_name]['cufile_read_lat']
        kv_write_lat = results[size_name]['kvikio_write_lat']
        kv_read_lat = results[size_name]['kvikio_read_lat']

        write_bw_ratio = kv_write_bw / cf_write_bw if cf_write_bw > 0 else 0
        read_bw_ratio = kv_read_bw / cf_read_bw if cf_read_bw > 0 else 0
        write_lat_ratio = kv_write_lat / cf_write_lat if cf_write_lat > 0 else 0
        read_lat_ratio = kv_read_lat / cf_read_lat if cf_read_lat > 0 else 0

        print(f"{size_name:<8} {write_bw_ratio:<20.3f} {read_bw_ratio:<20.3f} {write_lat_ratio:<20.3f} {read_lat_ratio:<20.3f}")

    print("=" * 100)
    print("\nNote: Ratio > 1.0 means kvikIO is better for bandwidth, worse for latency")
    print("      Ratio < 1.0 means cuFile is better for bandwidth, kvikIO is better for latency")
    print("=" * 100)


if __name__ == "__main__":
    run_comprehensive_benchmark()
    print("\n✓ ASYNC Benchmark completed successfully!")