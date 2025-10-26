import os
import sys
import ctypes
import fcntl
from contextlib import suppress

# Add paths for standalone execution
sys.path.insert(0, '/home/rladmin/cchia/cuda-python/cuda_bindings')

from cuda.bindings import cufile
import cuda.bindings.driver as cuda


class FileHandle:
    def __init__(self, use_direct: bool = False, flags=0, file_path_or_fd=None):
        self.use_direct = use_direct
        self.flags = flags
        self.file_path_or_fd = file_path_or_fd
        self._handle = None
        self._fd = None

        if isinstance(file_path_or_fd, str):
            # File path
            open_flags = os.O_RDWR
            if use_direct:
                open_flags |= os.O_DIRECT
            open_flags |= flags
            self._fd = os.open(file_path_or_fd, open_flags)
        else:
            # File descriptor
            self._fd = file_path_or_fd
        # assert that the cufileerror doesnt come back 
        self._register_handle()

    def __del__(self):
        """This will call deregisterHandle"""
        self._deregister_handle()
        if self._fd is not None:
            with suppress(OSError):
                os.close(self._fd)

    def verify_direct_io(self):
        """Verify if O_DIRECT flag is set on the file descriptor"""
        if self._fd is None:
            return False
        
        try:
            flags = fcntl.fcntl(self._fd, fcntl.F_GETFL)
            # O_DIRECT = 0x4000 on Linux (from asm-generic/fcntl.h)
            has_direct = bool(flags & os.O_DIRECT)
            
            print(f"File descriptor flags: {flags:#x}")
            print(f"O_DIRECT flag value: {os.O_DIRECT:#x}")
            print(f"O_DIRECT enabled: {has_direct}")
            print(f"use_direct setting: {self.use_direct}")
            
            return has_direct
        except Exception as e:
            print(f"Error checking flags: {e}")
            return False

    def _register_handle(self):
        """This will return an error if it fails"""
        descr = cufile.Descr()
        descr.type = cufile.FileHandleType.OPAQUE_FD
        descr.handle.fd = self._fd
        descr.fs_ops = 0

        self._handle = cufile.handle_register(descr.ptr)

    def _deregister_handle(self):
        """This will be an explicit way to deregisterHandle"""
        if self._handle is not None:
            try:
                cufile.handle_deregister(self._handle)
            except:
                # Ignore errors during shutdown - driver may already be closing
                pass
            self._handle = None

    def read(self, buffer, size: int = None, file_offset: int = 0, buf_offset: int = 0):
        if size is None:
            size = len(buffer)
        return cufile.read(self._handle, buffer.ctypes.data, size, file_offset, buf_offset)

    def write(self, buffer, size: int = None, file_offset: int = 0, buf_offset: int = 0):
        if size is None:
            size = len(buffer)
        return cufile.write(self._handle, buffer.ctypes.data, size, file_offset, buf_offset)


class BufferHandle:
    def __init__(self, buffer, size, flags=0):
        """
        Initialize buffer handle for cuFile operations.

        Args:
            buffer: Raw pointer (CUdeviceptr or int) from cuMemAlloc
            size: Buffer size in bytes (required)
            flags: Registration flags for cuFile
        """
        print(f"BufferHandle.__init__: Starting initialization...")
        print(f"  buffer={buffer}, size={size}, flags={flags}")

        self.buffer = buffer
        self.flags = flags
        self._buffer_size = size

        # Convert buffer pointer to integer (from cuMemAlloc)

        print(f"  About to call _register_buffer()...")
        self._register_buffer()
        print(f"  BufferHandle initialization complete!")

    def __del__(self):
        """Calls buf_deregister"""
        self._deregister_buffer()

    def _register_buffer(self):
        """Takes in a buffer and registers it with cuFile using integer pointer"""
        cufile.buf_register(self.buffer, self._buffer_size, self.flags)
        print(f"  _register_buffer: cufile.buf_register() completed successfully!")

    def _deregister_buffer(self):
        """Calls buf_deregister using integer pointer"""
        if self.buffer is not None:
            try:
                cufile.buf_deregister(self.buffer)
                print(f"  _deregister_buffer: Successfully deregistered buffer")
            except Exception as e:
                print(f"  _deregister_buffer: Error during deregistration: {e}")
                # Ignore errors during shutdown - driver may already be closing
                pass