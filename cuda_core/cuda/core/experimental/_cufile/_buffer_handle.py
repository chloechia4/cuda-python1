from cuda.bindings import cufile

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
        self._registered = False

        # Convert buffer pointer to integer (from cuMemAlloc)

        print(f"  About to call _register_buffer()...")
        self._register_buffer()
        print(f"  BufferHandle initialization complete!")

    def __enter__(self):
        """Context manager entry - returns self for use in 'with' statement"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures proper cleanup"""
        self.close()
        return False  # Don't suppress exceptions

    def close(self):
        """Explicitly close the buffer handle and deregister"""
        self._deregister_buffer()

    def __del__(self):
        """Calls buf_deregister"""
        self.close()

    def _register_buffer(self):
        """Takes in a buffer and registers it with cuFile using integer pointer"""
        if not self._registered:
            cufile.buf_register(self.buffer, self._buffer_size, self.flags)
            self._registered = True
            print(f"  _register_buffer: cufile.buf_register() completed successfully!")

    def _deregister_buffer(self):
        """Calls buf_deregister using integer pointer"""
        if self.buffer is not None and self._registered:
            try:
                cufile.buf_deregister(self.buffer)
                self._registered = False
                print(f"  _deregister_buffer: Successfully deregistered buffer")
            except Exception as e:
                print(f"  _deregister_buffer: Error during deregistration: {e}")
                # Ignore errors during shutdown - driver may already be closing
                pass