import os

class FileHandle:
    def __init__(self, use_direct: bool = False, flags=0, file_path=None): # change to file path only 
        self.use_direct = use_direct
        self.flags = flags
        self.file_path = file_path
        self._handle = None
        self._fd = None

        # File path
        open_flags = os.O_RDWR
        if use_direct:
            open_flags |= os.O_DIRECT
        open_flags |= flags
        self._fd = os.open(file_path, open_flags)
    
        self._register_handle()

    def __enter__(self):
        """Context manager entry - returns self for use in 'with' statement"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures proper cleanup"""
        self.close()
        return False  # Don't suppress exceptions

    def close(self):
        """Explicitly close the file handle and deregister"""
        self._deregister_handle()
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __del__(self):
        """This will call deregisterHandle"""
        self.close()

    def _register_handle(self):
        """This will return an error if it fails"""
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

    def read(self, buffer, size: int, file_offset: int = 0, buf_offset: int = 0):
        # Handle both integer pointers and buffers with ctypes.data
        if isinstance(buffer, int):
            buffer_ptr = buffer
        else:
            buffer_ptr = buffer.ctypes.data
        
        return cufile.read(self._handle, buffer_ptr, size, file_offset, buf_offset)

    def write(self, buffer, size: int, file_offset: int = 0, buf_offset: int = 0):
        # Handle both integer pointers and buffers with ctypes.data
        if isinstance(buffer, int):
            buffer_ptr = buffer
        else:
            buffer_ptr = buffer.ctypes.data
        
        return cufile.write(self._handle, buffer_ptr, size, file_offset, buf_offset)