import ctypes


class IOParams:
    def __init__(self, buffer, size, file_offset=0, buffer_offset=0):
        if hasattr(buffer, 'ptr'):
            self.buffer = int(buffer.ptr)
        elif not isinstance(buffer, int):
            self.buffer = int(buffer)
        else:
            self.buffer = buffer
        
        self.size = size
        self.file_offset = file_offset
        self.buffer_offset = buffer_offset
        
        self._size_ptr = ctypes.c_size_t(size)
        self._file_offset_ptr = ctypes.c_ssize_t(file_offset)
        self._buffer_offset_ptr = ctypes.c_ssize_t(buffer_offset)
        self._bytes_done_ptr = ctypes.c_ssize_t(0)
    
    def get_bytes_done(self) -> int:
        """Get the number of bytes transferred by the async operation."""
        return self._bytes_done_ptr.value
