from cuda.bindings import cufile


class StreamHandle:
    """Context manager for cuFile stream registration"""
    
    def __init__(self, stream, flags=0):
        """
        Register a CUDA stream with cuFile.
        
        Args:
            stream: CUDA stream (CUstream) as an integer pointer
            flags: Registration flags (default: 0)
        """
        if isinstance(stream, int):
            self._stream = stream
        else:
            self._stream = int(stream)
        
        self._flags = flags
        self._registered = False
        self._register_stream()
    
    def __enter__(self):
        """Context manager entry - returns self for use in 'with' statement"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures proper cleanup"""
        self.deregister()
        return False
    
    def __del__(self):
        """Cleanup on object destruction"""
        self.deregister()
    
    def _register_stream(self):
        """Register the stream with cuFile"""
        if not self._registered:
            cufile.stream_register(self._stream, self._flags)
            self._registered = True
    
    def deregister(self):
        """Explicitly deregister the stream handle"""
        if self._registered:
            try:
                cufile.stream_deregister(self._stream)
            except:
                pass
            self._registered = False
    
    def get(self):
        """Return the registered stream"""
        return self._stream
    
    def is_valid(self):
        """Returns whether the stream handle owns a valid registered CUDA stream resource"""
        return self._registered

