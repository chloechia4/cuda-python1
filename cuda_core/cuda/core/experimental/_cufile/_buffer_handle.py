class BufferHandle:
    def __init__(self, buffer, size, flags=0):
        self.buffer = buffer
        self.flags = flags
        self._buffer_size = size

        self._register_buffer()

    def __del__(self):
        """Calls buf_deregister"""
        self._deregister_buffer()

    def _register_buffer(self):
        """Takes in a buffer and registers it with cuFile using integer pointer"""
        cufile.buf_register(self.buffer, self._buffer_size, self.flags)

    def _deregister_buffer(self):
        """Calls buf_deregister using integer pointer"""
        if self.buffer is not None:
            cufile.buf_deregister(self.buffer)