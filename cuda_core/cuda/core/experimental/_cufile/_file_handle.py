class FileHandle:
    def __init__(self, use_direct: bool = False, flags=0, file_path=None): 
        self.use_direct = use_direct
        self.flags = flags
        self.file_path = file_path
        self._handle = None
        self._fd = None

        open_flags = os.O_RDWR
        if use_direct:
            open_flags |= os.O_DIRECT
        open_flags |= flags
        self._fd = os.open(file_path, open_flags)
    
        self._register_handle()

    def __del__(self):
        """This will call deregisterHandle"""
        self._deregister_handle()
        if self._fd is not None:
            with suppress(OSError):
                os.close(self._fd)

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