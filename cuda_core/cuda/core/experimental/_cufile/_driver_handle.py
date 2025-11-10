from cuda.bindings import cufile


class DriverHandle:
    def __init__(self, stream=None, flags=0):
        self._handle = None
        self._stream = stream
        self._flags = flags
        self._driver_opened = False
        
        # Open the cuFile driver
        self._driver_open()

    def __enter__(self):
        """Context manager entry - returns self for use in 'with' statement"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures proper cleanup"""
        self.close()
        return False  

    def close(self):
        """Explicitly close the cuFile driver"""
        if self._driver_opened:
            try:
                cufile.driver_close()
                self._driver_opened = False
            except:
                # Ignore errors during shutdown - driver may already be closing
                pass

    def __del__(self):
        """This will close driver on object destruction"""
        self.close()

    def _driver_open(self):
        """Initialize the cuFile library and open the nvidia-fs driver"""
        if not self._driver_opened:
            cufile.driver_open()
            self._driver_opened = True

