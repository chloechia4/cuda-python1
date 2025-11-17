from typing import List, Optional, Dict, Any
import ctypes
from cuda.bindings import cufile


class BatchIOParams:
    """Parameters for a single batch I/O operation"""
    def __init__(self, file_handle, buffer, size, file_offset=0, buf_offset=0, 
                 opcode=0, cookie=None):
        """
        Initialize batch I/O parameters.
        
        Args:
            file_handle: cuFile file handle
            buffer: Device buffer pointer (int)
            size: Size of the operation in bytes
            file_offset: Offset in file (default: 0)
            buf_offset: Offset in buffer (default: 0)
            opcode: Operation code (0=read, 1=write)
            cookie: Optional user data to associate with this operation
        """
        self.file_handle = file_handle
        self.buffer = buffer
        self.size = size
        self.file_offset = file_offset
        self.buf_offset = buf_offset
        self.opcode = opcode
        self.cookie = cookie or {}

class BatchIOResult:
    """Result of a batch I/O operation"""
    def __init__(self, status: int, result: int, error: Optional[str] = None, cookie=None):
        """
        Initialize batch I/O result.
        
        Args:
            status: Operation status code
            result: Number of bytes transferred or error code
            error: Optional error message
            cookie: User data associated with this operation
        """
        self.status = status
        self.result = result
        self.error = error
        self.cookie = cookie or {}
    
    def is_complete(self) -> bool:
        """Check if operation completed successfully"""
            return self.status == cufile.Status.COMPLETE

    
    def is_failed(self) -> bool:
        """Check if operation failed"""
            return self.status == cufile.Status.FAILED

    
    def has_error(self) -> bool:
        """Check if operation has an error"""
        return self.error is not None or self.status < 0
    

def make_read_operation(file_handle, buffer, size, file_offset=0, 
                       buffer_offset=0, cookie=None) -> BatchIOParams:
    """
    Helper function to create a read operation.
    
    Args:
        file_handle: cuFile file handle
        buffer: Device buffer pointer (int), cuda.core Buffer, or buffer-like object
        size: Size of the read operation in bytes
        file_offset: Offset in file (default: 0)
        buffer_offset: Offset in buffer (default: 0)
        cookie: Optional user data to associate with this operation
        
    Returns:
        BatchIOParams configured for a read operation
    """
    if hasattr(buffer, 'handle'):
        buffer = int(buffer.handle)
    elif hasattr(buffer, 'ptr'):
        buffer = int(buffer.ptr)
    elif not isinstance(buffer, int):
        buffer = int(buffer)
    
    return BatchIOParams(
        file_handle=file_handle,
        buffer=buffer,
        size=size,
        file_offset=file_offset,
        buf_offset=buffer_offset,
        opcode=0,  # 0 = read
        cookie=cookie
    )

def make_write_operation(file_handle, buffer, size, file_offset=0, 
                        buffer_offset=0, cookie=None) -> BatchIOParams:
    """
    Helper function to create a write operation.
    
    Args:
        file_handle: cuFile file handle
        buffer: Device buffer pointer (int), cuda.core Buffer, or buffer-like object
        size: Size of the write operation in bytes
        file_offset: Offset in file (default: 0)
        buffer_offset: Offset in buffer (default: 0)
        cookie: Optional user data to associate with this operation
        
    Returns:
        BatchIOParams configured for a write operation
    """
    if hasattr(buffer, 'handle'):
        buffer = int(buffer.handle)
    elif hasattr(buffer, 'ptr'):
        buffer = int(buffer.ptr)
    elif not isinstance(buffer, int):
        buffer = int(buffer)
    
    return BatchIOParams(
        file_handle=file_handle,
        buffer=buffer,
        size=size,
        file_offset=file_offset,
        buf_offset=buffer_offset,
        opcode=1,  # 1 = write
        cookie=cookie
    )

class BatchHandle:
    """Context manager for batch I/O operations"""

    def __init__(self, max_operations: int):
        """
        Initialize batch handle for cuFile batch operations.
        
        Args:
            max_operations: Maximum number of operations that can be submitted in batch
        """
        self.max_operations = max_operations
        self._handle = None
        self._pending_operations = []
        self._setup_batch()

    def __enter__(self):
        """Context manager entry - returns self for use in 'with' statement"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures proper cleanup"""
        self.close()
        return False  

    def close(self):
        """Explicitly close the batch handle and cleanup"""
        self._deregister_batch()

    def __del__(self):
        """Cleanup on object destruction"""
        self.close()

    def _setup_batch(self):
        """Setup batch I/O handle"""
        if self._handle is None:
            # Returns the batch handle as intptr_t
            self._handle = cufile.batch_io_set_up(self.max_operations)

    def _deregister_batch(self):
        """Explicitly deregister and destroy the batch handle"""
        if self._handle is not None:
            try:
                cufile.batch_io_destroy(self._handle)
            except:
                # Ignore errors during shutdown - driver may already be closing
                pass
            self._handle = None

    def submit(self, operations: List[BatchIOParams], flags: int = 0):
       
        if self._handle is None:
            raise RuntimeError("Batch handle not initialized")

        if len(operations) > self.max_operations:
            raise ValueError(f"Number of operations ({len(operations)}) exceeds max ({self.max_operations})")

        # Store operations for later retrieval in get_status
        self._pending_operations = operations

        # Create CUfileIOParams_t array
        io_params = cufile.IOParams(len(operations))

        # Convert BatchIOParams to CUfileIOParams_t structures
        for i, op in enumerate(operations):
            io_params[i].mode = cufile.BatchMode.BATCH
            io_params[i].fh = op.file_handle
            io_params[i].opcode = cufile.Opcode.READ if op.opcode == 0 else cufile.Opcode.WRITE
            io_params[i].cookie = id(op.cookie) if op.cookie else 0
            io_params[i].u.batch.dev_ptr_base = op.buffer
            io_params[i].u.batch.file_offset = op.file_offset
            io_params[i].u.batch.dev_ptr_offset = op.buf_offset
            io_params[i].u.batch.size_ = op.size

        # Submit batch operations to the driver
        cufile.batch_io_submit(self._handle, len(operations), io_params.ptr, flags)

    def get_status(self, min_completed: int = 1, max_events: Optional[int] = None,
                   timeout_ms: Optional[int] = None) -> List[BatchIOResult]:
        """
        Get status of submitted batch operations.

        Args:
            min_completed: Minimum number of completed operations to wait for (default: 1)
            max_events: Maximum number of events to retrieve (default: max_operations)
            timeout_ms: Optional timeout in milliseconds (None = wait indefinitely)

        Returns:
            List of BatchIOResult objects with status of operations

        Raises:
            RuntimeError: If batch handle is not initialized
        """
        if self._handle is None:
            raise RuntimeError("Batch handle not initialized")

        # Determine max events to retrieve
        if max_events is None:
            max_events = self.max_operations

        # Create the nr (number of events) parameter - this is input/output
        nr = ctypes.c_uint(max_events)

        # Create the events array to receive results
        events = cufile.IOEvents(max_events)

        # Create timeout structure if specified
        timeout_ptr = 0
        if timeout_ms is not None and timeout_ms > 0:
            # Create timespec structure: convert milliseconds to sec + nsec
            timeout = ctypes.create_string_buffer(16)  # sizeof(timespec) = 16 bytes
            timeout_struct = ctypes.cast(timeout, ctypes.POINTER(ctypes.c_long * 2))
            timeout_struct.contents[0] = timeout_ms // 1000  # tv_sec
            timeout_struct.contents[1] = (timeout_ms % 1000) * 1_000_000  # tv_nsec
            timeout_ptr = ctypes.addressof(timeout)

        # Call the lower-level batch_io_get_status function
        cufile.batch_io_get_status(
            self._handle,
            min_completed,
            ctypes.addressof(nr),
            events.ptr,
            timeout_ptr
        )

        # Extract results from events array
        results = []
        num_events = nr.value

        for i in range(num_events):
            event = events[i]
            cookie = None

            # Try to match cookie back to original operation
            cookie_id = event.cookie
            for op in self._pending_operations:
                if id(op.cookie) == cookie_id:
                    cookie = op.cookie
                    break

            results.append(BatchIOResult(
                status=event.status,
                result=event.ret,
                cookie=cookie
            ))

        return results

    def cancel(self):
        """
        Cancel all pending batch operations.
        
        Raises:
            RuntimeError: If batch handle is not initialized
        """
        if self._handle is None:
            raise RuntimeError("Batch handle not initialized")
        
        cufile.batch_io_cancel(self._handle)

