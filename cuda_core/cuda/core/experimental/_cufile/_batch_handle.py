from typing import List, Optional, Dict, Any
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
        return self.status == 0 and self.error is None
    
    def has_error(self) -> bool:
        """Check if operation has an error"""
        return self.error is not None or self.status != 0
    
    def __repr__(self):
        if self.is_complete():
            return f"BatchIOResult(complete, {self.result} bytes)"
        elif self.has_error():
            return f"BatchIOResult(error: {self.error})"
        else:
            return f"BatchIOResult(status={self.status})"


def make_read_operation(file_handle, buffer, size, file_offset=0, 
                       buffer_offset=0, cookie=None) -> BatchIOParams:
    """
    Helper function to create a read operation.
    
    Args:
        file_handle: cuFile file handle
        buffer: Device buffer pointer (int) or buffer-like object
        size: Size of the read operation in bytes
        file_offset: Offset in file (default: 0)
        buffer_offset: Offset in buffer (default: 0)
        cookie: Optional user data to associate with this operation
        
    Returns:
        BatchIOParams configured for a read operation
    """
    # If buffer is an object with a pointer, extract it
    if hasattr(buffer, 'ptr'):
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
        buffer: Device buffer pointer (int) or buffer-like object
        size: Size of the write operation in bytes
        file_offset: Offset in file (default: 0)
        buffer_offset: Offset in buffer (default: 0)
        cookie: Optional user data to associate with this operation
        
    Returns:
        BatchIOParams configured for a write operation
    """
    # If buffer is an object with a pointer, extract it
    if hasattr(buffer, 'ptr'):
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
        return False  # Don't suppress exceptions

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
        """
        Submit a batch of I/O operations.
        
        Args:
            operations: List of BatchIOParams describing the operations
            flags: Submission flags (default: 0)
            
        Raises:
            RuntimeError: If batch handle is not initialized or submission fails
        """
        if self._handle is None:
            raise RuntimeError("Batch handle not initialized")
        
        if len(operations) > self.max_operations:
            raise ValueError(f"Number of operations ({len(operations)}) exceeds max ({self.max_operations})")
        
        # Store operations for later retrieval in get_status
        self._pending_operations = operations
        
        # Note: Full cuFile batch submission requires creating CUfileIOParams_t C structures
        # For now, we store the operations and they can be executed individually
        # Future: Convert BatchIOParams to CUfileIOParams_t and call:
        # cufile.batch_io_submit(self._handle, len(operations), iocbp_ptr, flags)
        print(f"BatchHandle: {len(operations)} operations queued for submission")
    
    def execute_operations(self) -> List[BatchIOResult]:
        """
        Execute the pending operations (fallback when C structures not available).
        This performs the operations individually but returns results in batch format.
        
        Returns:
            List of BatchIOResult objects with status of operations
        """
        if not self._pending_operations:
            return []
        
        results = []
        for i, op in enumerate(self._pending_operations):
            try:
                # Get the file handle's _handle attribute
                file_handle = op.file_handle
                
                # Perform the operation based on opcode
                if op.opcode == 0:  # Read
                    from cuda.bindings import cufile
                    bytes_transferred = cufile.read(
                        file_handle._handle,
                        op.buffer,
                        op.size,
                        op.file_offset,
                        op.buf_offset
                    )
                elif op.opcode == 1:  # Write
                    from cuda.bindings import cufile
                    bytes_transferred = cufile.write(
                        file_handle._handle,
                        op.buffer,
                        op.size,
                        op.file_offset,
                        op.buf_offset
                    )
                else:
                    raise ValueError(f"Unknown opcode: {op.opcode}")
                
                # Create success result
                results.append(BatchIOResult(
                    status=0,
                    result=bytes_transferred,
                    error=None,
                    cookie=op.cookie
                ))
            except Exception as e:
                # Create error result
                results.append(BatchIOResult(
                    status=-1,
                    result=0,
                    error=str(e),
                    cookie=op.cookie
                ))
        
        return results

    def get_status(self, min_completed: int = 1, timeout_ms: Optional[int] = None) -> List[BatchIOResult]:
        """
        Get status of submitted batch operations.
        
        Args:
            min_completed: Minimum number of completed operations to wait for (default: 1)
            timeout_ms: Optional timeout in milliseconds
            
        Returns:
            List of BatchIOResult objects with status of operations
            
        Raises:
            RuntimeError: If batch handle is not initialized
        """
        if self._handle is None:
            raise RuntimeError("Batch handle not initialized")
        
        # TODO: Need to create appropriate structures for nr, iocbp, and timeout
        # This is a placeholder showing the interface
        # For now, return an empty list to show the API
        
        # Future implementation would be:
        # nr_completed = ctypes.c_uint(0)
        # events = create_io_events_array(self.max_operations)
        # timeout_spec = create_timespec_ms(timeout_ms) if timeout_ms else None
        # cufile.batch_io_get_status(self._handle, min_completed, 
        #                            ctypes.addressof(nr_completed),
        #                            events, timeout_spec)
        # 
        # results = []
        # for i in range(nr_completed.value):
        #     results.append(BatchIOResult(
        #         status=events[i].status,
        #         result=events[i].result,
        #         error=events[i].error if events[i].error else None,
        #         cookie=events[i].cookie
        #     ))
        # return results
        
        raise NotImplementedError(
            "Batch get_status requires creating result structures. "
            "This will be implemented when the IOEvents structure is available."
        )
        
        return []  # Placeholder

    def cancel(self):
        """
        Cancel all pending batch operations.
        
        Raises:
            RuntimeError: If batch handle is not initialized
        """
        if self._handle is None:
            raise RuntimeError("Batch handle not initialized")
        
        cufile.batch_io_cancel(self._handle)

