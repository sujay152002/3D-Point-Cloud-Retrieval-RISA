"""GPU memory management utilities to prevent OOM errors during sequential task execution."""

import gc
import torch


def cleanup_cuda_memory(verbose: bool = False) -> dict:
    """Aggressively clean up GPU memory.
    
    Returns dict with memory stats before/after cleanup.
    """
    if not torch.cuda.is_available():
        return {}
    
    stats = {}
    if verbose:
        stats['before_mb'] = torch.cuda.memory_allocated() / 1024**2
    
    # Garbage collection
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    if verbose:
        stats['after_mb'] = torch.cuda.memory_allocated() / 1024**2
        stats['freed_mb'] = stats['before_mb'] - stats['after_mb']
    
    return stats


def memory_summary(device: torch.device) -> dict:
    """Get current GPU memory usage summary."""
    if not torch.cuda.is_available():
        return {}
    
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    
    return {
        'allocated_gb': allocated,
        'reserved_gb': reserved,
        'total_gb': total,
        'free_gb': total - reserved,
        'utilization_pct': 100 * reserved / total
    }


def maybe_cleanup_after_batch(step: int, cleanup_interval: int = 10, verbose: bool = False) -> None:
    """Optionally clean up GPU memory every N batches."""
    if step % cleanup_interval == 0 and step > 0:
        cleanup_cuda_memory(verbose=verbose)


def clear_model_gradients(model: torch.nn.Module) -> None:
    """Explicitly set model gradients to None (more efficient than zero_grad in some cases)."""
    for param in model.parameters():
        if param.grad is not None:
            param.grad.detach_()
            param.grad = None
