"""
Device Utilities

Functions for detecting and managing compute devices (CPU/GPU).
"""

import torch


def get_device(prefer_gpu: bool = True, device_id: int = 0) -> torch.device:
    """
    Get the appropriate compute device

    Args:
        prefer_gpu: Whether to prefer GPU if available
        device_id: GPU device ID to use

    Returns:
        torch.device object
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device(f'cuda:{device_id}')
    else:
        device = torch.device('cpu')

    return device


def check_gpu_availability(verbose: bool = True) -> bool:
    """
    Check if GPU is available

    Args:
        verbose: Whether to print detailed information

    Returns:
        True if GPU is available, False otherwise
    """
    cuda_available = torch.cuda.is_available()

    if verbose:
        print("GPU Availability Check")
        print("=" * 50)

        if cuda_available:
            print(f"✓ CUDA is available")
            print(f"  GPU Count: {torch.cuda.device_count()}")
            print(f"  Current Device: {torch.cuda.current_device()}")
            print(f"  Device Name: {torch.cuda.get_device_name(0)}")

            # Get GPU memory info
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory / (1024 ** 3)  # Convert to GB
                print(f"\n  GPU {i}: {props.name}")
                print(f"    Total Memory: {total_memory:.2f} GB")
                print(f"    Compute Capability: {props.major}.{props.minor}")
        else:
            print("✗ CUDA is NOT available")
            print("  Training and inference will use CPU")
            print("\n  To enable GPU:")
            print("  1. Ensure you have an NVIDIA GPU")
            print("  2. Install CUDA Toolkit")
            print("  3. Install PyTorch with CUDA support")
            print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")

        print("=" * 50)

    return cuda_available


def get_device_info() -> dict:
    """
    Get detailed device information

    Returns:
        Dictionary with device information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available():
        info['devices'] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info['devices'].append({
                'id': i,
                'name': props.name,
                'total_memory_gb': props.total_memory / (1024 ** 3),
                'compute_capability': f"{props.major}.{props.minor}",
                'multi_processor_count': props.multi_processor_count
            })

    return info


def set_device(device_id: int = 0):
    """
    Set the current CUDA device

    Args:
        device_id: GPU device ID to use
    """
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        print(f"Set current device to GPU {device_id}")
    else:
        print("CUDA not available, using CPU")


def clear_gpu_cache():
    """Clear GPU cache to free up memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared")
    else:
        print("No GPU available to clear cache")


if __name__ == '__main__':
    # Test device utilities
    print("\nTesting Device Utilities\n")

    # Check GPU availability
    has_gpu = check_gpu_availability(verbose=True)

    # Get device info
    info = get_device_info()
    print("\nDevice Information:")
    print(f"  CUDA Available: {info['cuda_available']}")
    if info['cuda_available']:
        print(f"  CUDA Version: {info['cuda_version']}")
        print(f"  Device Count: {info['device_count']}")

    # Get default device
    device = get_device()
    print(f"\nDefault Device: {device}")

    # Test tensor creation
    print("\nTesting tensor creation on device...")
    try:
        x = torch.randn(3, 3).to(device)
        print(f"  ✓ Created tensor on {device}")
        print(f"  Tensor shape: {x.shape}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
