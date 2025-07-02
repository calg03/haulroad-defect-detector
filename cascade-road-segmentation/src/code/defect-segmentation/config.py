#!/usr/bin/env python
"""
Configuration file for UNet++ road defect segmentation training.
Contains all hyperparameters, paths, and settings.
"""

import os
import torch
import time
import pynvml
pynvml.nvmlInit()
# -----------------------------
# Architecture Configuration
# -----------------------------
# Available architectures: 'unetplusplus', 'unet', 'deeplabv3', 'deeplabv3plus', 
# 'fpn', 'pspnet', 'transunet', 'swin_unet'
ARCHITECTURE = 'unetplusplus_eca' #this to test different architectures

# Legacy encoder settings (kept for UNet++ compatibility)
ENCODER = 'efficientnet-b5'
ENCODER_WEIGHTS = 'imagenet'

# -----------------------------
# Training Configuration (Architecture-Aware)
# -----------------------------
SEED = 42

# Fixed GPU selection - stick to one GPU like the original script
def get_best_device():
    """Select best available GPU and stick to it (no switching during training)"""
    import warnings
    
    # Suppress CUDA warnings during device detection
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        if not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            return 'cpu'
        
        device_count = torch.cuda.device_count()
        if device_count == 0:
            print("No CUDA devices found, using CPU")
            return 'cpu'
    
    print(f"Found {device_count} CUDA device(s)")
    
    # Check available GPUs and their memory
    best_device = None
    best_free_memory = 0
    
    for i in range(device_count):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                handle     = pynvml.nvmlDeviceGetHandleByIndex(i)
                info       = pynvml.nvmlDeviceGetMemoryInfo(handle)
                free_memory = info.free  # en bytes
                total_memory = info.total
            
            free_gb = free_memory / (1024**3)
            total_gb = total_memory / (1024**3)

            # Select GPU with most free memory (prioritize cuda:2, then cuda:0, then cuda:1)
            if i == 2 and free_gb > 15.0:  # Prefer cuda:2 if it has >15GB free
                best_device = i
                best_free_memory = free_memory
            elif i == 0 and free_gb > 15.0 and best_device != 2:  # Then cuda:0 if >15GB free
                best_device = i
                best_free_memory = free_memory
            elif free_memory > best_free_memory and best_device not in [0, 2]:  # Otherwise most free
                best_device = i
                best_free_memory = free_memory
                
        except Exception as e:
            # Only print actual errors, not CUDA warnings
            if "CUDA" not in str(e) or "warning" not in str(e).lower():
                print(f"Could not access GPU {i}: {e}")
    
    # Use the best GPU found
    if best_device is not None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch.cuda.set_device(best_device)
                selected_device = f'cuda:{best_device}'
                free_gb = best_free_memory / (1024**3)
                print(f"✅ Selected device: {selected_device} ({free_gb:.1f}GB free)")
                pynvml.nvmlShutdown()
                return selected_device
        except Exception as e:
            print(f"Failed to set device cuda:{best_device}: {e}")
    
    # Final fallback to cuda:0 or CPU
    try:
        torch.cuda.set_device(0)
        print("⚠️ Using cuda:0 as fallback")
        pynvml.nvmlShutdown()   
        return 'cuda:0'
    except:
        print("❌ All GPU options failed, falling back to CPU")
        pynvml.nvmlShutdown()
        return 'cpu'

DEVICE = get_best_device()  # Select best GPU and stick to it

# Get GPU memory for dynamic batch size optimization
try:
    if 'cuda' in DEVICE:
        gpu_memory_gb = torch.cuda.get_device_properties(DEVICE).total_memory / (1024**3)
    else:
        gpu_memory_gb = None
except:
    gpu_memory_gb = None

# Architecture-specific training parameters will override these defaults
EPOCHS = 150
IMG_SIZE = 512  # Match original script dimensions
EARLY_STOPPING_PATIENCE = 10

# Legacy parameters (for backward compatibility - will be overridden by architecture config)
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 6
LEARNING_RATE = 3e-4
MAX_LR = 1e-3
USE_AMP = True  # Automatic mixed precision
USE_FP16 = True
USE_ONECYCLE = True
LR_ENCODER_FACTOR = 0.1  # Encoder learning rate factor
WEIGHT_DECAY = 1e-5
WARMUP_EPOCHS = 5
MIN_DELTA = 1e-4  # Minimum improvement threshold

# Debug mode for fast testing
DEBUG_MODE = True  # Enable for quick architecture testing
DEBUG_SAMPLES_PER_DATASET = 10  # Very small for quick testing
DEBUG_EPOCHS = 2  # Quick epochs for testing


# Memory management
if torch.cuda.is_available():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.backends.cudnn.benchmark = False
    # Set memory fraction to avoid OOM on shared GPUs
    torch.cuda.set_per_process_memory_fraction(0.85)

# GPU memory management functions
def cleanup_gpu_memory():
    """Clean up GPU memory and switch device if needed"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Also try to collect garbage
        import gc
        gc.collect()

def get_gpu_memory_info(device_id):
    """Get detailed memory information for a specific GPU"""
    if not torch.cuda.is_available():
        return None
        
    try:
        torch.cuda.set_device(device_id)
        total_memory = torch.cuda.get_device_properties(device_id).total_memory
        allocated_memory = torch.cuda.memory_allocated(device_id)
        cached_memory = torch.cuda.memory_reserved(device_id)
        free_memory = total_memory - cached_memory
        
        return {
            'total': total_memory,
            'allocated': allocated_memory,
            'cached': cached_memory,
            'free': free_memory,
            'utilization': (cached_memory / total_memory) * 100
        }
    except Exception as e:
        print(f"Error getting memory info for GPU {device_id}: {e}")
        return None

def check_gpu_memory_and_switch(min_free_gb=2.0):
    """Check GPU memory and switch to another GPU if current one is full"""
    if not torch.cuda.is_available():
        return DEVICE
        
    current_device_id = torch.cuda.current_device()
    device_count = torch.cuda.device_count()
    
    print(f"Current device: cuda:{current_device_id}")
    
    # Check current device memory
    current_memory = get_gpu_memory_info(current_device_id)
    if current_memory:
        free_gb = current_memory['free'] / (1024**3)
        print(f"GPU {current_device_id} - Free: {free_gb:.1f}GB, Utilization: {current_memory['utilization']:.1f}%")
        
        # If current GPU has enough memory, stay with it
        if free_gb >= min_free_gb:
            return f'cuda:{current_device_id}'
    
    # Try to find a better GPU
    best_device = current_device_id
    max_free_memory = current_memory['free'] if current_memory else 0
    
    print("Checking all GPUs for available memory...")
    for i in range(device_count):
        if i != current_device_id:
            memory_info = get_gpu_memory_info(i)
            if memory_info:
                free_gb = memory_info['free'] / (1024**3)
                print(f"GPU {i} - Free: {free_gb:.1f}GB, Utilization: {memory_info['utilization']:.1f}%")
                
                if memory_info['free'] > max_free_memory and free_gb >= min_free_gb:
                    max_free_memory = memory_info['free']
                    best_device = i
    
    if best_device != current_device_id:
        try:
            torch.cuda.set_device(best_device)
            print(f"✓ Switched from GPU {current_device_id} to GPU {best_device}")
            return f'cuda:{best_device}'
        except Exception as e:
            print(f"✗ Failed to switch to GPU {best_device}: {e}")
            return f'cuda:{current_device_id}'
    else:
        print(f"✓ Staying with GPU {current_device_id}")
        return f'cuda:{current_device_id}'

def handle_gpu_oom_error(model, optimizer=None, scaler=None):
    """Handle GPU out of memory errors with comprehensive cleanup and recovery"""
    print("🚨 GPU Out of Memory Error - Attempting Recovery...")
    
    # Step 1: Clean up current GPU memory
    cleanup_gpu_memory()
    
    # Step 2: Try to switch to a different GPU
    new_device = check_gpu_memory_and_switch(min_free_gb=1.5)
    
    # Step 3: Move model to new device if different
    current_device = next(model.parameters()).device
    if str(current_device) != new_device:
        try:
            print(f"Moving model from {current_device} to {new_device}")
            model = model.to(new_device)
            
            # Update optimizer device if provided
            if optimizer:
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(new_device)
                            
            print(f"✓ Successfully moved model to {new_device}")
            
        except Exception as e:
            print(f"✗ Failed to move model to new device: {e}")
            raise
    
    return model, new_device

# Global device switching state
_device_switch_count = 0
_max_device_switches = 3  # Prevent infinite switching
_failed_gpus = set()  # Track GPUs that have failed with OOM
_gpu_failure_timestamps = {}  # Track when each GPU failed
_gpu_retry_delay = 300  # Wait 5 minutes before retrying a failed GPU

def safe_device_operation(func, *args, **kwargs):
    """Safely execute a function with automatic GPU memory management"""
    global _device_switch_count
    
    try:
        return func(*args, **kwargs)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and _device_switch_count < _max_device_switches:
            print(f"OOM Error (attempt {_device_switch_count + 1}/{_max_device_switches}): {e}")
            
            # Clean up and try to switch GPU
            cleanup_gpu_memory()
            new_device = check_gpu_memory_and_switch()
            _device_switch_count += 1
            
            # Update global device variable
            global DEVICE
            DEVICE = new_device
            
            # Try again
            try:
                return func(*args, **kwargs)
            except RuntimeError as e2:
                if _device_switch_count >= _max_device_switches:
                    print(f"✗ Max device switches reached. Final error: {e2}")
                raise e2
        else:
            raise e

# Update the original check_gpu_memory_and_switch function - Fixed recursive call
def find_best_available_gpu(min_free_gb=1.0):
    """Find the best available GPU with at least min_free_gb of free memory"""
    return check_gpu_memory_and_switch(min_free_gb=min_free_gb)

def force_gpu_switch():
    """Force switch to GPU with preference: cuda:2 -> cuda:1 -> best available, excluding failed GPUs"""
    global _failed_gpus, _gpu_failure_timestamps, _gpu_retry_delay
    
    if not torch.cuda.is_available():
        return 'cpu'
        
    device_count = torch.cuda.device_count()
    if device_count <= 1:
        return 'cuda:0'
    
    current_time = time.time()
    print("🔄 Force switching with GPU preference (excluding recently failed GPUs)...")
    
    # Check if any failed GPUs can be retried (after delay)
    retry_gpus = set()
    for gpu_id in list(_failed_gpus):
        if gpu_id in _gpu_failure_timestamps:
            if current_time - _gpu_failure_timestamps[gpu_id] > _gpu_retry_delay:
                print(f"🔄 GPU {gpu_id} retry delay expired, allowing retry")
                retry_gpus.add(gpu_id)
                _failed_gpus.discard(gpu_id)
                del _gpu_failure_timestamps[gpu_id]
    
    # Check all GPUs first
    gpu_memory = {}
    for i in range(device_count):
        if i in _failed_gpus:
            print(f"⛔ Skipping GPU {i} (recently failed with OOM)")
            continue
            
        try:
            memory_info = get_gpu_memory_info(i)
            if memory_info:
                gpu_memory[i] = memory_info
                free_gb = memory_info['free'] / (1024**3)
                status = "recently retried" if i in retry_gpus else "available"
                print(f"GPU {i} ({status}): {free_gb:.2f}GB free ({memory_info['utilization']:.1f}% used)")
        except Exception as e:
            print(f"Error checking GPU {i}: {e}")
            _failed_gpus.add(i)
            _gpu_failure_timestamps[i] = current_time
    
    # Try preferred GPUs in order: 2, 1, 0 (excluding failed ones)
    preference_order = [1, 0, 2]
    
    for preferred_gpu in preference_order:
        if (preferred_gpu < device_count and 
            preferred_gpu in gpu_memory and 
            preferred_gpu not in _failed_gpus):
            
            free_gb = gpu_memory[preferred_gpu]['free'] / (1024**3)
            if free_gb >= 1.5:  # Increased requirement to 1.5GB for safety
                try:
                    torch.cuda.set_device(preferred_gpu)
                    print(f"✅ Switched to preferred GPU {preferred_gpu} with {free_gb:.2f}GB free")
                    return f'cuda:{preferred_gpu}'
                except Exception as e:
                    print(f"Failed to switch to GPU {preferred_gpu}: {e}")
                    _failed_gpus.add(preferred_gpu)
                    _gpu_failure_timestamps[preferred_gpu] = current_time
    
    # Fallback to GPU with most memory (excluding failed ones)
    available_gpus = {k: v for k, v in gpu_memory.items() if k not in _failed_gpus}
    
    if available_gpus:
        best_device = max(available_gpus.keys(), key=lambda x: available_gpus[x]['free'])
        max_free_memory = available_gpus[best_device]['free']
        free_gb = max_free_memory / (1024**3)
        
        if free_gb >= 1.0:  # Lower threshold for fallback
            try:
                torch.cuda.set_device(best_device)
                print(f"🔄 Fallback: Switched to GPU {best_device} with {free_gb:.2f}GB free")
                return f'cuda:{best_device}'
            except Exception as e:
                print(f"✗ Failed to switch to GPU {best_device}: {e}")
                _failed_gpus.add(best_device)
                _gpu_failure_timestamps[best_device] = current_time
    
    # Last resort: try CPU or any available GPU (even failed ones if enough time passed)
    print("⚠️ All preferred GPUs failed, trying CPU or emergency fallback...")
    
    # If all GPUs failed recently, try the one that failed longest ago
    if _gpu_failure_timestamps:
        oldest_failed_gpu = min(_gpu_failure_timestamps.keys(), 
                              key=lambda x: _gpu_failure_timestamps[x])
        time_since_failure = current_time - _gpu_failure_timestamps[oldest_failed_gpu]
        
        if time_since_failure > 60:  # Try again after 1 minute as emergency
            try:
                torch.cuda.set_device(oldest_failed_gpu)
                print(f"🆘 Emergency retry of GPU {oldest_failed_gpu} (failed {time_since_failure:.0f}s ago)")
                return f'cuda:{oldest_failed_gpu}'
            except Exception as e:
                print(f"✗ Emergency retry of GPU {oldest_failed_gpu} failed: {e}")
    
    print("❌ All GPU options exhausted, falling back to CPU")
    return 'cpu'

# -----------------------------
# Class Configuration
# -----------------------------
CLASSES = ['background', 'pothole', 'crack', 'puddle', 'distressed_patch', 'mud']
NUM_CLASSES = len(CLASSES)

# -----------------------------
# Directory Configuration  
# -----------------------------
MODEL_DIR = "./models/"
LOGS_DIR = "./logs/"
RESULTS_DIR = "./results/"

# Create directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------
# Dataset Paths
# -----------------------------
PATH_POTHOLE_MIX_TRAIN = "../data/pothole_mix/training"
PATH_POTHOLE_MIX_VAL = "../data/pothole_mix/validation"
PATH_RTK = "../data/RTK"
PATH_RTK_VAL = "../data/RTK"
PATH_R2S100K_TRAIN = "../data/r2s100k/train"
PATH_R2S100K_TRAIN_LABELS = "../data/r2s100k/train-labels"
PATH_R2S100K_VAL = "../data/r2s100k/val"
PATH_R2S100K_VAL_LABELS = "../data/r2s100k/val_labels"
PATH_AUTOMINE = "../data/automine/train"
PATH_AUTOMINE_VAL = "../data/automine/val"

# Fallback paths for common dataset locations
FALLBACK_PATHS = {
    'pothole_mix_train': ["../data/pothole_mix/train", "../data/pothole-mix/training"],
    'pothole_mix_val': ["../data/pothole_mix/val", "../data/pothole-mix/validation"],
    'automine_train': ["../data/train/train_train", "../data/automine/train"],
    'automine_val': ["../data/valid/train_valid", "../data/automine/val"]
}

VAL_SPLIT_RATIO = 0.2

# -----------------------------
# Color Mappings
# -----------------------------
# SHREC colors (pothole_mix)
SHREC_COLORS = {
    "crack_red": (255, 0, 0),   # Maps to class 'crack' (2)
    "crack_green": (0, 255, 0), # Maps to class 'crack' (2)
    "pothole": (0, 0, 255)      # Maps to class 'pothole' (1)
}

# R2S100K colors
R2S_COLORS = {
    "water_puddle": (140, 160, 222),  # Maps to class 'puddle' (3)
    "distressed_patch": (119, 61, 128), # Maps to class 'distressed_patch' (4)
    "mud": (112, 84, 62)              # Maps to class 'mud' (5)
}

# Automine classes mapping
AUTOMINE_MAPPING = {
    0: 0,  # background -> background
    1: 4,  # defect -> distressed_patch
    2: 1,  # pothole -> pothole
    3: 3,  # puddle -> puddle
    4: 0,  # road -> background
}

# RTK mapping
RTK_TO_MODEL = {
    11: 1,  # pothole → pothole
    12: 2,  # craks → crack
    10: 3,  # waterPuddle → puddle
    9: 4,   # patchs → distressed_patch
}

# Wandb configuration
WANDB_ENABLED = True
WANDB_PROJECT = "road-defect-segmentation"

# Model configuration - matching original script exactly
NUM_CLASSES = 6  # background, pothole, crack, puddle, distressed_patch, mud  
CLASSES = ['background', 'pothole', 'crack', 'puddle', 'distressed_patch', 'mud']

# Dynamic checkpoint path based on architecture
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
CHECKPOINT_PATH = f"./models/{ARCHITECTURE}_road_defect_{TIMESTAMP}.pt"
SAVE_EVERY_N_EPOCHS = 20

# Dataset paths - using absolute paths to avoid confusion
DATA_ROOT = "/home/cloli/experimentation/cascade-road-segmentation/src/data"
PATH_POTHOLE_MIX_TRAIN = f"{DATA_ROOT}/pothole_mix/training"
PATH_POTHOLE_MIX_VAL = f"{DATA_ROOT}/pothole_mix/validation"
PATH_RTK = f"{DATA_ROOT}/RTK"
PATH_RTK_VAL = f"{DATA_ROOT}/RTK"
PATH_R2S100K_TRAIN = f"{DATA_ROOT}/r2s100k/train"
PATH_R2S100K_TRAIN_LABELS = f"{DATA_ROOT}/r2s100k/train-labels"
PATH_R2S100K_VAL = f"{DATA_ROOT}/r2s100k/val"
PATH_R2S100K_VAL_LABELS = f"{DATA_ROOT}/r2s100k/val_labels"  # Note: actual directory name
PATH_AUTOMINE = f"{DATA_ROOT}/automine/train"
PATH_AUTOMINE_VAL = f"{DATA_ROOT}/automine/valid"

# Fallback paths for dataset locations - including both absolute and relative paths
FALLBACK_PATHS = {
    'pothole_mix_train': [
        "../data/pothole_mix/training", 
        "../data/pothole_mix/train", 
        "../data/pothole-mix/training",
        f"{DATA_ROOT}/pothole_mix/training"
    ],
    'pothole_mix_val': [
        "../data/pothole_mix/validation", 
        "../data/pothole_mix/val", 
        "../data/pothole-mix/validation",
        f"{DATA_ROOT}/pothole_mix/validation"
    ],
    'automine_train': [
        "../data/train/train_train", 
        "../data/automine/train",
        f"{DATA_ROOT}/train/train_train"
    ],
    'automine_val': [
        "../data/valid/train_valid", 
        "../data/automine/val",
        f"{DATA_ROOT}/valid/train_valid"
    ],
    'r2s100k_train': [
        "../data/r2s100k/train",
        f"{DATA_ROOT}/r2s100k/train"
    ],
    'r2s100k_train_labels': [
        "../data/r2s100k/train-labels",
        f"{DATA_ROOT}/r2s100k/train-labels"
    ],
    'r2s100k_val_labels': [
        "../data/r2s100k/val_labels",
        "../data/r2s100k/val-labels",
        f"{DATA_ROOT}/r2s100k/val_labels"
    ]
}

# Legacy path lists for compatibility (using absolute paths)
RTK_DATASET_PATH = [PATH_RTK, PATH_RTK_VAL, f"{DATA_ROOT}/RTK"]
R2S_DATASET_PATH = [PATH_R2S100K_TRAIN, f"{DATA_ROOT}/r2s100k", f"{DATA_ROOT}/R2S100K"]  
POTHOLE_DATASET_PATH = [PATH_POTHOLE_MIX_TRAIN, f"{DATA_ROOT}/pothole-mix", f"{DATA_ROOT}/pothole_mix"]
AUTOMINE_DATASET_PATH = [PATH_AUTOMINE, f"{DATA_ROOT}/automine", f"{DATA_ROOT}/train/train_train"]

# Data loading
NUM_WORKERS = 4

# -----------------------------
# Architecture-Aware Configuration
# -----------------------------
def get_training_config_for_architecture(architecture=None):
    """
    Get training configuration optimized for the selected architecture
    
    Args:
        architecture (str, optional): Architecture name. Uses ARCHITECTURE if None.
        
    Returns:
        dict: Training configuration with optimized parameters
    """
    arch = architecture or ARCHITECTURE
    
    try:
        from architectures import get_training_config, get_recommended_batch_size
        
        # Get architecture-specific training config
        training_config = get_training_config(arch)
        
        # Get optimized batch size based on available GPU memory
        batch_size, grad_accum = get_recommended_batch_size(arch, gpu_memory_gb)
        
        # Override with architecture-specific values
        training_config['batch_size'] = batch_size
        training_config['gradient_accumulation'] = grad_accum
        training_config['effective_batch_size'] = batch_size * grad_accum
        
        return training_config
        
    except ImportError:
        print("⚠️ architectures.py not available, using default config")
        return {
            'base_lr': LEARNING_RATE,
            'max_lr': MAX_LR,
            'batch_size': BATCH_SIZE,
            'gradient_accumulation': GRADIENT_ACCUMULATION_STEPS,
            'effective_batch_size': BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
            'scheduler': 'onecycle',
            'warmup_epochs': 5,
            'weight_decay': 1e-4,
            'encoder_lr_factor': 0.1,
            'optimizer': 'adamw',
            'complexity': 'medium'
        }

def update_config_for_architecture(architecture=None):
    """
    Update global config variables based on architecture
    This function modifies global variables to match architecture requirements
    
    Args:
        architecture (str, optional): Architecture name. Uses ARCHITECTURE if None.
    """
    global BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS, LEARNING_RATE, MAX_LR
    
    arch = architecture or ARCHITECTURE
    training_config = get_training_config_for_architecture(arch)
    
    # Update global variables
    BATCH_SIZE = training_config['batch_size']
    GRADIENT_ACCUMULATION_STEPS = training_config['gradient_accumulation']
    LEARNING_RATE = training_config['base_lr']
    MAX_LR = training_config.get('max_lr', LEARNING_RATE * 3)
    
    print(f"🔧 Updated config for {arch}:")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Gradient Accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"   Effective Batch Size: {training_config['effective_batch_size']}")
    print(f"   Base LR: {LEARNING_RATE:.2e}")
    print(f"   Max LR: {MAX_LR:.2e}")
    print(f"   Scheduler: {training_config.get('scheduler', 'onecycle')}")

def print_architecture_summary():
    """Print current architecture configuration summary"""
    print(f"\n🏗️ Current Architecture: {ARCHITECTURE}")
    print(f"📊 Device: {DEVICE}")
    if gpu_memory_gb:
        print(f"🔧 GPU Memory: {gpu_memory_gb:.1f}GB")
    
    try:
        from architectures import get_architecture_info, print_training_recommendations
        
        arch_info = get_architecture_info(ARCHITECTURE) 
        if arch_info:
            print(f"📋 Model: {arch_info['name']}")
            if arch_info.get('encoder'):
                print(f"🧠 Encoder: {arch_info['encoder']}")
            
            complexity = arch_info.get('training_config', {}).get('complexity', 'unknown')
            print(f"⚡ Complexity: {complexity.title()}")
        
        # Show detailed training recommendations
        print_training_recommendations(ARCHITECTURE)
        
    except ImportError:
        print("⚠️ architectures.py not available for detailed info")
    except Exception as e:
        print(f"⚠️ Error getting architecture info: {e}")

# Auto-update config based on selected architecture
if ARCHITECTURE:
    try:
        update_config_for_architecture(ARCHITECTURE)
    except Exception as e:
        print(f"⚠️ Could not auto-update config for {ARCHITECTURE}: {e}")
        print("   Using default configuration values")
