#!/usr/bin/env python3
"""
Test Configuration for Modular Road Defect Segmentation

This configuration file allows you to easily adjust testing parameters
without modifying the main script. Simply modify the values below and run:

    python test_modular_segmentation.py --config test_config.py

Or override specific parameters:

    python test_modular_segmentation.py --config test_config.py --batch_size 16
"""

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Architecture to test - must match one from architectures.py
ARCHITECTURE = 'unetplusplus_scse'  # Change this to test different architectures

# Encoder override (optional) - useful for testing older weights
# For FPN: use 'resnet50' for older weights, 'resnet101' for newer weights
# Set to None to use default encoder from architecture config
ENCODER_OVERRIDE = None # Options: None, 'resnet50', 'resnet101'

# Auto-detect encoder from checkpoint (if ENCODER_OVERRIDE is None)
AUTO_DETECT_ENCODER = True

# Fix for tensor dtype issues (DoubleTensor/HalfTensor mismatch)
FIX_TENSOR_DTYPE = True

# Path to trained model checkpoint
MODEL_PATH = '/home/cloli/experimentation/cascade-road-segmentation/src/models/unetplusplus_scse_road_defect_20250626_233608_best.pt'  # Adjust path as needed

# Alternative model paths for batch testing
MODEL_PATHS = {
    'unet': './models/unet_best.pt',
    'unetplusplus': './models/unetpp_best.pt', 
    'deeplabv3': './models/deeplabv3_best.pt',
    'deeplabv3plus': './models/deeplabv3plus_best.pt',
    'fpn': './models/fpn_best.pt',
    'pspnet': './models/pspnet_best.pt',
    'transunet': './models/transunet_best.pt',
    'swin_unet': './models/swin_unet_best.pt',
}

# =============================================================================
# DATA CONFIGURATION  
# =============================================================================

# Input directory containing test images
INPUT_DIR = '../data/r2s100k/test'

# Output directory for results  
OUTPUT_DIR = './results/test_output'

# Ground truth directory for evaluation (set to None to skip evaluation)
GT_DIR = '../data/r2s100k/test-labels'  # Set to None if no ground truth available

# =============================================================================
# DIRECTORY DEBUGGING
# =============================================================================

# Force absolute paths to prevent any confusion
import os
_current_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.abspath(os.path.join(_current_dir, INPUT_DIR))
GT_DIR = os.path.abspath(os.path.join(_current_dir, GT_DIR)) if GT_DIR else None
OUTPUT_DIR = os.path.abspath(os.path.join(_current_dir, OUTPUT_DIR))

# =============================================================================
# PROCESSING CONFIGURATION
# =============================================================================

# Batch size for inference (adjust based on GPU memory)
BATCH_SIZE = 8

# Number of data loading workers
NUM_WORKERS = 4

# Input image size (should match training configuration)
IMG_SIZE = 512

# Device configuration ('auto', 'cuda', 'cpu', or specific like 'cuda:0')
DEVICE = 'auto'

# Model configuration
NUM_CLASSES = 6  # Number of segmentation classes

DATASET = 'r2s100k' #automine, pothole_mix, r2s100k
# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================

# Save overlay images (prediction overlaid on original image)
SAVE_OVERLAYS = False

# Save prediction masks as grayscale images
SAVE_MASKS = False

# Save probability maps for each class (takes more disk space)
SAVE_PROBABILITY_MAPS = False

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

# Run evaluation against ground truth (requires GT_DIR to be set)
EVALUATE = True

# Use direct evaluation mode (faster, only computes metrics without saving outputs)
DIRECT_EVALUATION = True  # Set to False to save overlay masks and visualizations


# Class names (must match training configuration)
CLASS_NAMES = ['background', 'pothole', 'crack', 'puddle', 'distressed_patch', 'mud']

# =============================================================================
# PERFORMANCE CONFIGURATION
# =============================================================================

# Use mixed precision for faster inference (if supported)
MIXED_PRECISION = True

# Verbose output (print per-image statistics) - set to False to reduce output
VERBOSE = True  # Set to True to see detailed image-mask pairing information

# =============================================================================
# BATCH TESTING CONFIGURATION
# =============================================================================

# Architectures to test in batch mode (used by batch testing scripts)
BATCH_TEST_ARCHITECTURES = [
    'unet',
    'unetplusplus', 
    'deeplabv3',
    'deeplabv3plus',
    'fpn',
    'pspnet',
    # 'transunet',  # Uncomment if transformers are available
    # 'swin_unet',  # Uncomment if transformers are available
]

# Batch test output directory
BATCH_OUTPUT_DIR = './results/batch_test'

# =============================================================================
# ADVANCED CONFIGURATION
# =============================================================================

# Color mapping for visualization (RGB values)
COLOR_MAP = {
    0: (0, 0, 0),         # background: black
    1: (0, 0, 255),       # pothole: blue  
    2: (0, 255, 0),       # crack: green
    3: (140, 160, 222),   # puddle: light blue
    4: (119, 61, 128),    # distressed_patch: purple
    5: (112, 84, 62)      # mud: brown
}

# Training color mappings for ground truth processing (from training)
SHREC_COLORS = {
    "crack_red": (255, 0, 0),   # Maps to class 'crack' (2)
    "crack_green": (0, 255, 0), # Maps to class 'crack' (2) 
    "pothole": (0, 0, 255)      # Maps to class 'pothole' (1)
}

R2S_COLORS = {
    "water_puddle": (140, 160, 222),     # Maps to class 'puddle' (3)
    "distressed_patch": (119, 61, 128),  # Maps to class 'distressed_patch' (4)
    "mud": (112, 84, 62)                 # Maps to class 'mud' (5)
}

AUTOMINE_MAPPING = {
    0: 0,    # background -> background
    1: 4,    # defect -> distressed_patch (matches training)
    2: 1,    # pothole -> pothole
    3: 3,    # puddle -> puddle
    4: 0,    # road -> background (road surface is not a defect!)
    255: 0   # Unknown/invalid -> background
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_model_path(architecture):
    """Get model path for given architecture"""
    return MODEL_PATHS.get(architecture, f'./models/{architecture}_best.pt')

def get_output_dir(architecture, base_dir=None):
    """Get architecture-specific output directory"""
    if base_dir is None:
        base_dir = OUTPUT_DIR
    return os.path.join(base_dir, architecture)

def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check if architecture is valid
    try:
        from architectures import get_available_architectures
        available = get_available_architectures()
        if ARCHITECTURE not in available:
            errors.append(f"Architecture '{ARCHITECTURE}' not available. Choose from: {available}")
    except ImportError:
        errors.append("Could not import architectures module")
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        errors.append(f"Model file not found: {MODEL_PATH}")
    
    # Check if input directory exists
    if not os.path.exists(INPUT_DIR):
        errors.append(f"Input directory not found: {INPUT_DIR}")
    
    # Check if ground truth directory exists (if evaluation is enabled)
    if EVALUATE and GT_DIR and not os.path.exists(GT_DIR):
        errors.append(f"Ground truth directory not found: {GT_DIR}")
    
    return errors

if __name__ == "__main__":
    """Test configuration validation"""
    import os
    
    print("🧪 Test Configuration Validation")
    print("=" * 50)
    
    # Debug directory paths
    print(f"\n🔍 Directory Path Resolution:")
    print(f"   Script location: {os.path.dirname(os.path.abspath(__file__))}")
    print(f"   Input directory: {INPUT_DIR}")
    print(f"   GT directory: {GT_DIR}")
    print(f"   Output directory: {OUTPUT_DIR}")
    
    # Check actual file counts
    if os.path.exists(INPUT_DIR):
        jpg_files = len([f for f in os.listdir(INPUT_DIR) if f.endswith('.jpg')])
        mask_files = len([f for f in os.listdir(INPUT_DIR) if f.endswith('_mask.png')])
        print(f"   Files in input dir: {jpg_files} JPG, {mask_files} masks")
    else:
        print(f"   ❌ Input directory does not exist: {INPUT_DIR}")
    
    errors = validate_config()
    
    if errors:
        print("❌ Configuration errors found:")
        for error in errors:
            print(f"   - {error}")
    else:
        print("✅ Configuration is valid!")
    
    print(f"\nCurrent Configuration:")
    print(f"   Architecture: {ARCHITECTURE}")
    print(f"   Model Path: {MODEL_PATH}")
    print(f"   Input Directory: {INPUT_DIR}")
    print(f"   Output Directory: {OUTPUT_DIR}")
    print(f"   Ground Truth Directory: {GT_DIR}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Image Size: {IMG_SIZE}")
    print(f"   Device: {DEVICE}")
    print(f"   Evaluation: {EVALUATE}")
