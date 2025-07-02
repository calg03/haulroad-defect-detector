#!/usr/bin/env python
"""
Main script to run UNet++ training for road defect segmentation.
This uses the FIXED modular approach that follows the original script's stable patterns.
"""
import pkgutil
if not hasattr(pkgutil, 'ImpImporter'):
    pkgutil.ImpImporter = None
    
import torch
from trainer_fixed import FixedTrainer


def main():
    """Main execution function - uses the fixed trainer"""
    
    print("🚀 Starting UNet++ road defect segmentation training...")
    print("📋 Using FIXED modular trainer (stable like original script)")
    
    # Clear any existing GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"🧹 Cleared GPU cache")
    
    try:
        # Create and run the fixed trainer
        trainer = FixedTrainer()
        trainer.train()
        
        print("✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
