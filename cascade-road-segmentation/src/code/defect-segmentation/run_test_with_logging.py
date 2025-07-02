#!/usr/bin/env python3
"""
Run test with logging and defect analysis
"""

import os
import sys
import subprocess
import datetime
import glob
import cv2
import numpy as np
from pathlib import Path

def count_defects_in_dataset(data_dir):
    """Count images containing defects (values 1, 2, 3) in the dataset"""
    print(f"🔍 Analyzing defects in dataset: {data_dir}")
    
    mask_files = glob.glob(os.path.join(data_dir, "*_mask.png"))
    print(f"📊 Found {len(mask_files)} mask files")
    
    defect_stats = {
        1: [],  # defect
        2: [],  # pothole  
        3: [],  # puddle
    }
    
    total_with_defects = 0
    total_files = len(mask_files)
    
    for mask_file in mask_files:
        try:
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
                
            unique_vals = np.unique(mask)
            has_defects = False
            
            # Check for each defect type
            for defect_val in [1, 2, 3]:
                if defect_val in unique_vals:
                    defect_stats[defect_val].append(os.path.basename(mask_file))
                    has_defects = True
            
            if has_defects:
                total_with_defects += 1
                
        except Exception as e:
            print(f"⚠️ Error reading {mask_file}: {e}")
    
    # Print results
    print(f"\n📈 Defect Analysis Results:")
    print(f"{'='*50}")
    print(f"Total mask files: {total_files}")
    print(f"Files with defects: {total_with_defects} ({100*total_with_defects/total_files:.1f}%)")
    print(f"Files without defects: {total_files - total_with_defects} ({100*(total_files-total_with_defects)/total_files:.1f}%)")
    
    print(f"\n🎯 Defect Type Breakdown:")
    defect_names = {1: "defect", 2: "pothole", 3: "puddle"}
    
    for defect_val, files in defect_stats.items():
        count = len(files)
        percentage = (count / total_files) * 100 if total_files > 0 else 0
        print(f"  {defect_names[defect_val]:>8s} (value {defect_val}): {count:>3d} files ({percentage:>5.1f}%)")
        
        # Show first few files as examples
        if files:
            print(f"    Examples: {', '.join(files[:3])}")
            if len(files) > 3:
                print(f"    ... and {len(files)-3} more")
    
    return defect_stats, total_with_defects

def main():
    # Get current timestamp for log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"test_run_{timestamp}.log"
    
    print(f"🚀 Starting test run with logging")
    print(f"📝 Log file: {log_file}")
    
    # First, analyze the dataset for defects
    try:
        # Read config to get data directory
        sys.path.append('.')
        import test_config
        data_dir = test_config.INPUT_DIR
        
        if os.path.exists(data_dir):
            print(f"\n🔍 Pre-test defect analysis:")
            defect_stats, total_with_defects = count_defects_in_dataset(data_dir)
        else:
            print(f"⚠️ Data directory not found: {data_dir}")
            
    except Exception as e:
        print(f"⚠️ Error during defect analysis: {e}")
    
    # Run the main test and capture output
    print(f"\n🧪 Running main test...")
    print(f"{'='*60}")
    
    try:
        # Run the test and capture both stdout and stderr
        result = subprocess.run(
            [sys.executable, "test_modular_segmentation.py", "--config", "test_config.py"],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        # Write to log file
        with open(log_file, 'w') as f:
            f.write(f"Test Run Log - {timestamp}\n")
            f.write("="*60 + "\n\n")
            
            if result.stdout:
                f.write("STDOUT:\n")
                f.write(result.stdout)
                f.write("\n\n")
            
            if result.stderr:
                f.write("STDERR:\n") 
                f.write(result.stderr)
                f.write("\n\n")
            
            f.write(f"Return code: {result.returncode}\n")
        
        # Display results
        print("✅ Test completed!")
        print(f"📄 Full output saved to: {log_file}")
        print(f"🔢 Return code: {result.returncode}")
        
        if result.returncode == 0:
            print("🎉 Test finished successfully!")
        else:
            print("❌ Test finished with errors")
            
        # Show last few lines of output
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            print(f"\n📋 Last few lines of output:")
            for line in lines[-10:]:
                print(f"   {line}")
                
        if result.stderr:
            print(f"\n⚠️ Errors/warnings:")
            error_lines = result.stderr.strip().split('\n')
            for line in error_lines[-5:]:
                print(f"   {line}")
        
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out after 10 minutes")
    except Exception as e:
        print(f"❌ Error running test: {e}")
    
    print(f"\n📊 Summary:")
    print(f"   Log file: {log_file}")
    print(f"   Defect analysis completed")
    print(f"   Test execution completed")

if __name__ == "__main__":
    main()
