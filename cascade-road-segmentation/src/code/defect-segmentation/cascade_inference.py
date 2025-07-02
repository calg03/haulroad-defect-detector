#!/usr/bin/env python3
"""
Cascade Road+Defect Segmentation Inference Script

- Uses a road segmentation model (SegFormer) to mask road regions
- Applies a defect segmentation model (UNet++) only on road regions using tiling
- Generates overlay visualization of original image + predicted defects
- Modular design with configurable architecture support
- Fixed weights paths for deployment
- No evaluation metrics - pure inference for production use
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import segmentation_models_pytorch as smp
import logging
from datetime import datetime
from typing import Tuple, Dict, Any, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import architecture support
try:
    from architectures import get_model, get_preprocessing_fn as get_arch_preprocessing
    ARCHITECTURES_AVAILABLE = True
except ImportError:
    ARCHITECTURES_AVAILABLE = False
    print("⚠️ architectures.py not available, using default UNet++ configuration")


class CascadeInference:
    """
    Cascade Road+Defect Segmentation Inference Engine
    
    Performs two-stage inference:
    1. Road segmentation using SegFormer
    2. Defect segmentation using UNet++ (or other architecture) on road regions only
    """
    
    def __init__(
        self,
        road_model_path: str = "/home/cloli/experimentation/cascade-road-segmentation/src/utils/segformer/best_epoch26_besto.pth",
        defect_model_path: str = "/home/cloli/experimentation/cascade-road-segmentation/src/models/unetplusplus_scse_road_defect_20250626_233608_best.pt",
        architecture: str = "unetplusplus_scse",
        device: Optional[str] = None
    ):
        """
        Initialize the cascade inference engine
        
        Args:
            road_model_path: Path to SegFormer road segmentation weights
            defect_model_path: Path to defect segmentation model weights
            architecture: Architecture name for defect segmentation model
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
        """
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.architecture = architecture
        
        # Fixed configuration for deployment
        self.segformer_pretrain = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
        self.target_size = (512, 512)
        self.img_size = (512, 512)
        
        # Defect classes
        self.classes = ['background', 'pothole', 'crack', 'puddle', 'distressed_patch', 'mud']
        self.num_classes = len(self.classes)
        
        # Color mapping for visualization (RGB format)
        self.defect_colors = {
            0: (0, 0, 0),           # background: black
            1: (0, 0, 255),         # pothole: blue 
            2: (0, 255, 0),         # crack: green
            3: (140, 160, 222),     # puddle: light blue
            4: (119, 61, 128),      # distressed_patch: purple
            5: (112, 84, 62)        # mud: brown
        }
        
        # Tiling configuration for large images
        self.tile_size = 512
        self.stride = 256
        self.min_road_coverage = 0.05  # Minimum road coverage to process tile
        self.confidence_threshold = 0.6  # Minimum confidence threshold
        
        # Load models
        self._load_models(road_model_path, defect_model_path)
        
        print(f"✅ Cascade inference initialized on {self.device}")
        print(f"🏗️ Architecture: {self.architecture}")
        print(f"🛣️ Road model: {os.path.basename(road_model_path)}")
        print(f"🚧 Defect model: {os.path.basename(defect_model_path)}")
    
    def _load_models(self, road_model_path: str, defect_model_path: str):
        """Load road and defect segmentation models"""
        try:
            # Load SegFormer for road segmentation
            self.road_processor = SegformerImageProcessor.from_pretrained(self.segformer_pretrain)
            self.road_model = SegformerForSemanticSegmentation.from_pretrained(
                self.segformer_pretrain, 
                num_labels=2, 
                ignore_mismatched_sizes=True
            )
            
            # Load road model weights
            road_state_dict = torch.load(road_model_path, map_location=self.device)
            self.road_model.load_state_dict(road_state_dict, strict=False)
            self.road_model.to(self.device).eval()
            
            # Load defect segmentation model
            if ARCHITECTURES_AVAILABLE:
                # Use modular architecture system
                self.defect_model = get_model(
                    architecture=self.architecture,
                    num_classes=self.num_classes,
                    device=self.device
                )
                self.preprocessing_fn = get_arch_preprocessing(self.architecture)
            else:
                # Default UNet++ configuration
                self.defect_model = smp.UnetPlusPlus(
                    encoder_name="efficientnet-b5",
                    encoder_depth=5,
                    encoder_weights="imagenet",
                    decoder_use_batchnorm=True,
                    decoder_channels=(256, 128, 64, 32, 16),
                    decoder_attention_type="scse",
                    in_channels=3,
                    classes=self.num_classes,
                    activation=None
                ).to(self.device)
                
                from segmentation_models_pytorch.encoders import get_preprocessing_fn
                self.preprocessing_fn = get_preprocessing_fn("efficientnet-b5", "imagenet")
            
            # Load defect model weights
            try:
                defect_state_dict = torch.load(defect_model_path, map_location=self.device, weights_only=False)
            except TypeError:
                defect_state_dict = torch.load(defect_model_path, map_location=self.device)
            
            if 'model_state_dict' in defect_state_dict:
                self.defect_model.load_state_dict(defect_state_dict['model_state_dict'])
            else:
                self.defect_model.load_state_dict(defect_state_dict)
            
            self.defect_model.eval()
            
            print("✅ Models loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for defect model"""
        return self.preprocessing_fn(image).astype(np.float32)
    
    def _sliding_windows(self, height: int, width: int):
        """Generate sliding windows for tiled inference"""
        tile, stride = self.tile_size, self.stride
        
        if height <= tile and width <= tile:
            yield 0, 0, width, height
            return
        
        # Regular grid
        for y in range(0, max(1, height - tile + 1), stride):
            for x in range(0, max(1, width - tile + 1), stride):
                yield x, y, min(x + tile, width), min(y + tile, height)
        
        # Right edge
        if width > tile:
            for y in range(0, height - tile + 1, stride):
                yield width - tile, y, width, y + tile
        
        # Bottom edge
        if height > tile:
            for x in range(0, width - tile + 1, stride):
                yield x, height - tile, x + tile, height
        
        # Bottom-right corner
        if width > tile and height > tile:
            yield width - tile, height - tile, width, height
    
    def _mask_to_rgb(self, mask: np.ndarray) -> np.ndarray:
        """Convert class mask to RGB visualization"""
        rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_idx, color in self.defect_colors.items():
            rgb[mask == class_idx] = color
        return rgb
    
    def predict_road_mask(self, image: Image.Image) -> np.ndarray:
        """
        Predict road mask using SegFormer
        
        Args:
            image: PIL Image
            
        Returns:
            Binary road mask (1=road, 0=non-road)
        """
        width, height = image.size
        
        # Preprocess for SegFormer
        inputs = self.road_processor(
            images=image, 
            return_tensors="pt", 
            size=self.target_size
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            logits = self.road_model(**inputs).logits
            # Interpolate back to original size
            road_mask = F.interpolate(
                logits, 
                size=(height, width), 
                mode="bilinear", 
                align_corners=False
            ).argmax(1)[0].cpu().numpy()
        
        return road_mask.astype(np.uint8)
    
    def predict_defects_on_road(self, image: np.ndarray, road_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict defects only on road regions using tiled inference
        
        Args:
            image: Original RGB image as numpy array
            road_mask: Binary road mask
            
        Returns:
            Tuple of (defect_mask, confidence_map)
        """
        height, width = image.shape[:2]
        
        # Apply road mask to image
        road_image = image * road_mask[:, :, None]
        
        # Initialize accumulation arrays
        probs_accumulated = np.zeros((self.num_classes, height, width), dtype=np.float32)
        counts_accumulated = np.zeros((height, width), dtype=np.float32)
        confidence_accumulated = np.zeros((height, width), dtype=np.float32)
        
        # Process each tile
        for x1, y1, x2, y2 in self._sliding_windows(height, width):
            tile_height, tile_width = y2 - y1, x2 - x1
            
            if tile_height == 0 or tile_width == 0:
                continue
            
            # Extract road mask for this tile
            tile_road_mask = road_mask[y1:y2, x1:x2]
            
            # Skip tiles with insufficient road coverage
            if tile_road_mask.size == 0 or tile_road_mask.mean() < self.min_road_coverage:
                continue
            
            # Extract road-masked image patch
            tile_image = road_image[y1:y2, x1:x2]
            if tile_image.size == 0 or np.sum(tile_image) == 0:
                continue
            
            # Resize to model input size
            tile_resized = cv2.resize(
                tile_image.astype(np.uint8), 
                self.img_size, 
                interpolation=cv2.INTER_LINEAR
            )
            
            # Preprocess
            tile_preprocessed = self._preprocess_image(tile_resized)
            tile_tensor = to_tensor(tile_preprocessed).unsqueeze(0).float().to(self.device)
            
            # Inference
            with torch.no_grad():
                logits = self.defect_model(tile_tensor)
                probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            
            # Resize probabilities back to tile size
            probs_resized = np.stack([
                cv2.resize(p, (tile_width, tile_height), interpolation=cv2.INTER_LINEAR) 
                for p in probs
            ])
            
            # Calculate confidence (max probability across classes)
            max_confidence = np.max(probs_resized, axis=0)
            
            # Only accumulate high-confidence predictions
            high_conf_mask = max_confidence > self.confidence_threshold
            
            if np.any(high_conf_mask):
                probs_accumulated[:, y1:y2, x1:x2][:, high_conf_mask] += probs_resized[:, high_conf_mask]
                counts_accumulated[y1:y2, x1:x2][high_conf_mask] += 1
                confidence_accumulated[y1:y2, x1:x2] = np.maximum(
                    confidence_accumulated[y1:y2, x1:x2], 
                    max_confidence
                )
        
        # Normalize accumulated probabilities
        counts_accumulated[counts_accumulated == 0] = 1
        probs_accumulated /= counts_accumulated
        
        # Generate final prediction
        defect_mask = probs_accumulated.argmax(0).astype(np.uint8)
        defect_mask = np.clip(defect_mask, 0, self.num_classes - 1)
        
        # Set low-confidence areas to background
        low_conf_mask = confidence_accumulated < self.confidence_threshold
        defect_mask[low_conf_mask] = 0
        
        # Only keep defects within road boundaries
        defect_mask[road_mask == 0] = 0
        
        return defect_mask, confidence_accumulated
    
    def create_overlay(self, original_image: np.ndarray, defect_mask: np.ndarray, alpha: float = 0.6) -> np.ndarray:
        """
        Create overlay of original image with defect predictions
        
        Args:
            original_image: Original RGB image
            defect_mask: Predicted defect mask
            alpha: Transparency factor for overlay (0=transparent, 1=opaque)
            
        Returns:
            RGB image with defect overlay
        """
        # Convert defect mask to RGB
        defect_rgb = self._mask_to_rgb(defect_mask)
        
        # Create overlay
        overlay = original_image.copy()
        
        # Only overlay where there are defects (non-background)
        defect_pixels = defect_mask > 0
        if np.any(defect_pixels):
            overlay[defect_pixels] = (
                (1 - alpha) * original_image[defect_pixels] + 
                alpha * defect_rgb[defect_pixels]
            ).astype(np.uint8)
        
        return overlay
    
    def predict(self, image_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Run full cascade inference on a single image
        
        Args:
            image_path: Path to input image
            output_dir: Optional directory to save outputs
            
        Returns:
            Dictionary containing results and file paths
        """
        try:
            # Load image
            image_pil = Image.open(image_path).convert("RGB")
            image_np = np.array(image_pil)
            
            # Get image info
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            
            print(f"🔍 Processing: {image_name}")
            print(f"📐 Image size: {image_np.shape[:2]}")
            
            # Step 1: Road segmentation
            print("🛣️ Segmenting roads...")
            road_mask = self.predict_road_mask(image_pil)
            road_coverage = road_mask.mean()
            print(f"📊 Road coverage: {road_coverage:.1%}")
            
            # Step 2: Defect segmentation on roads
            print("🚧 Detecting defects...")
            defect_mask, confidence_map = self.predict_defects_on_road(image_np, road_mask)
            
            # Count defects by class
            defect_counts = {}
            total_defect_pixels = 0
            for class_idx, class_name in enumerate(self.classes):
                if class_idx == 0:  # Skip background
                    continue
                count = np.sum(defect_mask == class_idx)
                defect_counts[class_name] = int(count)
                total_defect_pixels += count
            
            print(f"🔍 Total defect pixels: {total_defect_pixels}")
            for class_name, count in defect_counts.items():
                if count > 0:
                    print(f"   {class_name}: {count} pixels")
            
            # Step 3: Create overlay
            print("🎨 Creating overlay...")
            overlay = self.create_overlay(image_np, defect_mask)
            
            # Prepare results
            results = {
                'image_name': image_name,
                'image_shape': image_np.shape,
                'road_coverage': float(road_coverage),
                'total_defect_pixels': int(total_defect_pixels),
                'defect_counts': defect_counts,
                'mean_confidence': float(confidence_map[defect_mask > 0].mean()) if total_defect_pixels > 0 else 0.0,
                'processing_status': 'success'
            }
            
            # Save outputs if directory specified
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
                # Save overlay
                overlay_path = os.path.join(output_dir, f"{image_name}_overlay.png")
                cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                results['overlay_path'] = overlay_path
                
                # Save defect mask
                mask_rgb = self._mask_to_rgb(defect_mask)
                mask_path = os.path.join(output_dir, f"{image_name}_defects.png")
                cv2.imwrite(mask_path, cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))
                results['defect_mask_path'] = mask_path
                
                # Save road mask
                road_rgb = np.stack([road_mask * 255] * 3, axis=-1)
                road_path = os.path.join(output_dir, f"{image_name}_road.png")
                cv2.imwrite(road_path, road_rgb)
                results['road_mask_path'] = road_path
                
                print(f"💾 Outputs saved to: {output_dir}")
            
            # Store arrays in results for API use
            results['overlay_array'] = overlay
            results['defect_mask_array'] = defect_mask
            results['road_mask_array'] = road_mask
            results['confidence_map_array'] = confidence_map
            
            print("✅ Inference completed successfully")
            return results
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            return {
                'image_name': os.path.splitext(os.path.basename(image_path))[0],
                'processing_status': 'error',
                'error_message': str(e)
            }
    
    def predict_batch(self, image_paths: list, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Run cascade inference on multiple images
        
        Args:
            image_paths: List of image file paths
            output_dir: Optional directory to save outputs
            
        Returns:
            Dictionary containing batch results
        """
        results = []
        successful = 0
        
        print(f"🚀 Starting batch inference on {len(image_paths)} images")
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"\n[{i}/{len(image_paths)}] Processing: {os.path.basename(image_path)}")
            
            result = self.predict(image_path, output_dir)
            results.append(result)
            
            if result['processing_status'] == 'success':
                successful += 1
        
        batch_results = {
            'total_images': len(image_paths),
            'successful': successful,
            'failed': len(image_paths) - successful,
            'results': results
        }
        
        print(f"\n🎯 Batch completed: {successful}/{len(image_paths)} successful")
        return batch_results


def main():
    """Example usage of cascade inference"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cascade Road+Defect Segmentation Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image or directory")
    parser.add_argument("--output", type=str, default="./inference_output", help="Output directory")
    parser.add_argument("--road-model", type=str, 
                       default="/home/cloli/experimentation/cascade-road-segmentation/src/utils/segformer/best_epoch26_besto.pth",
                       help="Path to road segmentation model")
    parser.add_argument("--defect-model", type=str,
                       default="/home/cloli/experimentation/cascade-road-segmentation/src/models/unetplusplus_scse_road_defect_20250626_233608_best.pt",
                       help="Path to defect segmentation model")
    parser.add_argument("--architecture", type=str, default="unetplusplus_scse",
                       help="Architecture name for defect model")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], 
                       help="Device to run inference on")
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inference = CascadeInference(
        road_model_path=args.road_model,
        defect_model_path=args.defect_model,
        architecture=args.architecture,
        device=args.device
    )
    
    # Process single image or batch
    if os.path.isfile(args.image):
        # Single image
        result = inference.predict(args.image, args.output)
        print(f"\n📊 Results: {result}")
    
    elif os.path.isdir(args.image):
        # Batch processing
        import glob
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(args.image, ext)))
            image_paths.extend(glob.glob(os.path.join(args.image, ext.upper())))
        
        if not image_paths:
            print(f"❌ No images found in {args.image}")
            return
        
        results = inference.predict_batch(image_paths, args.output)
        print(f"\n📊 Batch Results: {results}")
    
    else:
        print(f"❌ Invalid input path: {args.image}")


if __name__ == "__main__":
    main()