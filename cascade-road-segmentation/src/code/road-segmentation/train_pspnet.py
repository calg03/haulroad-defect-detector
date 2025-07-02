#!/usr/bin/env python3
"""
PSPNet Fine-tuning Script for Off-Road Segmentation

This script uses a project-style PSPNet model built on a custom ResNet50
(inlined below) that includes the extra convolutional layers (conv2 and conv3)
as originally used during training. The dataset is handled as intended by the
original author: the script recursively searches for "*small.txt" files,
reads base image names, appends ".jpg" to get the image path and "*label_raw.png"
to get the segmentation mask path, and splits the data into train/validation/test
(as 10% of the training samples are held out for validation).

Pretrained weights (if available) can be loaded for fine-tuning. Validation
metrics (loss, pixel accuracy, and mIoU) are computed at the end of every epoch,
and the best model is saved.
"""
import pkgutil
if not hasattr(pkgutil, 'ImpImporter'):
    pkgutil.ImpImporter = None

import os
import glob
import random
import numpy as np
from collections import OrderedDict
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF


def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

# ---------------------------
# Custom ResNet Code (inlined)
# ---------------------------


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, deep_base=True):
        super(ResNet, self).__init__()
        self.deep_base = deep_base
        if not self.deep_base:
            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
        else:
            self.inplanes = 128
            self.conv1 = conv3x3(3, 64, stride=2)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = conv3x3(64, 64)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = conv3x3(64, 128)
            self.bn3 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        if self.deep_base:
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model_path = './initmodel/resnet50_v2.pth'
        model.load_state_dict(torch.load(model_path), strict=False)
    return model

# =============================================================================
# End Custom ResNet Code
# =============================================================================

# ---------------------------
# PSPNet Model Definition
# ---------------------------

class PPMModule(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPMModule, self).__init__()
        self.features = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ) for bin in bins
        ])
    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), size=x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, dim=1)

class PSPNet(nn.Module):
    def __init__(self, layers=50, bins=(1,2,3,6), dropout=0.1, classes=2, zoom_factor=8,
                 use_ppm=True, criterion=nn.CrossEntropyLoss(ignore_index=255), pretrained=True):
        super(PSPNet, self).__init__()
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion
        # Use our custom resnet50 with deep_base=True for expanded layer0.
        resnet = resnet50(pretrained=pretrained, deep_base=True)
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.conv2,
            resnet.bn2,
            resnet.relu,
            resnet.conv3,
            resnet.bn3,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation = (2,2)
                m.padding = (2,2)
                m.stride = (1,1)
            elif 'downsample.0' in n:
                m.stride = (1,1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation = (4,4)
                m.padding = (4,4)
                m.stride = (1,1)
            elif 'downsample.0' in n:
                m.stride = (1,1)
        fea_dim = 2048
        if use_ppm:
            self.ppm = PPMModule(fea_dim, int(fea_dim/len(bins)), bins)
            fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1)
        )
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1)
            )
    def forward(self, x, y=None):
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0, "Input dims must satisfy (H-1)%8==0 and (W-1)%8==0."
        h = int((x_size[2]-1)/8 * self.zoom_factor + 1)
        w = int((x_size[3]-1)/8 * self.zoom_factor + 1)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x = self.layer4(x_tmp)
        if self.use_ppm:
            x = self.ppm(x)
        x = self.cls(x)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h,w), mode='bilinear', align_corners=True)
        if self.training and y is not None:
            aux = self.aux(x_tmp)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h,w), mode='bilinear', align_corners=True)
            main_loss = self.criterion(x, y)
            aux_loss = self.criterion(aux, y)
            return x.max(1)[1], main_loss, aux_loss
        else:
            return x

# ---------------------------
# Dataset Definition
# ---------------------------

class OffRoadSmallDataset(Dataset):
    def __init__(self, dataset_dir, split="train", target_size=(473,473), augment=False, binary=True):
        self.dataset_dir = dataset_dir
        self.split = split.lower()
        self.target_size = target_size
        self.augment = augment
        self.binary = binary
        label_suffix = '*label_raw.png'
        image_suffix = '.jpg'
        txt_pattern = os.path.join(dataset_dir, '**', '**', '**', '*small.txt')
        txt_files = glob.glob(txt_pattern, recursive=True)
        train_img_list, train_lbl_list = [], []
        test_img_list, test_lbl_list = [], []
        for txt_file in txt_files:
            with open(txt_file, 'r') as f:
                for line in f:
                    base_name = line.strip()
                    base_path = os.path.join(os.path.dirname(txt_file), base_name)
                    img_path = base_path + image_suffix
                    lbl_matches = glob.glob(base_path + label_suffix)
                    if not lbl_matches:
                        continue
                    label_path = lbl_matches[0]
                    if 'train' in label_path:
                        train_img_list.append(img_path)
                        train_lbl_list.append(label_path)
                    elif 'test' in label_path:
                        test_img_list.append(img_path)
                        test_lbl_list.append(label_path)
        random.Random(0).shuffle(train_img_list)
        random.Random(0).shuffle(train_lbl_list)
        split_idx = round(len(train_img_list) * 0.10)
        self.train_img_list = train_img_list[split_idx:]
        self.train_lbl_list = train_lbl_list[split_idx:]
        self.val_img_list = train_img_list[:split_idx]
        self.val_lbl_list = train_lbl_list[:split_idx]
        self.test_img_list = test_img_list
        self.test_lbl_list = test_lbl_list
        if self.split == "train":
            self.img_paths = self.train_img_list
            self.lbl_paths = self.train_lbl_list
        elif self.split in ["val", "validation"]:
            self.img_paths = self.val_img_list
            self.lbl_paths = self.val_lbl_list
        elif self.split == "test":
            self.img_paths = self.test_img_list
            self.lbl_paths = self.test_lbl_list
        else:
            raise ValueError("Unsupported split: " + self.split)
        self.image_transform = transforms.Compose([
            transforms.Resize(self.target_size, interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(self.target_size, interpolation=Image.NEAREST),
            transforms.Lambda(lambda img: torch.from_numpy(np.array(img)).long())
        ])
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        mask = Image.open(self.lbl_paths[idx])
        if self.binary:
            mask_np = np.array(mask)
            mask_np = np.where(mask_np == 1, 1, 0).astype(np.uint8)
            mask = Image.fromarray(mask_np)
        if self.augment and self.split == "train":
            img, mask = self.random_augment(img, mask)
        img = self.image_transform(img)
        mask = self.mask_transform(mask)
        return img, mask
    def random_augment(self, img, mask):
        # Basic random crop and horizontal flip.
        width, height = img.size
        target_h, target_w = self.target_size
        if width < target_w or height < target_h:
            img = img.resize((max(width, target_w), max(height, target_h)), Image.BILINEAR)
            mask = mask.resize((max(width, target_w), max(height, target_h)), Image.NEAREST)
            width, height = img.size
        left = random.randint(0, width - target_w)
        upper = random.randint(0, height - target_h)
        img = img.crop((left, upper, left + target_w, upper + target_h))
        mask = mask.crop((left, upper, left + target_w, upper + target_h))
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask

# ---------------------------
# Evaluation Metrics
# ---------------------------

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_pixels = 0
    intersection = [0, 0]
    union = [0, 0]
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)  # logits
            loss = model.criterion(outputs, masks)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            total_correct += (preds == masks).sum().item()
            total_pixels += torch.numel(masks)
            for cls in [0, 1]:
                pred_inds = (preds == cls)
                target_inds = (masks == cls)
                intersection[cls] += (pred_inds & target_inds).sum().item()
                union[cls] += (pred_inds | target_inds).sum().item()
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_pixels
    iou = [(intersection[c] / union[c] if union[c] > 0 else 1.0) for c in [0, 1]]
    mIoU = sum(iou) / len(iou)
    return avg_loss, accuracy, mIoU

# ---------------------------
# Training Loop
# ---------------------------

def train_model(model, train_loader, val_loader, optimizer, num_epochs, device, aux_loss_weight=0.4):
    best_miou = 0.0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for i, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            _, main_loss, aux_loss = model(images, masks)
            loss = main_loss + aux_loss_weight * aux_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {avg_train_loss:.4f}")
        val_loss, val_acc, val_miou = evaluate(model, val_loader, device)
        print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, mIoU: {val_miou:.4f}")
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), "best_pspnet_offroad.pth")
            print(f"Saved best model with mIoU: {best_miou:.4f}")
    return best_miou

# ---------------------------
# Main Execution
# ---------------------------

if __name__ == "__main__":
    # Fixed parameters
    dataset_dir = "/home/cloli/experimentation/cascade-road-segmentation/src/data/offroad-dataset"  # Change to your dataset directory path.
    target_size = (473, 473)
    batch_size = 4
    num_epochs = 50
    base_lr = 0.01
    momentum = 0.9
    weight_decay = 0.0001
    aux_weight = 0.4
    zoom_factor = 8
    pretrained_weights = "/home/cloli/experimentation/cascade-road-segmentation/src/models/pspnet_train_epoch_200.pth"  # Path to your pretrained PSPNet weights; if not found, training starts from scratch.
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set random seed for reproducibility.
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Create dataset objects.
    train_dataset = OffRoadSmallDataset(dataset_dir, split="train", target_size=target_size, augment=True, binary=True)
    val_dataset = OffRoadSmallDataset(dataset_dir, split="val", target_size=target_size, augment=False, binary=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Build PSPNet.
    model = PSPNet(layers=50, bins=(1,2,3,6), dropout=0.1, classes=2,
                   zoom_factor=zoom_factor, use_ppm=True, pretrained=False)
    
    # Load pretrained weights if available.
    if os.path.exists(pretrained_weights):
        checkpoint = torch.load(pretrained_weights, map_location=device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            ckpt = remove_module_prefix(checkpoint["state_dict"])
            print("Loaded pretrained weights from checkpoint.")
        else:
            ckpt = remove_module_prefix(checkpoint)
            print("Loaded pretrained weights.")
        for key in list(ckpt.keys()):
            if key.startswith("cls.4") or key.startswith("aux.4"):
                del ckpt[key]
        model.load_state_dict(ckpt, strict=False)
        print("Loaded pretrained weights (excluding classifier layers).")
    else:
        print("Pretrained weights not found. Training from scratch.")
    
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)
    
    print("Starting training...")
    best_miou = train_model(model, train_loader, val_loader, optimizer, num_epochs, device, aux_loss_weight=aux_weight)
    val_loss, val_acc, val_miou = evaluate(model, val_loader, device)
    print(f"Final Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, mIoU: {val_miou:.4f}")
    torch.save(model.state_dict(), "finetuned_pspnet_offroad.pth")
    print("Training complete. Model saved to finetuned_pspnet_offroad.pth")
