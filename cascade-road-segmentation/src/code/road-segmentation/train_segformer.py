
import pkgutil
if not hasattr(pkgutil, 'ImpImporter'):
    pkgutil.ImpImporter = None

import os
import glob
import random
import time
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

# -----------------------------
# Losses & Metrics
# -----------------------------
def dice_loss_fn(logits, labels, smooth=1e-6):
    B, C, h, w = logits.shape
    H, W = labels.shape[-2:]
    if (h, w) != (H, W):
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
    probs = F.softmax(logits, dim=1)[:,1]
    labels_f = labels.float()
    p = probs.view(B, -1)
    l = labels_f.view(B, -1)
    inter = (p * l).sum(dim=1)
    union = p.sum(dim=1) + l.sum(dim=1)
    dice = (2*inter + smooth) / (union + smooth)
    return (1 - dice).mean()

def compute_metrics(logits, labels):
    # logits: (B,2,H',W'), labels: (B,H,W)
    logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
    preds = logits.argmax(dim=1).cpu().numpy()
    labs  = labels.cpu().numpy()
    ious = []
    for cls in (0,1):
        pi, ti = preds==cls, labs==cls
        inter = (pi & ti).sum()
        uni   = (pi | ti).sum()
        ious.append(inter/uni if uni>0 else 1.0)
    miou = sum(ious)/2
    tp = ((preds==1)&(labs==1)).sum()
    fp = ((preds==1)&(labs!=1)).sum()
    fn = ((preds!=1)&(labs==1)).sum()
    tn = ((preds==0)&(labs==0)).sum()
    iou_cls   = tp/(tp+fp+fn) if (tp+fp+fn)>0 else 1.0
    dice      = 2*tp/(2*tp+fp+fn)   if (2*tp+fp+fn)>0 else 1.0
    precision = tp/(tp+fp)         if (tp+fp)>0    else 1.0
    recall    = tp/(tp+fn)         if (tp+fn)>0    else 1.0
    pixel_acc = (tp+tn)/(tp+tn+fp+fn) if (tp+tn+fp+fn)>0 else 1.0
    return {
        "miou":miou,
        "iou_cls":iou_cls,
        "dice":dice,
        "precision":precision,
        "recall":recall,
        "pixel_acc":pixel_acc
    }

# -----------------------------
# Dataset definitions
# -----------------------------
class OffRoadSegDataset(Dataset):
    def __init__(self, root, split="train", size=(512,512), augment=False, processor=None):
        self.size = size
        self.augment = augment and split=="train"
        self.processor = processor
        txts = glob.glob(os.path.join(root,"**","*small.txt"), recursive=True)
        train_imgs, train_lbls, test_imgs, test_lbls = [], [], [], []
        for t in txts:
            base = os.path.dirname(t)
            for ln in open(t):
                fn = ln.strip()
                img = os.path.join(base, fn+".jpg")
                msk = glob.glob(os.path.join(base, fn+"*label_raw.png"))[0]
                if "train" in msk:
                    train_imgs.append(img); train_lbls.append(msk)
                else:
                    test_imgs.append(img); test_lbls.append(msk)
        combined = list(zip(train_imgs,train_lbls))
        random.seed(0); random.shuffle(combined)
        train_imgs,train_lbls = zip(*combined)
        split_idx = int(len(train_imgs)*0.1)
        self.splits = {
            "train": (train_imgs[split_idx:], train_lbls[split_idx:]),
            "val":   (train_imgs[:split_idx], train_lbls[:split_idx]),
            "test":  (test_imgs, test_lbls)
        }
        self.imgs, self.lbls = self.splits[split]
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        im = Image.open(self.imgs[i]).convert("RGB")
        ma = Image.open(self.lbls[i])
        ma = Image.fromarray((np.array(ma)==1).astype(np.uint8))
        if self.augment:
            if random.random()>0.5:
                im,ma = im.transpose(Image.FLIP_LEFT_RIGHT), ma.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random()>0.5:
                ang = random.uniform(-15,15)
                im = im.rotate(ang, resample=Image.BILINEAR)
                ma = ma.rotate(ang, resample=Image.NEAREST)
            if random.random()>0.5:
                im = ImageEnhance.Color(im).enhance(random.uniform(0.8,1.2))
                im = ImageEnhance.Brightness(im).enhance(random.uniform(0.8,1.2))
                im = ImageEnhance.Contrast(im).enhance(random.uniform(0.8,1.2))
            if random.random()>0.3:
                im = im.filter(ImageFilter.GaussianBlur(radius=random.uniform(0,1.5)))
            if random.random()>0.3:
                arr = np.array(im).astype(np.int16)
                noise = np.random.normal(0,5,arr.shape).astype(np.int16)
                arr = np.clip(arr+noise,0,255).astype(np.uint8)
                im = Image.fromarray(arr)
        enc = self.processor(images=im, segmentation_maps=ma,
                             return_tensors="pt", size=self.size)
        return {"pixel_values": enc["pixel_values"].squeeze(0),
                "labels":       enc["labels"].squeeze(0)}

class RoboFlowSegDataset(Dataset):
    POS = {10,11,14,15,17}
    def __init__(self, root, split="train", size=(512,512), augment=False, processor=None):
        self.size = size
        self.augment = augment and split=="train"
        self.processor = processor
        base = os.path.join(root, f"train_{split}") if os.path.isdir(os.path.join(root, f"train_{split}")) else root
        imgs = sorted(glob.glob(os.path.join(base, "*.jpg")))
        self.pairs = [(i, i.replace(".jpg","_mask.png")) for i in imgs]
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        im_p, ms_p = self.pairs[i]
        im = Image.open(im_p).convert("RGB")
        ma = Image.fromarray(np.isin(np.array(Image.open(ms_p)), list(self.POS)).astype(np.uint8))
        if self.augment:
            if random.random()>0.5:
                ang = random.uniform(-15,15)
                im = im.rotate(ang,Image.BILINEAR)
                ma = ma.rotate(ang,Image.NEAREST)
        enc = self.processor(images=im, segmentation_maps=ma,
                             return_tensors="pt", size=self.size)
        return {"pixel_values": enc["pixel_values"].squeeze(0),
                "labels":       enc["labels"].squeeze(0)}

# -----------------------------
# Training loop
# -----------------------------
def train():
    off_dir    = "/home/cloli/experimentation/cascade-road-segmentation/src/data/offroad-dataset/off-road"
    robo_tr    = "/home/cloli/experimentation/cascade-road-segmentation/src/data/train/train_train"
    robo_val   = "/home/cloli/experimentation/cascade-road-segmentation/src/data/valid/train_val"
    device     = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    proc  = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
        num_labels=2, ignore_mismatched_sizes=True
    ).to(device)

    off_tr = OffRoadSegDataset(off_dir,   "train", augment=True,  processor=proc)
    off_vl = OffRoadSegDataset(off_dir,   "val",   augment=False, processor=proc)
    robo_t = RoboFlowSegDataset(robo_tr,  "train", augment=True,  processor=proc)
    robo_v = RoboFlowSegDataset(robo_val, "val",   augment=False, processor=proc)

    train_ds = ConcatDataset([off_tr, robo_t])
    val_ds   = ConcatDataset([off_vl, robo_v])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=8, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=8)

    # differential LR
    head_params, body_params = [], []
    for n,p in model.named_parameters():
        (head_params if "decode_head" in n else body_params).append(p)
    optimizer = AdamW([
        {"params": body_params, "lr": 6e-5},
        {"params": head_params, "lr": 6e-4}
    ], weight_decay=1e-2)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    ce = nn.CrossEntropyLoss(ignore_index=255, label_smoothing=0.1)

    best_miou, best_loss = 0, float("inf")
    miou_pat, loss_pat, PAT = 0, 0, 5

    metrics_hist = {"miou":[], "iou_cls":[], "dice":[], "precision":[], "recall":[], "pixel_acc":[]}
    train_losses, val_losses = [], []

    for epoch in range(1,51):
        # --- train ---
        model.train(); running_loss = 0
        for batch in tqdm(train_loader, desc=f"Train {epoch}/50"):
            pv, lbl = batch["pixel_values"].to(device), batch["labels"].to(device)
            logits = model(pixel_values=pv).logits
            up     = F.interpolate(logits, size=lbl.shape[-2:], mode="bilinear", align_corners=False)
            loss   = ce(up, lbl.long()) + dice_loss_fn(logits, lbl)
            loss.backward()
            optimizer.step(); optimizer.zero_grad()
            scheduler.step(epoch + len(train_loader)/len(train_loader))
            running_loss += loss.item()
        train_loss = running_loss/len(train_loader)
        train_losses.append(train_loss)

        # --- validate ---
        model.eval(); val_running=0
        sum_m = {k:0 for k in metrics_hist}
        all_probs, all_lbls = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f" Val  {epoch}/50"):
                pv, lbl = batch["pixel_values"].to(device), batch["labels"].to(device)
                logits = model(pixel_values=pv).logits
                up     = F.interpolate(logits, size=lbl.shape[-2:], mode="bilinear", align_corners=False)
                loss   = ce(up, lbl.long()) + dice_loss_fn(logits, lbl)
                val_running += loss.item()
                m = compute_metrics(up, lbl)
                # <-- only accumulate the keys we track -->
                for k,v in m.items():
                    if k in sum_m:
                        sum_m[k] += v
                all_probs.append(torch.softmax(up,1)[:,1].cpu().numpy().ravel())
                all_lbls.append(lbl.cpu().numpy().ravel())

        val_loss = val_running/len(val_loader)
        val_losses.append(val_loss)
        epoch_m = {k: sum_m[k]/len(val_loader) for k in sum_m}
        for k in metrics_hist:
            metrics_hist[k].append(epoch_m[k])

        # --- early stopping checks ---
        if epoch_m["miou"] > best_miou:
            best_miou, miou_pat = epoch_m["miou"], 0
            torch.save(model.state_dict(), "best_miou.pth")
        else:
            miou_pat += 1

        if val_loss < best_loss:
            best_loss, loss_pat = val_loss, 0
            torch.save(model.state_dict(), "best_loss.pth")
        else:
            loss_pat += 1

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_miou={epoch_m['miou']:.4f}")
        if miou_pat>=PAT and loss_pat>=PAT:
            print("Early stopping triggered.")
            break

    # --- PR curve at end ---
    all_probs = np.concatenate(all_probs)
    all_lbls  = np.concatenate(all_lbls)
    prec, rec, _ = precision_recall_curve(all_lbls, all_probs)
    ap = auc(rec, prec)
    plt.figure(); plt.plot(rec,prec,label=f"AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve"); plt.legend(); plt.grid(True)
    plt.savefig("pr_curve_improved.png")
    print("Training complete.")

if __name__=="__main__":
    train()
