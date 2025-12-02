# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 16:28:59 2025

@author: Rahil

"""
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import optuna
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import all_gather_object

# --- Dataset Loader Using data.txt ---
class SegmentationDataset(Dataset):
    def __init__(self, root_dir, data_txt, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        with open(os.path.join(root_dir, data_txt), "r") as f:
            self.pairs = [line.strip().split() for line in f.readlines()]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_name, mask_name = self.pairs[idx]
        image = Image.open(os.path.join(self.root_dir, img_name)).convert("RGB")
        mask = Image.open(os.path.join(self.root_dir, mask_name)).convert("L")
        if self.transform:
            image, mask = self.transform(image, mask)
        return image, mask, os.path.basename(mask_name)  # keep filename for prediction

# --- Data Augmentation and Preprocessing ---
class RandomTransform:
    def __init__(self, size=(256, 256)):
        self.size = size

    def __call__(self, image, mask):
        image = TF.resize(image, self.size)
        mask = TF.resize(mask, self.size, interpolation=Image.NEAREST)

        if torch.rand(1) > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if torch.rand(1) > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        image = TF.to_tensor(image)
        mask = torch.from_numpy(np.array(mask)).long()
        mask = (mask > 128).long()  # binary mask threshold

        return image, mask

# --- Metrics ---
def pixel_accuracy(output, mask):
    preds = (torch.sigmoid(output) > 0.5).long()
    correct = (preds == mask).float()
    return correct.sum() / correct.numel()

def compute_r2(output, mask):
    preds = torch.sigmoid(output).detach().cpu().numpy().flatten()
    targets = mask.detach().cpu().numpy().flatten()
    return r2_score(targets, preds)

# --- Train ---
def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs):
    scaler = GradScaler()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_r2 = 0.0
        for images, masks, _ in tqdm(train_loader, desc=f"Epoch {epoch+1} Training", disable=(dist.get_rank() != 0)):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)['out'].squeeze(1)
                loss = criterion(outputs, masks.float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            running_r2 += compute_r2(outputs, masks)

        avg_loss = running_loss / len(train_loader)
        avg_r2 = running_r2 / len(train_loader)

        if dist.get_rank() == 0:
            print(f"Epoch {epoch+1} Training Loss: {avg_loss:.4f}, R2: {avg_r2:.4f}")

        val_loss, val_r2, val_acc = validate_model(model, val_loader, criterion, device)
        if dist.get_rank() == 0:
            print(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}, R2: {val_r2:.4f}, Pixel Acc: {val_acc:.4f}")

# --- Validation ---
def validate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_r2 = 0.0
    running_acc = 0.0
    with torch.no_grad():
        for images, masks, _ in tqdm(val_loader, desc="Validation", disable=(dist.get_rank() != 0)):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)['out'].squeeze(1)
            loss = criterion(outputs, masks.float())
            running_loss += loss.item()
            running_r2 += compute_r2(outputs, masks)
            running_acc += pixel_accuracy(outputs, masks).item()
    return (running_loss / len(val_loader),
            running_r2 / len(val_loader),
            running_acc / len(val_loader))

# --- Save Predictions ---
def save_predictions(model, loader, device, out_dir="predictions"):
    model.eval()

    # Ensure output directory is created once
    if dist.get_rank() == 0:
        os.makedirs(out_dir, exist_ok=True)
    dist.barrier()  # Sync all processes

    gathered_filenames = []
    gathered_preds = []

    local_preds = []
    local_filenames = []

    with torch.no_grad():
        for images, _, filenames in tqdm(loader, desc=f"Rank {dist.get_rank()} Saving", disable=(dist.get_rank() != 0)):
            images = images.to(device)
            outputs = model(images)['out'].squeeze(1)
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy().astype(np.uint8) * 255

            for i in range(len(filenames)):
                local_preds.append(preds[i])
                local_filenames.append(filenames[i])

    # Gather predictions and filenames to rank 0
    all_gather_object(gathered_preds, local_preds)
    all_gather_object(gathered_filenames, local_filenames)

    if dist.get_rank() == 0:
        flat_preds = [p for rank_preds in gathered_preds for p in rank_preds]
        flat_names = [n for rank_names in gathered_filenames for n in rank_names]

        for pred, filename in zip(flat_preds, flat_names):
            Image.fromarray(pred).save(os.path.join(out_dir, filename))


# --- Optuna Objective ---
def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])

    transform = RandomTransform(size=(256, 256))
    dataset = SegmentationDataset(root_dir="luad_batch_1",
                                  data_txt="label.txt",
                                  transform=transform)

    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(dataset, val_idx)

    train_sampler = DistributedSampler(train_ds)
    val_sampler = DistributedSampler(val_ds, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, sampler=val_sampler, num_workers=4, pin_memory=True)

    device = torch.device(f'cuda:{dist.get_rank()}')
    model = deeplabv3_resnet50(pretrained_backbone=True, num_classes=1).to(device)
    model = DDP(model, device_ids=[dist.get_rank()])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=3)

    _, val_r2, _ = validate_model(model, val_loader, criterion, device)
    return val_r2

def main():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    torch.backends.cudnn.benchmark = True

    if rank == 0:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10)
        print("Best hyperparameters:", study.best_params)
    dist.barrier()

    # Broadcast best params from rank 0 to all other ranks (simple way)
    if rank == 0:
        obj_list = [study.best_params]
    else:
        obj_list = [None]

    dist.broadcast_object_list(obj_list, src=0)
    best_params = obj_list[0]

    transform = RandomTransform(size=(256, 256))
    dataset = SegmentationDataset(root_dir="luad_batch_1",
                                  data_txt="label.txt",
                                  transform=transform)

    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=best_params["batch_size"], sampler=sampler, num_workers=4, pin_memory=True)

    model = deeplabv3_resnet50(pretrained_backbone=True, num_classes=1).to(device)
    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
    criterion = nn.BCEWithLogitsLoss()

    train_model(model, loader, loader, optimizer, criterion, device, epochs=10)

    # Save predictions on validation set
    val_dataset = SegmentationDataset(root_dir="luad_validation_1",
                                     data_txt="label.txt",
                                     transform=transform)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=best_params["batch_size"], sampler=val_sampler, num_workers=4, pin_memory=True)

    save_predictions(model, val_loader, device, out_dir="predictions")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()

