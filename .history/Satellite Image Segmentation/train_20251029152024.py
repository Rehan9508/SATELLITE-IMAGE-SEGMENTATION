import argparse
import os
from typing import Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class DeepGlobeModel(nn.Module):
    def __init__(self, num_classes: int = 6):
        super().__init__()
        # Use torchvision resnet with explicit weights API to avoid deprecation
        # Lazy import to keep startup light if CUDA probes happen
        from torchvision.models import resnet34, ResNet34_Weights

        encoder = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        # Remove classification head so forward returns 512-d embedding after avgpool
        encoder.fc = nn.Identity()
        self.encoder = encoder

        # Decoder upsamples from 1x1 feature to 256x256 segmentation map
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(4, num_classes, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)              # (N, 512)
        x = x.view(x.size(0), 512, 1, 1)
        x = self.decoder(x)              # (N, C, 256, 256)
        return x


class DeepGlobeDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, transform: Optional[T.Compose] = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.image_filenames = sorted([f for f in os.listdir(image_dir) if not f.startswith('.')])
        self.mask_filenames = sorted([f for f in os.listdir(mask_dir) if not f.startswith('.')])

        if len(self.image_filenames) == 0:
            raise FileNotFoundError(f"No images found in {image_dir}")
        if len(self.mask_filenames) == 0:
            raise FileNotFoundError(f"No masks found in {mask_dir}")
        if len(self.image_filenames) != len(self.mask_filenames):
            print("Warning: image/mask counts differ; using min length")
            n = min(len(self.image_filenames), len(self.mask_filenames))
            self.image_filenames = self.image_filenames[:n]
            self.mask_filenames = self.mask_filenames[:n]

        self.mask_transform = T.Compose([
            T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST),
            T.PILToTensor(),  # uint8, shape (1, H, W)
        ])

    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, idx: int):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # class ids 0..K-1 expected

        if self.transform:
            image = self.transform(image)

        mask_tensor = self.mask_transform(mask).squeeze(0).long()
        return image, mask_tensor


def build_transforms() -> T.Compose:
    return T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepGlobe segmentation training")
    parser.add_argument("--train-image-dir", required=True)
    parser.add_argument("--train-mask-dir", required=True)
    parser.add_argument("--val-image-dir", required=False, default=None)
    parser.add_argument("--val-mask-dir", required=False, default=None)
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="deepglobe_model.pth")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]) 
    return parser.parse_args()


def get_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
    return total_loss / max(1, len(loader))


def main() -> None:
    args = parse_args()

    device = get_device(args.device)
    print(f"Using device: {device}")

    transform = build_transforms()
    train_ds = DeepGlobeDataset(args.train_image_dir, args.train_mask_dir, transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    val_loader = None
    if args.val_image_dir and args.val_mask_dir and os.path.isdir(args.val_image_dir) and os.path.isdir(args.val_mask_dir):
        val_ds = DeepGlobeDataset(args.val_image_dir, args.val_mask_dir, transform)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = DeepGlobeModel(num_classes=args.num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train = total_loss / max(1, len(train_loader))
        log = f"Epoch {epoch + 1}/{args.epochs} - train_loss: {avg_train:.4f}"
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device)
            log += f" - val_loss: {val_loss:.4f}"
        print(log)

    torch.save(model.state_dict(), args.out)
    print(f"Saved model to {args.out}")


if __name__ == "__main__":
    main()


