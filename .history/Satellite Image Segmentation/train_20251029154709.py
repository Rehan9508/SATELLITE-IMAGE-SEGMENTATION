import argparse
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.models import resnet34, ResNet34_Weights


class DeepGlobeModel(nn.Module):
    def __init__(self, num_classes: int = 6):
        super().__init__()
        # Use torchvision resnet34 to avoid torch.hub and deprecation
        encoder = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        encoder.fc = nn.Identity()
        self.encoder = encoder

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
        # resnet34 with fc=Identity returns shape [B, 512] after avgpool
        feats = self.encoder(x)  # [B, 512]
        feats = feats.view(feats.size(0), 512, 1, 1)
        out = self.decoder(feats)
        return out


class DeepGlobeDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, transform: Optional[transforms.Compose] = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))

        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.NEAREST),
            transforms.PILToTensor(),  # uint8 [1,H,W]
        ])

    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, idx: int):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform is not None:
            image = self.transform(image)

        mask = self.mask_transform(mask).squeeze(0).long()  # [H,W], int64
        return image, mask


def build_dataloader(image_dir: str, mask_dir: str, batch_size: int, shuffle: bool) -> DataLoader:
    tfm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    dataset = DeepGlobeDataset(image_dir, mask_dir, tfm)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_training(
    train_image_dir: str,
    train_mask_dir: str,
    val_image_dir: Optional[str],
    val_mask_dir: Optional[str],
    num_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    output_path: str,
    device: torch.device,
):
    model = DeepGlobeModel(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader = build_dataloader(train_image_dir, train_mask_dir, batch_size, shuffle=True)
    val_loader = None
    if val_image_dir and val_mask_dir and os.path.isdir(val_image_dir) and os.path.isdir(val_mask_dir):
        val_loader = build_dataloader(val_image_dir, val_mask_dir, batch_size, shuffle=False)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch + 1}/{epochs} - train loss: {avg_loss:.4f}")

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device)
                    masks = masks.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()
            val_avg = val_loss / max(len(val_loader), 1)
            print(f"           val loss: {val_avg:.4f}")

    torch.save(model.state_dict(), output_path)
    print(f"Saved model to {output_path}")


def run_dry_run(num_classes: int, epochs: int, batch_size: int, lr: float, device: torch.device) -> None:
    model = DeepGlobeModel(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dummy_images = torch.randn(batch_size, 3, 256, 256, device=device)
    dummy_masks = torch.randint(0, num_classes, (batch_size, 256, 256), device=device)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(dummy_images)
        loss = criterion(outputs, dummy_masks)
        loss.backward()
        optimizer.step()
        print(f"Dry-run epoch {epoch + 1}/{epochs} - loss: {loss.item():.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DeepGlobe-like segmentation model without Jupyter")
    parser.add_argument("--train-image-dir", type=str, default="", help="Path to training images directory")
    parser.add_argument("--train-mask-dir", type=str, default="", help="Path to training masks directory")
    parser.add_argument("--val-image-dir", type=str, default="", help="Optional path to validation images directory")
    parser.add_argument("--val-mask-dir", type=str, default="", help="Optional path to validation masks directory")
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="deepglobe_model.pth")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--dry-run", action="store_true", help="Run on random tensors (no data required)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")

    if args.dry_run:
        run_dry_run(args.num_classes, args.epochs, args.batch_size, args.lr, device)
        return

    if not os.path.isdir(args.train_image_dir) or not os.path.isdir(args.train_mask_dir):
        raise SystemExit(
            "Training directories not found. Provide --train-image-dir and --train-mask-dir with valid folders "
            "or use --dry-run to verify the pipeline without data."
        )

    run_training(
        train_image_dir=args.train_image_dir,
        train_mask_dir=args.train_mask_dir,
        val_image_dir=args.val_image_dir if args.val_image_dir else None,
        val_mask_dir=args.val_mask_dir if args.val_mask_dir else None,
        num_classes=args.num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_path=args.output,
        device=device,
    )


if __name__ == "__main__":
    main()

 
