import io
import argparse
from typing import Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse


class DeepGlobeModel(nn.Module):
    def __init__(self, num_classes: int = 6):
        super().__init__()
        from torchvision.models import resnet34, ResNet34_Weights

        encoder = resnet34(weights=None)
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
        x = self.encoder(x)
        x = x.view(x.size(0), 512, 1, 1)
        x = self.decoder(x)
        return x


def colorize_mask(mask: np.ndarray) -> Image.Image:
    unique_classes = np.unique(mask)
    rng = np.random.default_rng(42)
    colors = rng.integers(0, 255, size=(unique_classes.max() + 1, 3), dtype=np.uint8)
    rgb = colors[mask]
    return Image.fromarray(rgb, mode="RGB")


def build_preprocess():
    return T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])


def load_model(model_path: str, num_classes: int, device: torch.device) -> nn.Module:
    model = DeepGlobeModel(num_classes=num_classes).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def create_app(model_path: str, num_classes: int, device_str: str = "cpu") -> FastAPI:
    device = torch.device(device_str)
    model = load_model(model_path, num_classes, device)
    preprocess = build_preprocess()

    app = FastAPI(title="DeepGlobe Segmentation API")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/predict")
    async def predict(file: UploadFile = File(...), return_mask: Optional[bool] = False):
        try:
            content = await file.read()
            image = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file")

        orig_size = image.size[::-1]  # (H, W)
        tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Resize prediction back to original size
        pred_img = Image.fromarray(pred, mode="L").resize((orig_size[1], orig_size[0]), Image.NEAREST)

        if return_mask:
            png = to_png_bytes(pred_img)
            return StreamingResponse(io.BytesIO(png), media_type="image/png")

        colorized = colorize_mask(np.array(pred_img))
        png = to_png_bytes(colorized)
        return StreamingResponse(io.BytesIO(png), media_type="image/png")

    return app


def parse_args():
    p = argparse.ArgumentParser(description="Serve DeepGlobe model on localhost")
    p.add_argument("--model", required=True, help="Path to .pth model file")
    p.add_argument("--num-classes", type=int, default=6)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = create_app(args.model, args.num_classes, args.device)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


