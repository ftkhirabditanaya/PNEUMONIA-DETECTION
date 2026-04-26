import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from src.models.densenet_attention import DenseNetAttention

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = DenseNetAttention(
        num_classes=2,
        dropout_rate=0.4,
        freeze_backbone=False
    ).to(DEVICE)

    checkpoint = torch.load(
        "models/phase1_best.pth",
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def generate_gradcam(image_path):
    model = load_model()

    img = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # Use dicts instead of lists to avoid stale/duplicate captures
    store = {"features": None, "gradients": None}

    def forward_hook(module, input, output):
        store["features"] = output

    def backward_hook(module, grad_in, grad_out):
        store["gradients"] = grad_out[0]

    target_layer = model.features.denseblock4

    #  Save handles so hooks can be cleanly removed after use
    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    # Enable grad computation even in eval mode
    with torch.enable_grad():
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()  #  .item() avoids tensor indexing issues

        model.zero_grad()
        output[0, pred_class].backward()

    #  Remove hooks after use to prevent memory leaks or duplicate captures
    fwd_handle.remove()
    bwd_handle.remove()

    grad = store["gradients"]
    fmap = store["features"]

    #  Guard against None in case hooks didn't fire
    if grad is None or fmap is None:
        raise RuntimeError("Hooks did not capture features/gradients. Check target_layer path.")

    weights = torch.mean(grad, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * fmap, dim=1).squeeze()

    cam = torch.relu(cam)
    cam = cam / (cam.max() + 1e-8)
    cam = cam.cpu().detach().numpy()

    cam = cv2.resize(cam, (224, 224))

    img_np = np.array(img.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    superimposed = (heatmap_rgb * 0.4 + img_np * 0.6).astype(np.uint8)

    plt.imshow(superimposed)
    plt.title(f"Grad-CAM — Predicted class: {pred_class}")
    plt.axis("off")
    plt.savefig("outputs/gradcam_pneumonia_result.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    generate_gradcam(r"data/raw/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg")