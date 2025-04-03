import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

print(torch.__version__)


def convert_pth_to_onnx():
    checkpoint = "models/sam2.1_hiera_tiny.pt"
    model_cfg = "sam2/sam2/sam2_hiera_t.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        x = torch.randn(3, 256, 256)
        predictor.set_image(x)
        input_point = np.array([[500, 375]])
        input_label = np.array([1])
        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

    # Convert to ONNX

    # torch.onnx.export(model, x, f"{model_name}.onnx", verbose=True)
    # onnx.save(
    #     onnx.shape_inference.infer_shapes(onnx.load(f"{model_name}.onnx")),
    #     f"{model_name}.onnx",
    # )
