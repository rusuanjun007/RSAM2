from collections import OrderedDict

import numpy as np
import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sam2.build_sam import _load_checkpoint
from sam2.sam2_video_predictor import SAM2VideoPredictor


def build_rsam2_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
):
    hydra_overrides = [
        "++model._target_=rsam2_video_predictor.RSAM2VideoPredictor",
    ]

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


class RSAM2VideoPredictor(SAM2VideoPredictor):
    """Subclass of SAM2VideoPredictor for real-time image stream processing."""

    @torch.inference_mode()
    def init_state(
        self,
        first_frame,
        original_height,
        original_width,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
    ):
        """Initialize an inference state with the first frame for real-time processing."""
        compute_device = self.device
        # Assume first_frame is resized to self.image_size
        first_frame = self.preprocess_image(first_frame)
        images = [first_frame.to(compute_device if not offload_video_to_cpu else "cpu")]
        inference_state = {
            "images": images,
            "num_frames": 1,
            "offload_video_to_cpu": offload_video_to_cpu,
            "offload_state_to_cpu": offload_state_to_cpu,
            "storage_device": torch.device("cpu")
            if offload_state_to_cpu
            else compute_device,
            "cached_features": {},
            "constants": {},
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            "output_dict_per_obj": {},
            "temp_output_dict_per_obj": {},
            "frames_tracked_per_obj": {},
            "video_height": original_height,
            "video_width": original_width,
            "device": compute_device,
        }
        # Cache image feature for frame 0
        self._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        return inference_state

    def preprocess_image(
        self, image, img_mean=(0.485, 0.456, 0.406), img_std=(0.229, 0.224, 0.225)
    ):
        """Preprocess the input image to the model's expected format."""
        # Resize and normalize the image
        image = np.array(image.resize((self.image_size, self.image_size)))
        if image.dtype == np.uint8:  # np.uint8 is expected for JPEG images
            image = image / 255.0
        else:
            raise RuntimeError(f"Unknown image dtype: {image.dtype}")
        image = torch.from_numpy(image).permute(2, 0, 1)
        # normalize by mean and std
        img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
        img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
        image -= img_mean
        image /= img_std

        # Add batch dimension
        # image = image.unsqueeze(0)

        return image

    @torch.inference_mode()
    def add_new_frame(self, inference_state, new_frame):
        """Add a new frame to the inference state for real-time processing."""
        device = (
            inference_state["device"]
            if not inference_state["offload_video_to_cpu"]
            else "cpu"
        )
        # Assume new_frame is resized to self.image_size
        new_frame = self.preprocess_image(new_frame)
        inference_state["images"].append(new_frame.to(device))
        inference_state["num_frames"] += 1

    @torch.inference_mode()
    def process_latest_frame(self, inference_state, new_frame):
        """Process the latest frame in the inference state and return the prediction mask."""
        self.add_new_frame(inference_state, new_frame)

        frame_idx = inference_state["num_frames"] - 1
        if frame_idx == 0:
            raise ValueError(
                "The first frame should be processed with user prompts via add_new_points_or_box or add_new_mask."
            )
        batch_size = self._get_obj_num(inference_state)
        obj_ids = inference_state["obj_ids"]
        pred_masks_per_obj = [None] * batch_size

        for obj_idx in range(batch_size):
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            if frame_idx in obj_output_dict["cond_frame_outputs"]:
                current_out = obj_output_dict["cond_frame_outputs"][frame_idx]
                device = inference_state["device"]
                pred_masks = current_out["pred_masks"].to(device, non_blocking=True)
                if self.clear_non_cond_mem_around_input:
                    self._clear_obj_non_cond_mem_around_input(
                        inference_state, frame_idx, obj_idx
                    )
            else:
                current_out, pred_masks = self._run_single_frame_inference(
                    inference_state=inference_state,
                    output_dict=obj_output_dict,
                    frame_idx=frame_idx,
                    batch_size=1,
                    is_init_cond_frame=False,
                    point_inputs=None,
                    mask_inputs=None,
                    reverse=False,
                    run_mem_encoder=True,
                )
                obj_output_dict["non_cond_frame_outputs"][frame_idx] = current_out
            inference_state["frames_tracked_per_obj"][obj_idx][frame_idx] = {
                "reverse": False
            }
            pred_masks_per_obj[obj_idx] = pred_masks

        all_pred_masks = (
            torch.cat(pred_masks_per_obj, dim=0)
            if len(pred_masks_per_obj) > 1
            else pred_masks_per_obj[0]
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, all_pred_masks
        )

        # Memory management: remove image if not a conditioning frame
        if all(
            frame_idx not in obj_output_dict["cond_frame_outputs"]
            for obj_output_dict in inference_state["output_dict_per_obj"].values()
        ):
            inference_state["images"][frame_idx] = None
            self._remove_old_frames(inference_state, frame_idx)

        return frame_idx, obj_ids, video_res_masks

    def _remove_old_frames(self, inference_state, current_frame_idx):
        """Efficiently remove old non-conditioning frames' outputs to save memory."""
        max_mem = max(self.num_maskmem, self.max_obj_ptrs_in_encoder) - 1
        min_frame_to_keep = current_frame_idx - max_mem
        for obj_idx in range(self._get_obj_num(inference_state)):
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            non_cond_frame_outputs = obj_output_dict["non_cond_frame_outputs"]
            # Remove old frames efficiently
            while non_cond_frame_outputs:
                frame, _ = next(iter(non_cond_frame_outputs.items()))
                if frame >= min_frame_to_keep:
                    break
                non_cond_frame_outputs.pop(frame)

    def _get_image_feature(self, inference_state, frame_idx, batch_size):
        """Compute the image features on a given frame, ensuring the image is available."""
        image, backbone_out = inference_state["cached_features"].get(
            frame_idx, (None, None)
        )
        if backbone_out is None:
            device = inference_state["device"]
            image = inference_state["images"][frame_idx]
            if image is None:
                raise ValueError(
                    f"Image at frame_idx {frame_idx} has been cleared and cannot be processed."
                )
            image = image.to(device).float().unsqueeze(0)
            backbone_out = self.forward_image(image)
            inference_state["cached_features"] = {frame_idx: (image, backbone_out)}
        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_backbone_out = {
            "backbone_fpn": [
                feat.expand(batch_size, -1, -1, -1)
                for feat in backbone_out["backbone_fpn"]
            ],
            "vision_pos_enc": [
                pos.expand(batch_size, -1, -1, -1)
                for pos in backbone_out["vision_pos_enc"]
            ],
        }
        features = self._prepare_backbone_features(expanded_backbone_out)
        features = (expanded_image,) + features
        return features
