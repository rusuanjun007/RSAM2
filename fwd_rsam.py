import time

import numpy as np
import torch
from PIL import Image

from gemini2 import Gemini2
from rsam2_video_predictor import build_rsam2_video_predictor


def test_rsam2_fps(sam2_checkpoint, model_cfg):
    # Build the SAM2 camera predictor
    color_image = Image.open("imgs/1920x1080.jpg")

    predictor = build_rsam2_video_predictor(
        model_cfg,
        sam2_checkpoint,
    )
    inference_state = predictor.init_state(
        color_image, original_height=1080, original_width=1920
    )

    frames_processed = 0
    video_segments = {}
    global_start_time = time.time()

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # Step 1: Capture and annotate the first frame
        predictor.add_new_points_or_box(
            inference_state,
            frame_idx=0,
            obj_id=1,
            points=[[500, 500]],
            labels=[1],
        )
        predictor.propagate_in_video_preflight(
            inference_state
        )  # Consolidate annotations

        while frames_processed < 1000:
            frames_processed += 1
            frame_idx, obj_ids, masks = predictor.process_latest_frame(
                inference_state, color_image
            )
            # Save the segmentation results
            video_segments[frames_processed] = {
                out_obj_id: (masks[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(obj_ids)
            }

    # Calculate global elapsed time
    global_elapsed_time = time.time() - global_start_time
    fps = frames_processed / global_elapsed_time
    print(f"Average sam2-real-time FPS: {fps:.2f}")


def test_rsam2_camera_fps(sam2_checkpoint, model_cfg, verbose=False):
    """
    Test the FPS of the SAM2 model on a video.
    :param sam2_checkpoint: Path to the SAM2 checkpoint file.
    :param model_cfg: Path to the SAM2 model configuration file.
    """

    # Build the SAM2 camera predictor
    predictor = build_rsam2_video_predictor(model_cfg, sam2_checkpoint)

    gemini2 = Gemini2()
    gemini2.init_camera()

    if_init = False
    frames_processed = 0
    video_segments = {}

    # Record the global start time
    global_start_time = time.time()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        while frames_processed < 1000:
            time.sleep(0.01)
            frame_data = gemini2.get_latest_frame()
            if frame_data is None:
                continue
            color_image = frame_data["color_image"]
            depth_data = frame_data["depth_data"]
            frames_processed += 1

            if True:
                if not if_init:
                    # Step 1: Capture and annotate the first frame
                    inference_state = predictor.init_state(
                        Image.fromarray(color_image),
                        original_height=1080,
                        original_width=1920,
                    )
                    points = np.array([[1920 / 2, 1080 / 2]], dtype=np.float32)
                    predictor.add_new_points_or_box(
                        inference_state,
                        frame_idx=0,
                        obj_id=1,
                        points=points,
                        labels=np.array([1], np.int32),
                    )
                    # Consolidate annotations
                    predictor.propagate_in_video_preflight(inference_state)
                    if_init = True

                    # Display the annotation frame
                    if verbose:
                        gemini2.visualise_frame(color_image, depth_data, points=points)

                else:
                    frame_idx, obj_ids, masks = predictor.process_latest_frame(
                        inference_state, Image.fromarray(color_image)
                    )
                    # Save the segmentation results
                    video_segments[frames_processed] = {
                        out_obj_id: (masks[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(obj_ids)
                    }

                    # Display the segmentation results every few frames
                    if verbose:
                        gemini2.visualise_frame(
                            color_image,
                            depth_data,
                            points=points,
                            masks=masks,
                        )

    # Calculate global elapsed time
    global_elapsed_time = time.time() - global_start_time
    fps = frames_processed / global_elapsed_time
    print(f"Average camera FPS: {fps:.2f}")


if __name__ == "__main__":
    checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    # checkpoint = "./checkpoints/sam2.1_hiera_base_plus.pt"
    # model_cfg = "./configs/sam2.1/sam2.1_hiera_b+.yaml"

    # checkpoint = "./checkpoints/sam2.1_hiera_small.pt"
    # model_cfg = "./configs/sam2.1/sam2.1_hiera_s.yaml"

    # checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
    # model_cfg = "./configs/sam2.1/sam2.1_hiera_t.yaml"

    # test_rsam2_fps(checkpoint, model_cfg)
    test_rsam2_camera_fps(checkpoint, model_cfg, verbose=False)
