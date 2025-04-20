import os
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor


def show_mask(mask, ax, obj_id=None, random_color=False):
    """Show the mask on the given axis.

    Args:
        mask (torch.Tensor): The mask to show.
        ax (matplotlib axis): The matplotlib axis to show the mask.
        obj_id (int, optional): The object id of the mask. Defaults to None.
        random_color (bool, optional): The color of the mask. Defaults to False.
    """
    # Generate a random color if random_color is True, otherwise use a default color
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])

    # Create an image of the mask with the specified color
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    # Display the mask image on the given axis
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    """Display points on a given matplotlib axis.

    Args:
        coords (np.ndarray): The coordinates of the points.
        labels (np.ndarray): The labels of the points.
        ax (matplotlib axis): The matplotlib axis to display the points.
        marker_size (int, optional): The size of the markers. Defaults to 200.
    """
    # Separate the positive and negative points
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]

    # Scatter plot for positive points
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )

    # Scatter plot for negative points
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def copy_images(image_path, video_dir, num_copies=300):
    """Copy one image N times with name 0.jpg, 1.jpg, ... to the video directory

    Args:
        image_path (str): The path to the image to copy
        video_dir (str): The directory to copy the images to
    """
    # Check if the video directory exists, if not create it
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    # Copy and save it 100 times with different names
    img = Image.open(image_path)
    for i in range(num_copies):
        img.save(os.path.join(video_dir, f"{i}.jpg"))


def test_sam2_video_fps(checkpoint, model_cfg, verbose=False):
    """Test the SAM2 video FPS.
    Args:
        checkpoint (str): The path to the checkpoint file.
        model_cfg (str): The path to the model configuration file.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
    """
    frames_processed = 1000

    # Load the model.
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, vos_optimized=False)

    # Print the total number of parameters in the model
    pytorch_total_params = sum(p.numel() for p in predictor.parameters())
    print(f"Total number of parameters: {pytorch_total_params / 10**6:.2f}M")

    image_path = "imgs/1920x1080.jpg"
    video_dir = "./videos/1920x1080"
    copy_images(image_path, video_dir, frames_processed)

    # scan all the JPEG frame names in this directory
    frame_names = [
        p
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]

    # Sort the frame names by their frame index
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # Test the inference time for all images in the video directory
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # initialize the inference state
        inference_state = predictor.init_state(video_dir)

        # Annotation for the first frame

        ann_frame_idx = 0

        # Give a unique id to each object we interact with (it can be any integers)
        ann_obj_id = 1
        points = np.array([[250, 1000], [300, 400], [310, 420]], dtype=np.float32)
        # Set input prompt points and labels
        # label 1 indicates a positive click (to add a region)
        # label 0 indicates a negative click (to remove a region)
        labels = np.array([1, 1, 0], np.int32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )

        ann_obj_id = 7425
        points = np.array([[52, 52], [53, 53]], dtype=np.float32)
        labels = np.array([1, 0], np.int32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )

        ann_obj_id = 7425
        points = np.array([[54, 54], [55, 55]], dtype=np.float32)
        labels = np.array([0, 1], np.int32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
            clear_old_points=False,
        )

        ann_frame_idx = 1
        ann_obj_id = 666
        points = np.array([[75, 75], [76, 76]], dtype=np.float32)
        labels = np.array([0, 1], np.int32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )

        ann_obj_id = 7425
        points = np.array([[40, 40], [41, 41]], dtype=np.float32)
        labels = np.array([0, 0], np.int32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )

        # Display the annotation frame
        if verbose:
            plt.figure(figsize=(9, 6))
            plt.title(f"frame {ann_frame_idx}")
            plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
            show_points(points, labels, plt.gca())
            show_mask(
                (out_mask_logits[0] > 0.0).cpu().numpy(),
                plt.gca(),
                obj_id=out_obj_ids[0],
            )
            plt.show()
            plt.close("all")

        video_segments = {}
        # Record the global start time
        global_start_time = time.time()

        # Propagate the annotation to the rest of the video
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state
        ):
            # Save the segmentation results
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

    # Calculate global elapsed time
    global_elapsed_time = time.time() - global_start_time

    # Calculate true FPS based on total time
    fps = frames_processed / global_elapsed_time
    print(f"Average FPS: {fps:.2f}")

    # Display the segmentation results every few frames
    if verbose:
        vis_frame_stride = 1
        plt.close("all")
        for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
            plt.figure(figsize=(6, 4))
            plt.title(f"frame {out_frame_idx}")
            plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
                plt.show()


if __name__ == "__main__":
    checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "./configs/sam2.1/sam2.1_hiera_l.yaml"

    # checkpoint = "./checkpoints/sam2.1_hiera_base_plus.pt"
    # model_cfg = "./configs/sam2.1/sam2.1_hiera_b+.yaml"

    # checkpoint = "./checkpoints/sam2.1_hiera_small.pt"
    # model_cfg = "./configs/sam2.1/sam2.1_hiera_s.yaml"

    # checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
    # model_cfg = "./configs/sam2.1/sam2.1_hiera_t.yaml"

    # Test the FPS of the SAM2 model
    test_sam2_video_fps(checkpoint, model_cfg, verbose=True)
