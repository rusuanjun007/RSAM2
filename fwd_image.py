import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def show_mask(mask, ax, random_color=False, borders=True):
    """
    Show the mask on the given axis.

    Args:
        mask (torch.Tensor): The mask to show.
        ax (matplotlib axis): The matplotlib axis to show the mask.
        random_color (bool, optional): The color of the mask. Defaults to False.
        borders (bool, optional): Show the borders of the mask. Defaults to True.
    """
    # Generate a random color if random_color is True, otherwise use a default color
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])

    # Get the height and width of the mask
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)

    # Create an image of the mask with the specified color
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    # If borders are to be shown, find and draw contours
    if borders:
        import cv2

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Smooth the contours
        contours = [
            cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours
        ]

        # Draw the contours on the mask image
        mask_image = cv2.drawContours(
            mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2
        )

    # Display the mask image on the given axis
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    """display points on a given matplotlib axis

    Args:
        coords (np.ndarray): The coordinates of the points
        labels (np.ndarray): The labels of the points
        ax (matplotlib axis): The matplotlib axis to display the points
        marker_size (int, optional): The size of the markers. Defaults to 375.
    """
    # Separate positive and negative points based on labels
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


def show_box(box, ax):
    """display a bounding box on a given matplotlib axis

    Args:
        box (np.ndarray): The bounding box coordinates in the format [x0, y0, x1, y1]
        ax (matplotlib axis): The matplotlib axis to display the bounding box.
    """
    # Extract the coordinates of the box
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]

    # Add a rectangle patch to the axis
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def show_masks(
    image,
    masks,
    scores,
    point_coords=None,
    box_coords=None,
    input_labels=None,
    borders=True,
):
    """Display multiple masks on an image

    Args:
        image (np.ndarray): The image to display the masks on
        masks (List[np.ndarray]): The masks to display
        scores (List[float]): The scores of the masks
        point_coords (np.ndarray, optional): The coordinates of the points. Defaults to None.
        box_coords (np.ndarray, optional): The coordinates of the bounding boxes. Defaults to None.
        input_labels (np.ndarray, optional): The labels of the points. Defaults to None.
        borders (bool, optional): Show the borders of the masks. Defaults to True.
    """
    # Iterate over each mask and its corresponding score
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)

        # Display the mask on the image
        show_mask(mask, plt.gca(), borders=borders)

        # If point coordinates are provided, display the points
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())

        # If box coordinates are provided, display the box
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())

        # If there are multiple scores, add a title with the mask number and score
        if len(scores) > 1:
            plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)

        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    # Load the model.
    model_name = "sam2.1_hiera_large"
    checkpoint = f"./checkpoints/{model_name}.pt"
    model_cfg = "./configs/sam2.1/sam2.1_hiera_l.yaml"

    sam2_model = build_sam2(model_cfg, checkpoint)
    # x = torch.randn(3, 256, 256)

    # Convert to ONNX
    # torch.onnx.export(sam2_model, x, f"{model_name}.onnx", verbose=True)
    # onnx.save(
    #     onnx.shape_inference.infer_shapes(onnx.load(f"{model_name}.onnx")),
    #     f"{model_name}.onnx",
    # )

    predictor = SAM2ImagePredictor(sam2_model)

    # Load the raw image and display it.
    img = Image.open("imgs/1920x1080.jpg")
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis("on")
    plt.show()

    # Set prompt points and labels.
    point_coords = np.array([[250.0, 1000.0]])
    point_labels = np.array([1.0])

    # Display the image with the prompt points.
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    show_points(point_coords, point_labels, plt.gca())
    plt.axis("on")
    plt.show()

    # Test single image inference time for 300 times.
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        start_t = torch.cuda.Event(enable_timing=True)
        end_t = torch.cuda.Event(enable_timing=True)

        for i in range(300):
            start_t.record()

            # Forward pass.
            predictor.set_image(img)
            masks, scores, logits = predictor.predict(point_coords, point_labels)

            end_t.record()
            torch.cuda.synchronize()
            print(f"Time propagation frame {i}: {start_t.elapsed_time(end_t):.3f} ms")

    # Display the masks on the image.
    show_masks(
        img,
        masks,
        scores,
        point_coords=point_coords,
        input_labels=point_labels,
        borders=True,
    )
