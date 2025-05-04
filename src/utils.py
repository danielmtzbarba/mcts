import numpy as np
import matplotlib.pyplot as plt


def rgb_to_semantic_mask(rgb_image):
    """
    Convert RGB frame from simulator into a 5-channel semantic mask.

    Args:
        rgb_image (np.ndarray): (H, W, 3) RGB image.

    Returns:
        np.ndarray: (5, H, W) semantic mask (binary channels).
    """
    # Normalize input if needed (assume 0-255)
    rgb = rgb_image.astype(np.uint8)
    h, w, _ = rgb.shape
    mask = np.zeros((6, h, w), dtype=np.float32)

    # Define color thresholds (loose matching in case of slight rendering variation)
    white = np.array([255, 255, 255])
    red = np.array([255, 0, 0])
    blue = np.array([0, 7, 175])
    green = np.array([0, 255, 0])
    gray = np.array([150, 150, 150])  # Adjusted if your gray is different
    grays = np.array([220, 220, 220])  # Adjusted if your gray is different

    # Create masks
    is_white = np.all(rgb == white, axis=-1)
    is_red = np.all(rgb == red, axis=-1)
    is_blue = np.all(rgb == blue, axis=-1)
    is_green = np.all(rgb == green, axis=-1)
    is_gray = np.all(rgb == gray, axis=-1)
    is_grays = np.all(rgb == grays, axis=-1)

    # Fill semantic channels
    mask[0, is_gray] = 1.0  # Non-drivable
    mask[1, is_white] = 1.0  # Drivable
    mask[2, is_grays] = 1.0  # Non-drivable
    mask[3, is_blue] = 1.0  # Vehicle
    mask[4, is_red] = 1.0  # Pedestrian
    mask[5, is_green] = 1.0  # Checkpoint

    return mask


def plot_semantic_img(mask):
    """
    Plot the 6-channel semantic mask with correct colors.

    Args:
        mask (np.ndarray): (6, H, W) semantic mask
    """
    h, w = mask.shape[1], mask.shape[2]
    debug_img = np.zeros((h, w, 3), dtype=np.uint8)

    # Assign each class a color for visualization
    debug_img[mask[0] == 1] = [150, 150, 150]  # Non-drivable (dark gray)
    debug_img[mask[1] == 1] = [255, 255, 255]  # Drivable (white)
    debug_img[mask[2] == 1] = [220, 220, 220]  # Non-drivable (light gray)
    debug_img[mask[3] == 1] = [0, 7, 165]  # Vehicle (blueish)
    debug_img[mask[4] == 1] = [255, 0, 0]  # Pedestrian (red)
    debug_img[mask[5] == 1] = [0, 255, 0]  # Checkpoint (green)

    plt.figure(figsize=(6, 6))
    plt.imshow(debug_img)
    plt.title("Semantic Mask Visualization (6 classes)")
    plt.axis("off")
    plt.show()


def plot_semantic_masks(mask):
    """
    Plot the 6-channel semantic mask with correct colors for each channel separately.

    Args:
        mask (np.ndarray): (6, H, W) semantic mask
    """
    h, w = mask.shape[1], mask.shape[2]

    # Define colors for each mask
    colors = [
        [150, 150, 150],  # Non-drivable (dark gray)
        [255, 255, 255],  # Drivable (white)
        [220, 220, 220],  # Non-drivable (light gray)
        [0, 7, 165],  # Vehicle (blueish)
        [255, 0, 0],  # Pedestrian (red)
        [0, 255, 0],  # Checkpoint (green)
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))  # 2 rows, 3 columns
    axes = axes.flatten()

    for i in range(6):
        # Create an RGB image for this mask
        mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        mask_rgb[mask[i] == 1] = colors[i]

        axes[i].imshow(mask_rgb)
        axes[i].set_title(f"Class {i}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()
