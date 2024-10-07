import os
import cv2
import PIL
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenet_v3_large
from torchvision.models.segmentation import (DeepLabV3_ResNet50_Weights, DeepLabV3_ResNet101_Weights, DeepLabV3_MobileNet_V3_Large_Weights)


def find_orange_line(image):
    # Convert the image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper borders for the orange color
    lower_border = np.array([15, 17, 100])
    upper_border = np.array([36, 77, 255])

    # Create a mask for the orange color
    mask = cv2.inRange(hsv_image, lower_border, upper_border)

    # Find contours (edges) of the orange line
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours, mask


def image_overlay(image, segmented_image):
    alpha = 1  # transparency for the original image
    beta = 0.8  # transparency for the segmentation map
    gamma = 0  # scalar added to each sum

    image = np.array(image)
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.addWeighted(image, alpha, segmented_image, beta, gamma, image)

    return image


def load_model(model_name: str):
    if model_name.lower() not in ("mobilenet", "resnet_50", "resnet_101"):
        raise ValueError("'model_name' should be one of ('mobilenet', 'resnet_50', 'resnet_101')")

    if model_name == "resnet_50":
        model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        transforms = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1.transforms()

    elif model_name == "resnet_101":
        model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
        transforms = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1.transforms()

    else:
        model = deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)
        transforms = DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1.transforms()

    model.eval()

    # Warmup run
    _ = model(torch.randn(1, 3, 520, 520))

    return model, transforms


def perform_inference(model_name: str, num_images=10, image_dir=None, save_images=False, device=None):
    if save_images:
        seg_map_save_dir = os.path.join("results", model_name, "segmentation_map")
        overlayed_save_dir = os.path.join("results", model_name, "overlayed_images")

        os.makedirs(seg_map_save_dir, exist_ok=True)
        os.makedirs(overlayed_save_dir, exist_ok=True)

    device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    model, transforms = load_model(model_name)
    model.to(device)

    # Load image handles for the validation set.
    with open(r"C:\Users\golan\AppData\Roaming\JetBrains\PyCharmCE2024.1\light-edit\output.txt") as f:
        val_set_image_names = f.read().split("\n")

    # Randomly select 'num_images' from the whole set for inference.
    selected_images = np.random.choice(val_set_image_names, num_images, replace=False)

    # Iterate over selected images
    for img_handle in selected_images:

        # Load and pre-process image.
        image_name = img_handle + ".jpg"
        image_path = os.path.join(image_dir, image_name)
        img_raw = PIL.Image.open(image_path).convert("RGB")
        img_cv = cv2.imread(image_path)

        # Find the orange line
        contours, mask = find_orange_line(img_cv)

        # Draw contours on the original image
        if contours:
            cv2.drawContours(img_cv, contours, -1, (0, 255, 0), 3)

        # Display the mask and the image with detected contours
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Orange Mask")
        plt.imshow(mask, cmap='gray')

        plt.subplot(1, 2, 2)
        plt.title("Detected Orange Line")
        plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

        plt.show()

    return


ROOT_raw_image_directory = r"C:\Users\golan\AppData\Roaming\JetBrains\PyCharmCE2024.1\light-edit\JPEGImages"

model_name = 'resnet_50'  # "mobilenet", "resnet_50", resnet_101
num_images = 4
save = False

perform_inference(
    model_name=model_name,
    num_images=num_images,
    save_images=save,
    image_dir=ROOT_raw_image_directory
)

'''with open('output.txt', 'w') as file:
    for i in range(300):
        file.write(f'output_{i:03}\n')'''