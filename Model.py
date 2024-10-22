import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50

model = deeplabv3_resnet50(aux_loss=True)
model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1))
model.load_state_dict(torch.load("deeplabv3_orange_line.pth", map_location="cpu"))
model.eval()

# Инференс
def infer(image_path):
    image = torchvision.io.read_image(image_path).float() / 255.0
    input_tensor = image.unsqueeze(0).to("cpu")

    with torch.no_grad():
        output = model(input_tensor)['out'][0]
        mask = output.argmax(0).byte().cpu().numpy()

    plt.imshow(mask, cmap='gray')
    plt.show()


infer("JPEGImages/output_120.jpg")