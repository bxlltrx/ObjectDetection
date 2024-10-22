import cv2
import numpy as np
import os

def create_and_fill_orange_line_mask(image_path, lower_border, upper_border, save_path):
    """
    Создаёт маску для оранжевой линии и закрашивает её на оригинальном изображении.
    """
    # Загрузка изображения
    image = cv2.imread(image_path)

    # Преобразование изображения в HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Создание бинарной маски на основе диапазона HSV
    mask = cv2.inRange(hsv_image, lower_border, upper_border)

    # Морфологические операции для улучшения маски
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Создание цветного изображения, где линия закрашена (например, зелёным цветом)
    colored_image = image.copy()
    colored_image[mask > 0] = (0, 255, 0)  # Закраска зелёным

    # Сохранение маски
    mask_filename = os.path.basename(image_path).replace(".jpg", ".png")
    mask_path = os.path.join(save_path, mask_filename)
    cv2.imwrite(mask_path, mask)

    print(f"Маска сохранена: {mask_path}")


# Директории
images_dir = "JPEGImages"  # Папка с изображениями
masks_dir = "masks"         # Папка для сохранения масок и закрашенных изображений

# HSV-границы для оранжевой линии
lower_border = np.array([15, 17, 100])
upper_border = np.array([36, 77, 255])

# Убедитесь, что директория для масок существует
os.makedirs(masks_dir, exist_ok=True)

# Создание масок и закрашенных изображений для всех файлов
for image_name in os.listdir(images_dir):
    if image_name.endswith(".jpg"):
        image_path = os.path.join(images_dir, image_name)
        create_and_fill_orange_line_mask(image_path, lower_border, upper_border, masks_dir)
