import os
import cv2
import numpy as np

def extract_features_from_filename(filename):
    parts = filename.split('__')
    if len(parts) != 2:
        return None
    id, details = parts
    gender, handtype, fingertype, finger_ext = details.split('_')
    return {
        'id': id,
        'gender': gender,
        'hand_type': handtype,
        'finger_type': fingertype,
    }

def get_filenames_from_folder(folder_path):
    file_names = []
    for file_name in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            file_names.append(file_name)
    return file_names

def resize_image(image_path, width, height):
    image = cv2.imread(image_path)
    if image is not None:
        return cv2.resize(image, (width, height))
    else:
        print(f"Unable to read image: {image_path}")

def enhance_contrast(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray_image)
    return equalized_image

def noise_reduction(image, kernel_size=(5, 5)):
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
    return blurred_image

def read_and_preprocess_images(folder_path, target_size=(128, 128)):
    images = []
    labels = []
    images_with_their_labels = {}
    file_names = get_filenames_from_folder(folder_path)
    for name in file_names:
        features = extract_features_from_filename(name)
        if features:
            id = features['id']
            image_path = os.path.join(folder_path, name)
            image = resize_image(image_path, target_size[0], target_size[1])
            if image is not None:
                image = enhance_contrast(image)
                image = noise_reduction(image)
                images_with_their_labels[id] = np.array(image) / 255.0
    return images, labels, images_with_their_labels

def load_and_preprocess_images(directory, target_size=(128, 128)):
    images, labels, images_with_their_labels = read_and_preprocess_images(directory, target_size)
    return images, labels, images_with_their_labels
