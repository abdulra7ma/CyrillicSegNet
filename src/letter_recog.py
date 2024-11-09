import torch
import torchvision.transforms as transforms
from PIL import Image
from PIL import Image
import cv2
from model import Kyrgyz
import numpy as np
import torch.nn as nn

# Define the Kyrgyz model class
# Define device configuration (use GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model
model = Kyrgyz()
model.load_state_dict(torch.load("../results/best_model.pth", map_location=device))
model.to(device)
model.eval()  # Set the model to evaluation mode

# Define transformations for input image
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),  # Resize the image to 128x128 pixels
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
        ),  # Normalize pixel values to [-1, 1]
    ]
)


def recognize_letter(image):
    """
    Recognize the letter in the given image.
    Args:
        image (PIL.Image or np.ndarray): Input image.
    Returns:
        str: Predicted letter.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert('RGB')
    
    # Apply transformations
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    # Predict the label
    with torch.no_grad():
        output = model(image)
        predicted_idx = output.argmax(1).item()
    
    # Mapping predicted index to the corresponding label
    classes = [
        'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я', 'Ң', 'Ү', 'Ө', 'Ё'
    ]
    
    predicted_label = classes[predicted_idx]
    return predicted_label

def recognize_word(image_path):
    """
    Recognize the word in the given image by splitting it into individual letters.
    Args:
        image_path (str): Path to the input image.
    Returns:
        str: Predicted word.
    """
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to get a binary representation
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours of the letters
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours from left to right
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    predicted_word = ""

    # Loop through each contour to recognize each letter
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 5 and h > 5:  # Filter small noise
            letter_image = gray[y:y+h, x:x+w]
            letter_image = cv2.cvtColor(letter_image, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
            letter_image = Image.fromarray(letter_image)  # Convert to PIL Image
            predicted_letter = recognize_letter(letter_image)
            predicted_word += predicted_letter

    return predicted_word

# Test the function
image_path = "../data/raw/cyrilic_words/test/test0.png"  # Replace with the path to your jpg image

# predicted_label = recognize_letter(image_path)
# print(f"Predicted Label: {predicted_label}")

predicted_word = recognize_word(image_path)
print(f"Predicted Word: {predicted_word}")
