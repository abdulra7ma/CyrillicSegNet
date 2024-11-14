# Import all required libraries
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForMaskedLM
from typing import List, Tuple, Optional
import logging
from torchvision import transforms
from IPython.display import display, Image as IPythonImage


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreprocessingModule:
    """Handles image preprocessing steps."""

    @staticmethod
    def preprocess_image(image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing steps to the input image.

        Args:
            image: Input image in BGR format
        Returns:
            Preprocessed image
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian Blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11,
                2,
            )

            # Apply morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

            return morphed
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise


class SegmentationEngine:
    """Handles letter segmentation from the preprocessed image."""

    @staticmethod
    def segment_letters(
        preprocessed_image: np.ndarray,
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Segment individual letters from the preprocessed image.

        Args:
            preprocessed_image: Binary preprocessed image
        Returns:
            List of tuples containing (letter_image, bounding_box)
        """
        try:
            # Find contours
            contours, _ = cv2.findContours(
                preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Sort contours left to right
            contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

            letter_segments = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                # Filter out noise based on size
                if 15 < w < 100 and 15 < h < 100:
                    # Extract letter and add padding
                    letter_image = preprocessed_image[y : y + h, x : x + w]
                    letter_image = cv2.copyMakeBorder(
                        letter_image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255
                    )

                    # Resize to standard size
                    letter_image = cv2.resize(letter_image, (28, 28))
                    letter_segments.append((letter_image, (x, y, w, h)))

            return letter_segments
        except Exception as e:
            logger.error(f"Error in segmentation: {str(e)}")
            raise


class CNNRecognitionModel:
    """Handles letter recognition using the CNN model."""

    def __init__(self, model_path: str):
        """Initialize the CNN model."""
        self.classes = [
            "А",
            "Б",
            "В",
            "Г",
            "Д",
            "Е",
            "Ж",
            "З",
            "И",
            "Й",
            "К",
            "Л",
            "М",
            "Н",
            "О",
            "П",
            "Р",
            "С",
            "Т",
            "У",
            "Ф",
            "Х",
            "Ц",
            "Ч",
            "Ш",
            "Щ",
            "Ъ",
            "Ы",
            "Ь",
            "Э",
            "Ю",
            "Я",
            "Ң",
            "Ү",
            "Ө",
            "Ё",
        ]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = self.get_transforms()
        self.model = self.load_model(model_path)

    def create_model(self) -> nn.Module:
        class KyrgyzCyrillicNet(nn.Module):
            def __init__(self):
                super(KyrgyzCyrillicNet, self).__init__()
                self.features = nn.Sequential(
                    # Assuming input size (batch, 3, 112, 112)
                    nn.Conv2d(3, 16, kernel_size=3),  # -> (batch, 16, 110, 110)
                    nn.MaxPool2d(kernel_size=2),  # -> (batch, 16, 55, 55)
                    nn.Conv2d(16, 32, kernel_size=3),  # -> (batch, 32, 53, 53)
                    nn.MaxPool2d(kernel_size=2),  # -> (batch, 32, 26, 26)
                    nn.Conv2d(32, 64, kernel_size=3),  # -> (batch, 64, 24, 24)
                    nn.MaxPool2d(kernel_size=2),  # -> (batch, 64, 12, 12)
                )
                self.flatten = (
                    nn.Flatten()
                )  # Flatten the output for the fully connected layers

                # Fully connected layers
                self.classifier = nn.Sequential(
                    nn.Linear(64 * 15 * 15, 2048),  # Adjusted dense layer
                    nn.ReLU(),
                    nn.Dropout(p=0.5),
                    nn.Linear(2048, 512),
                    nn.ReLU(),
                    nn.Dropout(p=0.5),
                    nn.Linear(512, 1036),  # Adjust output classes here if needed
                )

            def forward(self, x):
                x = self.features(x)
                x = self.flatten(x)
                logits = self.classifier(x)
                return logits

        return KyrgyzCyrillicNet()

    def get_transforms(self):
        """Get image transformations matching the verified input size."""
        return transforms.Compose(
            [
                transforms.Resize((112, 112)),  # Changed input size to 112x112
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def load_model(self, model_path: str) -> nn.Module:
        """Load the trained CNN model with verified architecture."""
        try:
            model = self.create_model()
            state_dict = torch.load(model_path, map_location=self.device)

            # Debug information
            dummy_input = torch.randn(1, 3, 112, 112)
            features_output = model.features(dummy_input)
            flattened = features_output.view(1, -1)
            logger.info(f"Input shape: {dummy_input.shape}")
            logger.info(f"Features output shape: {features_output.shape}")
            logger.info(f"Flattened size: {flattened.shape[1]}")

            model.load_state_dict(state_dict)
            model = model.to(self.device)
            model.eval()

            return model

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def recognize_letter(self, letter_image: np.ndarray) -> Tuple[str, float]:
        """Recognize a single letter using the verified model."""
        try:
            if letter_image.dtype != np.uint8:
                letter_image = (letter_image * 255).astype(np.uint8)

            pil_image = Image.fromarray(letter_image).convert("RGB")
            tensor_image = self.transform(pil_image)
            tensor_image = tensor_image.unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(tensor_image)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)

                predicted_letter = self.classes[predicted_idx.item()]
                confidence_score = confidence.item()

                return predicted_letter, confidence_score

        except Exception as e:
            logger.error(f"Error in letter recognition: {str(e)}")
            raise


def inspect_model_state(model_path):
    """
    Utility function to inspect the saved model state.
    """
    state_dict = torch.load(model_path, map_location="cpu")
    print("Type of loaded file:", type(state_dict))

    if isinstance(state_dict, dict):
        print("\nKeys in the loaded file:")
        for key in state_dict.keys():
            print(f"- {key}")

        if "state_dict" in state_dict:
            print("\nKeys in state_dict:")
            for key in state_dict["state_dict"].keys():
                print(f"- {key}")


def print_model_structure(model_path: str):
    """Print the structure of the saved model."""
    state_dict = torch.load(model_path, map_location="cpu")
    print("\nModel state dict structure:")
    for key, value in state_dict.items():
        print(f"{key}: {value.shape}")


class LLMCorrectionModel:
    """Handles word correction using a Language Model."""

    def __init__(self, model_name: str = "bert-base-multilingual-cased"):
        """
        Initialize the LLM correction model.

        Args:
            model_name: Name of the pretrained model to use
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.eval()

    def correct_word(self, predicted_word: str) -> str:
        """
        Correct the predicted word using the language model.

        Args:
            predicted_word: Initial word prediction
        Returns:
            Corrected word
        """
        try:
            # For simplicity, we'll just return the word for now
            # In a real implementation, you would:
            # 1. Use the LLM to validate the word
            # 2. Get potential corrections
            # 3. Return the most likely correction
            return predicted_word
        except Exception as e:
            logger.error(f"Error in word correction: {str(e)}")
            raise


class KyrgyzWordRecognizer:
    """Main class that orchestrates the entire recognition process."""

    def __init__(self, cnn_model_path: str):
        """
        Initialize the word recognizer with all necessary components.

        Args:
            cnn_model_path: Path to the trained CNN model
        """
        self.preprocessor = PreprocessingModule()
        self.segmenter = SegmentationEngine()
        self.recognizer = CNNRecognitionModel(cnn_model_path)
        self.corrector = LLMCorrectionModel()

    def recognize_word(self, image_path: str, visualize: bool = True) -> str:
        """
        Recognize a word from an image.

        Args:
            image_path: Path to the input image
            visualize: Whether to show visualization of the process
        Returns:
            Recognized word
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")

            # Preprocess
            preprocessed = self.preprocessor.preprocess_image(image)

            # Segment letters
            letter_segments = self.segmenter.segment_letters(preprocessed)

            # Recognize letters
            predicted_word = ""
            confidences = []
            letter_results = []

            for letter_image, bbox in letter_segments:
                letter, confidence = self.recognizer.recognize_letter(letter_image)
                predicted_word += letter
                confidences.append(confidence)
                letter_results.append((letter_image, letter, confidence, bbox))

            # Correct word using LLM
            corrected_word = self.corrector.correct_word(predicted_word)

            # Visualize if requested
            if visualize:
                self._visualize_results(
                    image, letter_results, predicted_word, corrected_word
                )

            return corrected_word

        except Exception as e:
            logger.error(f"Error in word recognition: {str(e)}")
            raise

    def _visualize_results(
        self,
        original_image: np.ndarray,
        letter_results: List[Tuple],
        predicted_word: str,
        corrected_word: str,
    ) -> None:
        """Visualize the recognition results."""
        # Create a copy for visualization
        viz_image = original_image.copy()

        # Draw bounding boxes and predictions
        for _, letter, confidence, (x, y, w, h) in letter_results:
            cv2.rectangle(viz_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                viz_image,
                f"{letter} ({confidence:.2f})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        # Display results
        plt.figure(figsize=(15, 5))

        # Original image with annotations
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(viz_image, cv2.COLOR_BGR2RGB))
        plt.title("Detected Letters")
        plt.axis("off")

        # Individual letters
        plt.subplot(1, 2, 2)
        plt.text(
            0.5,
            0.7,
            f"Predicted: {predicted_word}",
            ha="center",
            va="center",
            fontsize=12,
        )
        plt.text(
            0.5,
            0.3,
            f"Corrected: {corrected_word}",
            ha="center",
            va="center",
            fontsize=12,
        )
        plt.axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    recognizer = KyrgyzWordRecognizer("./results/final_model.pth")
    result = recognizer.recognize_word("./data/raw/cyrilic_words/combined_word.png")
    print(f"Recognition result: {result}")
