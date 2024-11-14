import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from typing import List, Tuple, Dict, Optional, Union
import logging
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class RecognitionResult:
    """Data class to store letter recognition results."""

    letter: str
    confidence: float
    bounding_box: Tuple[int, int, int, int]
    image: np.ndarray


@dataclass
class WordRecognitionResult:
    """Data class to store word recognition results."""

    word: str
    letter_results: List[RecognitionResult]
    confidence: float
    original_image: np.ndarray
    processed_image: np.ndarray


class KyrgyzCNN(torch.nn.Module):
    """
    CNN architecture for Kyrgyz character recognition.

    The network consists of three convolutional blocks followed by
    a classifier network. Each conv block increases the feature depth
    while reducing spatial dimensions.
    """

    def __init__(self, num_classes: int = 36):
        """
        Initialize the CNN architecture.

        Args:
            num_classes: Number of Kyrgyz letters to classify
        """
        super(KyrgyzCNN, self).__init__()

        # Feature extraction layers
        self.features = torch.nn.Sequential(
            # First conv block: 3x128x128 -> 16x63x63
            torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
            # Second conv block: 16x63x63 -> 32x30x30
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
            # Third conv block: 32x30x30 -> 64x14x14
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
        )

        # Flatten layer
        self.flatten = torch.nn.Flatten()

        # Classifier layers
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(64 * 14 * 14, 2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, 128, 128)

        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = self.flatten(x)
        return self.classifier(x)


class ImagePreprocessor:
    """
    Handles image preprocessing for letter segmentation.
    """

    def __init__(self):
        """Initialize the preprocessor with default parameters."""
        self.blur_kernel = (5, 5)
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for letter segmentation.

        Args:
            image: Input BGR image

        Returns:
            Preprocessed binary image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Remove noise
        denoised = cv2.medianBlur(binary, 3)

        # Connect broken letter parts
        morph = cv2.morphologyEx(
            denoised, cv2.MORPH_CLOSE, self.morph_kernel, iterations=1
        )

        return morph


class LetterSegmenter:
    """
    Handles letter segmentation from word images.
    """

    def __init__(self):
        """Initialize segmenter with default parameters."""
        self.min_letter_width = 15
        self.max_letter_width = 100
        self.min_letter_height = 15
        self.max_letter_height = 100
        self.min_area = 100

    def _filter_contours(
        self, contours: List[np.ndarray], image_shape: Tuple[int, int]
    ) -> List[np.ndarray]:
        """
        Filter contours based on size and position constraints.

        Args:
            contours: List of contours to filter
            image_shape: (height, width) of the image

        Returns:
            Filtered list of contours
        """
        filtered = []
        image_height, image_width = image_shape

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)

            # Apply size constraints
            if (
                self.min_letter_width < w < self.max_letter_width
                and self.min_letter_height < h < self.max_letter_height
                and area > self.min_area
            ):

                # Check if contour is not on the image border
                if (
                    x > 2
                    and y > 2
                    and x + w < image_width - 2
                    and y + h < image_height - 2
                ):
                    filtered.append(contour)

        return filtered

    def segment_letters(
        self, image: np.ndarray, binary: np.ndarray
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Segment individual letters from the word image.

        Args:
            image: Original BGR image
            binary: Preprocessed binary image

        Returns:
            List of (letter_image, bounding_box) tuples
        """
        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter contours
        filtered_contours = self._filter_contours(contours, binary.shape)

        # Sort contours left to right
        sorted_contours = sorted(
            filtered_contours, key=lambda c: cv2.boundingRect(c)[0]
        )

        letters = []
        for contour in sorted_contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Extract letter with padding
            pad = 10
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(binary.shape[1], x + w + pad)
            y2 = min(binary.shape[0], y + h + pad)

            letter_img = image[y1:y2, x1:x2]

            # Ensure consistent size
            letter_img = cv2.resize(letter_img, (128, 128))

            letters.append((letter_img, (x, y, w, h)))

        return letters


class LetterRecognizer:
    """
    Handles recognition of individual letters using the CNN model,
    with case detection as a post-processing step.
    """

    def __init__(self, model_path: str):
        """
        Initialize the recognizer.

        Args:
            model_path: Path to the trained model weights
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Base classes (uppercase)
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

        # Create case mappings
        self.case_mappings = {upper: upper.lower() for upper in self.classes}

        self.model = self._load_model(model_path)
        self.transform = self._get_transforms()

    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load and prepare the CNN model."""
        model = KyrgyzCNN(len(self.classes))  # Original 36 classes
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        return model

    def _get_transforms(self) -> transforms.Compose:
        """Get image transformations for the model."""
        return transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    def _detect_case(self, letter_image: np.ndarray) -> bool:
        """
        Detect if the letter should be uppercase based on image characteristics.

        Args:
            letter_image: Letter image in BGR format

        Returns:
            bool: True if uppercase, False if lowercase
        """
        # Convert to grayscale if needed
        if len(letter_image.shape) == 3:
            gray = cv2.cvtColor(letter_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = letter_image

        # Get height of actual letter content
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return True  # Default to uppercase if no contours found

        # Get the main contour
        main_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)

        # Calculate relative metrics
        image_height = gray.shape[0]
        relative_height = h / image_height
        relative_position = y / image_height
        relative_width = w / gray.shape[1]

        # Check position and size characteristics
        is_uppercase = False

        # Height-based check
        if relative_height > 0.65:  # Letter takes up significant vertical space
            is_uppercase = True
        # Position-based check
        elif relative_position < 0.3:  # Letter starts high in the image
            is_uppercase = True
        # Size-ratio check
        elif relative_width > 0.4:  # Letter is relatively wide
            is_uppercase = True
        # Default for ambiguous cases
        else:
            is_uppercase = relative_height > 0.5

        return is_uppercase

    def recognize_letter(
        self, letter_image: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> RecognitionResult:
        """
        Recognize a single letter with case detection.

        Args:
            letter_image: Letter image in BGR format
            bbox: (x, y, w, h) bounding box of the letter

        Returns:
            RecognitionResult containing the prediction
        """
        # Convert to PIL
        pil_image = Image.fromarray(cv2.cvtColor(letter_image, cv2.COLOR_BGR2RGB))

        # Transform and predict
        tensor = self.transform(pil_image)
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

            # Get the predicted uppercase letter
            predicted_upper = self.classes[predicted_idx.item()]
            confidence_score = confidence.item()

            # Detect if the letter should be uppercase or lowercase
            is_uppercase = self._detect_case(letter_image)

            # Convert case if needed
            predicted_letter = (
                predicted_upper if is_uppercase else self.case_mappings[predicted_upper]
            )

            # Debug information
            logger.debug(f"Original prediction: {predicted_upper}")
            logger.debug(f"Detected case: {'upper' if is_uppercase else 'lower'}")
            logger.debug(f"Final prediction: {predicted_letter}")

        return RecognitionResult(
            letter=predicted_letter,
            confidence=confidence_score,
            bounding_box=bbox,
            image=letter_image,
        )

    def get_case_variants(self, letter: str) -> Tuple[str, str]:
        """
        Get both uppercase and lowercase variants of a letter.

        Args:
            letter: Input letter

        Returns:
            Tuple of (uppercase, lowercase) variants
        """
        if letter.upper() in self.classes:
            upper = letter.upper()
            lower = self.case_mappings[upper]
        else:
            upper = letter
            lower = letter
        return upper, lower


class WordRecognizer:
    """
    Main class for word recognition, combining preprocessing,
    segmentation, and recognition steps.
    """

    def __init__(self, model_path: str):
        """
        Initialize the word recognizer.

        Args:
            model_path: Path to the trained model weights
        """
        self.preprocessor = ImagePreprocessor()
        self.segmenter = LetterSegmenter()
        self.recognizer = LetterRecognizer(model_path)

    def recognize_word(self, image_path: str) -> WordRecognitionResult:
        """
        Recognize a word from an image.

        Args:
            image_path: Path to the word image

        Returns:
            WordRecognitionResult containing predictions and visualization
        """
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        binary = self.preprocessor.preprocess(image)

        # Segment letters
        letter_segments = self.segmenter.segment_letters(image, binary)

        # Recognize each letter
        results = []
        word = ""
        total_confidence = 0

        for letter_image, bbox in letter_segments:
            result = self.recognizer.recognize_letter(letter_image, bbox)
            results.append(result)
            word += result.letter
            total_confidence += result.confidence

            # Draw on image
            x, y, w, h = bbox
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"{result.letter} ({result.confidence:.2f})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        avg_confidence = total_confidence / len(results) if results else 0

        return WordRecognitionResult(
            word=word,
            letter_results=results,
            confidence=avg_confidence,
            original_image=image,
            processed_image=binary,
        )


class WordRecognitionVisualizer:
    """
    Handles visualization of word recognition results.
    """

    @staticmethod
    def visualize_results(
        result: WordRecognitionResult, save_path: Optional[str] = None
    ):
        """
        Visualize word recognition results.

        Args:
            result: WordRecognitionResult to visualize
            save_path: Optional path to save visualization
        """
        plt.figure(figsize=(15, 10))

        # Original image with predictions
        plt.subplot(2, 1, 1)
        plt.imshow(cv2.cvtColor(result.original_image, cv2.COLOR_BGR2RGB))
        plt.title(
            f"Recognized Word: {result.word} (Confidence: {result.confidence:.2f})"
        )
        plt.axis("off")

        # Individual letters
        n_letters = len(result.letter_results)
        plt.subplot(2, 1, 2)
        for i, letter_result in enumerate(result.letter_results):
            plt.subplot(2, n_letters, n_letters + i + 1)
            plt.imshow(cv2.cvtColor(letter_result.image, cv2.COLOR_BGR2RGB))
            plt.title(
                f"{letter_result.letter}\n{letter_result.confidence:.2f}", fontsize=8
            )
            plt.axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


class BatchWordRecognizer:
    """
    Handles batch processing of multiple word images.
    """

    def __init__(self, model_path: str, output_dir: Optional[str] = None):
        """
        Initialize batch processor.

        Args:
            model_path: Path to the trained model weights
            output_dir: Directory to save results and visualizations
        """
        self.recognizer = WordRecognizer(model_path)
        self.visualizer = WordRecognitionVisualizer()
        self.output_dir = Path(output_dir) if output_dir else None

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_folder(
        self, folder_path: str, file_pattern: str = "*.png"
    ) -> List[Dict]:
        """
        Process all images in a folder.

        Args:
            folder_path: Path to folder containing images
            file_pattern: Pattern to match image files

        Returns:
            List of recognition results
        """
        folder_path = Path(folder_path)
        results = []

        # Find all matching image files
        image_paths = list(folder_path.glob(file_pattern))
        logger.info(f"Found {len(image_paths)} images to process")

        for image_path in image_paths:
            try:
                # Process image
                result = self.recognizer.recognize_word(str(image_path))

                # Save visualization if output directory is specified
                if self.output_dir:
                    viz_path = self.output_dir / f"{image_path.stem}_result.png"
                    self.visualizer.visualize_results(result, str(viz_path))

                # Store results
                results.append(
                    {
                        "image_path": str(image_path),
                        "word": result.word,
                        "confidence": result.confidence,
                        "letter_confidences": [
                            r.confidence for r in result.letter_results
                        ],
                    }
                )

                logger.info(
                    f"Processed {image_path.name}: {result.word} "
                    f"(confidence: {result.confidence:.2f})"
                )

            except Exception as e:
                logger.error(f"Error processing {image_path.name}: {str(e)}")
                continue

        return results


class AccuracyEvaluator:
    """
    Evaluates recognition accuracy against ground truth.
    """

    @staticmethod
    def calculate_word_accuracy(
        predicted: str, actual: str
    ) -> Tuple[float, List[bool]]:
        """
        Calculate word recognition accuracy.

        Args:
            predicted: Predicted word
            actual: Actual word

        Returns:
            Tuple of (accuracy_percentage, list_of_correct_letters)
        """
        if not predicted or not actual:
            return 0.0, []

        # Pad shorter string with spaces
        max_len = max(len(predicted), len(actual))
        predicted = predicted.ljust(max_len)
        actual = actual.ljust(max_len)

        # Compare letters
        correct_letters = [p == a for p, a in zip(predicted, actual)]
        accuracy = sum(correct_letters) / max_len * 100

        return accuracy, correct_letters

    @staticmethod
    def evaluate_batch_results(
        results: List[Dict], ground_truth: Dict[str, str]
    ) -> Dict:
        """
        Evaluate batch recognition results.

        Args:
            results: List of recognition results
            ground_truth: Dictionary mapping image paths to correct words

        Returns:
            Dictionary containing evaluation metrics
        """
        total_words = 0
        total_letters = 0
        correct_words = 0
        correct_letters = 0

        word_accuracies = []
        letter_accuracies = []

        for result in results:
            image_path = result["image_path"]
            if image_path not in ground_truth:
                continue

            predicted = result["word"]
            actual = ground_truth[image_path]

            # Calculate accuracy
            accuracy, correct = AccuracyEvaluator.calculate_word_accuracy(
                predicted, actual
            )

            # Update statistics
            total_words += 1
            total_letters += len(actual)
            correct_words += predicted == actual
            correct_letters += sum(correct)

            word_accuracies.append(accuracy)
            letter_accuracies.extend(correct)

        # Calculate metrics
        avg_word_accuracy = (
            sum(word_accuracies) / len(word_accuracies) if word_accuracies else 0
        )
        word_accuracy = (correct_words / total_words * 100) if total_words else 0
        letter_accuracy = (
            (correct_letters / total_letters * 100) if total_letters else 0
        )

        return {
            "total_words": total_words,
            "correct_words": correct_words,
            "word_accuracy": word_accuracy,
            "avg_word_accuracy": avg_word_accuracy,
            "letter_accuracy": letter_accuracy,
            "word_accuracies": word_accuracies,
            "letter_accuracies": letter_accuracies,
        }


def process_folder_with_evaluation():
    """Process a folder of images and evaluate accuracy."""
    # Configuration
    model_path = "./results/final_model.pth"
    input_folder = "./data/test"
    output_folder = "./results/recognition_output"
    ground_truth_path = "./data/test/ground_truth.txt"

    try:
        # Load ground truth
        ground_truth = {}
        with open(ground_truth_path, "r", encoding="utf-8") as f:
            for line in f:
                image_path, word = line.strip().split("\t")
                ground_truth[image_path] = word

        # Initialize batch processor
        processor = BatchWordRecognizer(model_path, output_folder)

        # Process images
        results = processor.process_folder(input_folder)

        # Evaluate results
        evaluator = AccuracyEvaluator()
        metrics = evaluator.evaluate_batch_results(results, ground_truth)

        # Print results
        logger.info("\nRecognition Results:")
        logger.info(f"Total words processed: {metrics['total_words']}")
        logger.info(f"Correct words: {metrics['correct_words']}")
        logger.info(f"Word accuracy: {metrics['word_accuracy']:.2f}%")
        logger.info(f"Average word accuracy: {metrics['avg_word_accuracy']:.2f}%")
        logger.info(f"Letter accuracy: {metrics['letter_accuracy']:.2f}%")

    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        raise


def main():
    """Main function demonstrating the usage of the word recognition system."""
    # Initialize recognizer
    model_path = "./results/final_model.pth"
    recognizer = WordRecognizer(model_path)
    visualizer = WordRecognitionVisualizer()

    try:
        # Process single image
        image_path = "./data/raw/cyrilic_words/combined_word.png"
        logger.info(f"Processing image: {image_path}")

        result = recognizer.recognize_word(image_path)
        logger.info(
            f"Recognition result: {result.word} (confidence: {result.confidence:.2f})"
        )

        # Visualize results
        visualizer.visualize_results(result)

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise


if __name__ == "__main__":
    # For single image processing
    main()

    # For batch processing with evaluation
    # process_folder_with_evaluation()
