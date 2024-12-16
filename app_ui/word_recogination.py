import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
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
    Enhanced image preprocessor for handwritten Kyrgyz letter segmentation.
    """

    def __init__(self):
        """Initialize the preprocessor with optimized parameters."""
        self.blur_kernel = (3, 3)  # Reduced blur kernel size
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (2, 2)
        )  # Smaller kernel
        self.min_contour_area = 50  # Minimum area for noise filtering

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Enhanced preprocessing for handwritten letters.

        Args:
            image: Input BGR image

        Returns:
            Preprocessed binary image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Normalize brightness and contrast
        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        # Apply Gaussian blur to reduce noise while preserving edges
        blurred = cv2.GaussianBlur(normalized, self.blur_kernel, 0)

        # Use Otsu's thresholding for better binarization
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Clean up noise
        denoised = self._remove_small_noise(binary)

        # Enhance letter strokes
        enhanced = self._enhance_strokes(denoised)

        return enhanced

    def _remove_small_noise(self, binary: np.ndarray) -> np.ndarray:
        """Remove small noise components while preserving letter parts."""
        # Find all connected components
        nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        # Create output image
        cleaned = np.zeros_like(binary)

        # Skip background (label 0)
        for i in range(1, nlabels):
            if stats[i, cv2.CC_STAT_AREA] > self.min_contour_area:
                cleaned[labels == i] = 255

        return cleaned

    def _enhance_strokes(self, binary: np.ndarray) -> np.ndarray:
        """Enhance letter strokes while maintaining connectivity."""
        # Slight dilation to connect nearby components
        dilated = cv2.dilate(binary, self.morph_kernel, iterations=1)

        # Remove small holes
        filled = cv2.morphologyEx(
            dilated, cv2.MORPH_CLOSE, self.morph_kernel, iterations=1
        )

        return filled


class LetterSegmenter:
    """
    Enhanced letter segmenter for handwritten Kyrgyz text.
    """

    def __init__(self):
        """Initialize segmenter with optimized parameters."""
        self.min_letter_width = 10  # Reduced minimum width
        self.max_letter_width = 150  # Increased maximum width
        self.min_letter_height = 10  # Reduced minimum height
        self.max_letter_height = 150  # Increased maximum height
        self.min_area = 50  # Reduced minimum area
        self.overlap_threshold = 0.3  # Threshold for merging overlapping components

    def _merge_overlapping_components(
        self, contours: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Merge overlapping components that might be parts of the same letter."""
        if not contours:
            return []

        # Get bounding boxes
        bboxes = [cv2.boundingRect(c) for c in contours]
        merged = []
        used = set()

        for i, (x1, y1, w1, h1) in enumerate(bboxes):
            if i in used:
                continue

            merged_contour = contours[i]
            for j, (x2, y2, w2, h2) in enumerate(bboxes[i + 1 :], i + 1):
                if j in used:
                    continue

                # Check for horizontal proximity and vertical overlap
                horizontal_gap = abs((x1 + w1) - x2) if x2 > x1 else abs((x2 + w2) - x1)
                vertical_overlap = (min(y1 + h1, y2 + h2) - max(y1, y2)) / min(h1, h2)

                if horizontal_gap < 20 and vertical_overlap > self.overlap_threshold:
                    # Merge contours
                    merged_contour = np.concatenate([merged_contour, contours[j]])
                    used.add(j)
                    # Update bounding box
                    x = min(x1, x2)
                    y = min(y1, y2)
                    w = max(x1 + w1, x2 + w2) - x
                    h = max(y1 + h1, y2 + h2) - y
                    x1, y1, w1, h1 = x, y, w, h

            merged.append(merged_contour)

        return merged

    def _filter_contours(
        self, contours: List[np.ndarray], image_shape: Tuple[int, int]
    ) -> List[np.ndarray]:
        """Enhanced contour filtering with improved criteria."""
        filtered = []
        image_height, image_width = image_shape

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            aspect_ratio = float(w) / h if h > 0 else 0

            # Apply refined size and shape constraints
            if (
                self.min_letter_width < w < self.max_letter_width
                and self.min_letter_height < h < self.max_letter_height
                and area > self.min_area
                and 0.1 < aspect_ratio < 3.0
            ):  # Allow more variation in aspect ratio

                # Exclude image borders
                if (
                    x > 1
                    and y > 1
                    and x + w < image_width - 1
                    and y + h < image_height - 1
                ):
                    filtered.append(contour)

        return filtered

    def segment_letters(
        self, image: np.ndarray, binary: np.ndarray
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Enhanced letter segmentation with improved component handling."""
        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter contours
        filtered_contours = self._filter_contours(contours, binary.shape)

        # Merge overlapping components
        merged_contours = self._merge_overlapping_components(filtered_contours)

        # Sort contours left to right
        sorted_contours = sorted(merged_contours, key=lambda c: cv2.boundingRect(c)[0])

        letters = []
        for contour in sorted_contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Extract letter with dynamic padding
            pad = max(5, min(w, h) // 4)  # Dynamic padding based on letter size
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(binary.shape[1], x + w + pad)
            y2 = min(binary.shape[0], y + h + pad)

            letter_img = image[y1:y2, x1:x2].copy()

            # Ensure consistent size while maintaining aspect ratio
            target_size = (128, 128)
            h, w = letter_img.shape[:2]
            aspect = w / float(h)

            if aspect > 1:
                new_w = target_size[0]
                new_h = int(new_w / aspect)
            else:
                new_h = target_size[1]
                new_w = int(new_h * aspect)

            resized = cv2.resize(letter_img, (new_w, new_h))

            # Create a square image with the letter centered
            square_img = np.full(
                (target_size[1], target_size[0], 3), 255, dtype=np.uint8
            )
            y_offset = (target_size[1] - new_h) // 2
            x_offset = (target_size[0] - new_w) // 2
            square_img[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = (
                resized
            )

            letters.append((square_img, (x, y, w, h)))

        return letters


class LetterRecognizer:
    """
    Handles recognition of individual letters using the CNN model,
    with case detection as a post-processing step.
    """

    def __init__(self, model_path: str):
        """
        Initialize the recognizer with improved case handling.

        Args:
            model_path: Path to the trained model weights
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Base classes (uppercase only - for model compatibility)
        self.classes = [
            "А", "Б", "В", "Г", "Д", "Е", "Ж", "З", "И", "Й", "К", "Л", "М", "Н",
            "О", "П", "Р", "С", "Т", "У", "Ф", "Х", "Ц", "Ч", "Ш", "Щ", "Ъ", "Ы",
            "Ь", "Э", "Ю", "Я", "Ң", "Ү", "Ө", "Ё"
        ]

        # Create case mappings
        self.case_mappings = {upper: upper.lower() for upper in self.classes}

        # Initialize model and transforms
        self.model = self._load_model(model_path)
        self.transform = self._get_transforms()
        self.confidence_threshold = 0.7

    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load and prepare the CNN model with original 36 classes."""
        try:
            model = KyrgyzCNN(len(self.classes))  # 36 classes for model compatibility
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model = model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {str(e)}")
            raise

    def _get_transforms(self) -> transforms.Compose:
        """Enhanced image transformations."""
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def _enhance_image(self, letter_image: np.ndarray) -> np.ndarray:
        """Enhanced image preprocessing for better case detection."""
        if len(letter_image.shape) == 3:
            gray = cv2.cvtColor(letter_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = letter_image

        # Enhance contrast
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
        
        # Denoise while preserving edges
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        return binary

    def _detect_case(self, letter_image: np.ndarray) -> float:
        """
        Enhanced case detection returning a confidence score.
        
        Args:
            letter_image: Letter image in BGR format
            
        Returns:
            float: Score between 0 and 1 (0 = lowercase, 1 = uppercase)
        """
        binary = self._enhance_image(letter_image)
        
        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return 0.5
            
        # Get the main contour
        main_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)
        
        # Calculate metrics
        height, width = binary.shape
        relative_height = h / height
        relative_y = y / height
        relative_width = w / width
        aspect_ratio = w / h if h > 0 else 1
        area_ratio = cv2.contourArea(main_contour) / (height * width)
        
        # Calculate case score
        score = 0.0
        weights = {
            'height': 0.3,
            'position': 0.2,
            'width': 0.2,
            'aspect': 0.15,
            'area': 0.15
        }
        
        score += weights['height'] * (1.0 if relative_height > 0.65 else 0.0)
        score += weights['position'] * (1.0 if relative_y < 0.3 else 0.0)
        score += weights['width'] * (1.0 if relative_width > 0.4 else 0.0)
        score += weights['aspect'] * (1.0 if aspect_ratio > 0.8 else 0.0)
        score += weights['area'] * (1.0 if area_ratio > 0.3 else 0.0)
        
        return score

    def recognize_letter(
        self,
        letter_image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        confidence_threshold: Optional[float] = None,
    ) -> RecognitionResult:
        """
        Recognize letter with case detection as post-processing.
        
        Args:
            letter_image: Letter image in BGR format
            bbox: Bounding box coordinates
            confidence_threshold: Optional confidence threshold override
            
        Returns:
            RecognitionResult with recognized letter and metadata
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold

        # Convert to PIL and transform
        pil_image = Image.fromarray(cv2.cvtColor(letter_image, cv2.COLOR_BGR2RGB))
        tensor = self.transform(pil_image)
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            probabilities = F.softmax(output, dim=1)
            
            # Get top predictions
            topk = 3
            confidences, indices = torch.topk(probabilities, k=topk, dim=1)
            
            # Get base prediction and confidence
            confidence = confidences[0][0].item()
            predicted_idx = indices[0][0].item()
            predicted_upper = self.classes[predicted_idx]
            
            # Get case score
            case_score = self._detect_case(letter_image)
            
            # Determine final case based on case_score
            predicted_letter = (
                predicted_upper if case_score > 0.5 
                else self.case_mappings[predicted_upper]
            )

            # Log prediction details
            logger.debug(f"Base prediction: {predicted_upper} (conf: {confidence:.3f})")
            logger.debug(f"Case score: {case_score:.3f}")
            logger.debug(f"Final prediction: {predicted_letter}")
            
            # Log top-k predictions
            for i in range(topk):
                idx = indices[0][i].item()
                conf = confidences[0][i].item()
                logger.debug(f"Top-{i+1}: {self.classes[idx]} (conf: {conf:.3f})")

        return RecognitionResult(
            letter=predicted_letter,
            confidence=confidence * (0.5 + 0.5 * abs(case_score - 0.5)),  # Adjust confidence based on case certainty
            bounding_box=bbox,
            image=letter_image
        )

    def get_case_variants(self, letter: str) -> Tuple[str, str]:
        """Get both case variants of a letter."""
        letter_upper = letter.upper()
        if letter_upper in self.classes:
            return letter_upper, self.case_mappings[letter_upper]
        return letter, letter

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
