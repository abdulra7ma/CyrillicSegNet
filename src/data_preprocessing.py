import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """Class for preprocessing images for the Kyrgyz Letter Recognition model."""
    
    def __init__(self, target_size: Tuple[int, int] = (128, 128)):
        """
        Initialize the preprocessor with target image size.
        
        Args:
            target_size: Tuple of (width, height) for output images
        """
        self.target_size = target_size
    
    @staticmethod
    def _read_image(image_path: str) -> np.ndarray:
        """
        Read an image file safely.
        
        Args:
            image_path: Path to the image file
        
        Returns:
            numpy.ndarray: Image as a numpy array
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to read image: {image_path}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"Error reading image {image_path}: {str(e)}")
            raise

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing steps to an image.
        
        Args:
            image: Input image as numpy array
        
        Returns:
            numpy.ndarray: Preprocessed image
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply adaptive thresholding
            image = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Remove noise
            image = cv2.medianBlur(image, 3)
            
            # Resize image
            image = cv2.resize(image, self.target_size)
            
            # Normalize pixel values
            image = image.astype(np.float32) / 255.0
            
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise

class DatasetOrganizer:
    """Class for organizing and splitting the dataset."""
    
    def __init__(
        self,
        raw_data_path: str,
        processed_data_path: str,
        train_ratio: float = 0.8,
        random_state: int = 42
    ):
        """
        Initialize dataset organizer.
        
        Args:
            raw_data_path: Path to raw data
            processed_data_path: Path to save processed data
            train_ratio: Ratio of training to total data
            random_state: Random seed for reproducibility
        """
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.train_ratio = train_ratio
        self.random_state = random_state
        
    def create_directory_structure(self) -> None:
        """Create the necessary directory structure for processed data."""
        try:
            # Create main directories
            for split in ['train', 'test']:
                split_dir = self.processed_data_path / split
                split_dir.mkdir(parents=True, exist_ok=True)
                
                # Create class subdirectories
                for class_dir in self.raw_data_path.iterdir():
                    if class_dir.is_dir():
                        (split_dir / class_dir.name).mkdir(exist_ok=True)
                        
        except Exception as e:
            logger.error(f"Error creating directory structure: {str(e)}")
            raise

    def organize_dataset(self) -> None:
        """Organize and split the dataset into train and test sets."""
        try:
            # Process each class directory
            for class_dir in tqdm(list(self.raw_data_path.iterdir()), desc="Processing classes"):
                if not class_dir.is_dir():
                    continue
                
                # Get all image files
                image_files = [
                    f for f in class_dir.iterdir()
                    if f.suffix.lower() in ['.png', '.jpg', '.jpeg']
                ]
                
                # Split into train and test
                train_files, test_files = train_test_split(
                    image_files,
                    train_size=self.train_ratio,
                    random_state=self.random_state
                )
                
                # Copy files to respective directories
                for files, split in [(train_files, 'train'), (test_files, 'test')]:
                    dest_dir = self.processed_data_path / split / class_dir.name
                    for file in files:
                        shutil.copy2(file, dest_dir / file.name)
                        
        except Exception as e:
            logger.error(f"Error organizing dataset: {str(e)}")
            raise

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Preprocess Kyrgyz Letter Recognition dataset')
    
    parser.add_argument('--raw-dir', type=str, required=True,
                       help='Path to raw data directory')
    parser.add_argument('--processed-dir', type=str, required=True,
                       help='Path to save processed data')
    parser.add_argument('--image-size', type=int, default=128,
                       help='Target image size (width=height)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Ratio of training to total data')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    return parser.parse_args()

def main():
    """Main function to run the preprocessing pipeline."""
    args = parse_args()
    
    try:
        logger.info("Starting data preprocessing...")
        
        # Initialize preprocessor and organizer
        preprocessor = ImagePreprocessor(target_size=(args.image_size, args.image_size))
        organizer = DatasetOrganizer(
            args.raw_dir,
            args.processed_dir,
            train_ratio=args.train_ratio,
            random_state=args.random_seed
        )
        
        # Create directory structure
        logger.info("Creating directory structure...")
        organizer.create_directory_structure()
        
        # Organize dataset
        logger.info("Organizing dataset...")
        organizer.organize_dataset()
        
        # Process all images
        logger.info("Processing images...")
        for split in ['train', 'test']:
            split_dir = Path(args.processed_dir) / split
            for class_dir in tqdm(list(split_dir.iterdir()), desc=f"Processing {split} set"):
                if not class_dir.is_dir():
                    continue
                    
                for img_path in class_dir.iterdir():
                    if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
                        continue
                        
                    # Read and preprocess image
                    image = preprocessor._read_image(str(img_path))
                    processed_image = preprocessor.preprocess_image(image)
                    
                    # Save processed image
                    processed_image = (processed_image * 255).astype(np.uint8)
                    Image.fromarray(processed_image).save(img_path)
        
        logger.info("Preprocessing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    main()