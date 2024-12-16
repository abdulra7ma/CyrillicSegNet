import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

# Import your recognition system
from word_recogination import WordRecognitionResult, WordRecognizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RecognitionState:
    """Stores the current recognition state and results."""

    image_path: Optional[str] = None
    original_image: Optional[np.ndarray] = None
    result: Optional[WordRecognitionResult] = None
    is_processing: bool = False


class RecognitionWorker(QThread):
    """Worker thread for running recognition tasks."""

    finished = pyqtSignal(WordRecognitionResult)
    error = pyqtSignal(str)

    def __init__(self, recognizer: WordRecognizer, image_path: str):
        """Initialize the worker thread."""
        super().__init__()
        self.recognizer = recognizer
        self.image_path = image_path

    def run(self):
        """Run the recognition task."""
        try:
            result = self.recognizer.recognize_word(self.image_path)
            self.finished.emit(result)
        except Exception as e:
            logger.error(f"Recognition error: {str(e)}")
            self.error.emit(str(e))


class ImageWidget(QLabel):
    """Custom widget for displaying images with proper scaling."""

    def __init__(self):
        """Initialize the image widget."""
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setStyleSheet(
            """
            QLabel {
                background-color: #f0f0f0;
                border: 2px solid #cccccc;
                border-radius: 5px;
            }
        """
        )

    def setImage(self, image: np.ndarray):
        """Set and display an image with proper scaling."""
        if image is None:
            return

        # Convert to RGB
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:  # BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create QImage
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(
            image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
        )

        # Scale to fit widget while maintaining aspect ratio
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self.setPixmap(scaled_pixmap)


class ResultsWidget(QFrame):
    """Widget for displaying enhanced recognition results with detailed visualizations."""

    def __init__(self):
        """Initialize the enhanced results widget."""
        super().__init__()
        self.initUI()

        # Set color scheme for confidence levels
        self.confidence_colors = {
            "high": "#28a745",  # Green
            "medium": "#ffc107",  # Yellow
            "low": "#dc3545",  # Red
        }

    def initUI(self):
        """Set up the enhanced UI components."""
        layout = QVBoxLayout()

        # Word result container with styling
        word_container = QFrame()
        word_container.setStyleSheet(
            """
            QFrame {
                background-color: #f8f9fa;
                border-radius: 10px;
                padding: 15px;
                margin: 5px;
            }
        """
        )
        word_layout = QVBoxLayout(word_container)

        # Word result label with enhanced styling
        self.word_label = QLabel()
        self.word_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        self.word_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.word_label.setStyleSheet(
            """
            QLabel {
                padding: 15px;
                background-color: #e8e8e8;
                border-radius: 8px;
                color: #2c3e50;
            }
        """
        )

        # Statistics container
        stats_container = QFrame()
        stats_layout = QHBoxLayout(stats_container)

        # Confidence indicators
        self.confidence_label = QLabel()
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.confidence_label.setStyleSheet(
            """
            QLabel {
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
        """
        )

        # Letter count label
        self.letter_count_label = QLabel()
        self.letter_count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.letter_count_label.setStyleSheet(
            """
            QLabel {
                padding: 10px;
                background-color: #e8e8e8;
                border-radius: 5px;
            }
        """
        )

        stats_layout.addWidget(self.confidence_label)
        stats_layout.addWidget(self.letter_count_label)

        # Segmentation visualization area
        self.visualization_area = QScrollArea()
        self.visualization_area.setWidgetResizable(True)
        self.visualization_area.setMinimumHeight(250)
        self.visualization_area.setStyleSheet(
            """
            QScrollArea {
                border: 2px solid #ddd;
                border-radius: 8px;
                background-color: white;
            }
        """
        )

        # Container for letter details
        self.letter_container = QWidget()
        self.letter_grid = QGridLayout(self.letter_container)
        self.visualization_area.setWidget(self.letter_container)

        # Add all components to main layout
        word_layout.addWidget(self.word_label)
        word_layout.addWidget(stats_container)

        layout.addWidget(word_container)
        layout.addWidget(self.visualization_area)

        self.setLayout(layout)

    def _get_confidence_color(self, confidence: float) -> str:
        """Get color based on confidence level."""
        if confidence >= 0.85:
            return self.confidence_colors["high"]
        elif confidence >= 0.60:
            return self.confidence_colors["medium"]
        return self.confidence_colors["low"]

    def _create_letter_widget(self, letter_result, index: int) -> QFrame:
        """Create an enhanced widget for individual letter results."""
        letter_frame = QFrame()
        letter_frame.setStyleSheet(
            f"""
            QFrame {{
                background-color: #ffffff;
                border: 2px solid {self._get_confidence_color(letter_result.confidence)};
                border-radius: 8px;
                padding: 5px;
                margin: 2px;
            }}
        """
        )

        layout = QVBoxLayout(letter_frame)

        # Letter image with border
        image_label = ImageWidget()
        image_label.setMinimumSize(120, 120)
        image_label.setMaximumSize(120, 120)
        image_label.setImage(letter_result.image)

        # Letter information
        info_container = QFrame()
        info_layout = QVBoxLayout(info_container)

        # Letter and index
        letter_label = QLabel(f"Letter {index + 1}: {letter_result.letter}")
        letter_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        letter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Confidence with color-coded background
        confidence_frame = QFrame()
        confidence_frame.setStyleSheet(
            f"""
            QFrame {{
                background-color: {self._get_confidence_color(letter_result.confidence)};
                border-radius: 5px;
                padding: 3px;
            }}
        """
        )
        confidence_layout = QVBoxLayout(confidence_frame)

        confidence_label = QLabel(f"{letter_result.confidence:.1f}%")
        confidence_label.setStyleSheet("color: white; font-weight: bold;")
        confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        confidence_layout.addWidget(confidence_label)

        # Add components to info layout
        info_layout.addWidget(letter_label)
        info_layout.addWidget(confidence_frame)

        # Bounding box information
        bbox_label = QLabel(
            f"Position: ({letter_result.bounding_box[0]}, {letter_result.bounding_box[1]})"
        )
        bbox_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bbox_label.setStyleSheet("color: #666; font-size: 10px;")

        # Add all components to main layout
        layout.addWidget(image_label)
        layout.addWidget(info_container)
        layout.addWidget(bbox_label)

        return letter_frame

    def showResults(self, result: WordRecognitionResult):
        """Display enhanced recognition results."""
        # Clear previous results
        for i in reversed(range(self.letter_grid.count())):
            self.letter_grid.itemAt(i).widget().setParent(None)

        # Update word result
        self.word_label.setText(f"Recognized Word: {result.word}")

        # Update confidence with color
        conf_color = self._get_confidence_color(result.confidence)
        self.confidence_label.setStyleSheet(
            f"""
            QLabel {{
                padding: 10px;
                background-color: {conf_color};
                color: white;
                border-radius: 5px;
                font-weight: bold;
            }}
        """
        )
        self.confidence_label.setText(f"Confidence: {result.confidence:.1f}%")

        # Update letter count
        self.letter_count_label.setText(f"Letters: {len(result.letter_results)}")

        # Show individual letters with enhanced visualization
        for i, letter_result in enumerate(result.letter_results):
            letter_widget = self._create_letter_widget(letter_result, i)
            self.letter_grid.addWidget(letter_widget, 0, i)

            # Add separator except for last item
            if i < len(result.letter_results) - 1:
                separator = QFrame()
                separator.setFrameShape(QFrame.Shape.VLine)
                separator.setStyleSheet("background-color: #ddd;")
                self.letter_grid.addWidget(separator, 0, i + 1)


class KyrgyzRecognitionGUI(QMainWindow):
    """Main window for the Kyrgyz word recognition application."""

    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        self.state = RecognitionState()
        self.recognizer = WordRecognizer("./results/final_model.pth")
        self.initUI()

    def initUI(self):
        """Set up the user interface."""
        self.setWindowTitle("Kyrgyz Word Recognition")
        self.setMinimumSize(800, 600)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create splitter for main areas
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Top area - Image display
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)

        self.image_widget = ImageWidget()

        # Buttons
        button_layout = QHBoxLayout()
        self.open_button = QPushButton("Open Image")
        self.open_button.clicked.connect(self.openImage)
        self.process_button = QPushButton("Recognize Word")
        self.process_button.clicked.connect(self.processImage)
        self.process_button.setEnabled(False)

        button_layout.addWidget(self.open_button)
        button_layout.addWidget(self.process_button)

        top_layout.addWidget(self.image_widget)
        top_layout.addLayout(button_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        top_layout.addWidget(self.progress_bar)

        # Bottom area - Results
        self.results_widget = ResultsWidget()

        # Add widgets to splitter
        splitter.addWidget(top_widget)
        splitter.addWidget(self.results_widget)

        layout.addWidget(splitter)

    def openImage(self):
        """Open an image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_path:
            try:
                self.state.image_path = file_path
                self.state.original_image = cv2.imread(file_path)
                self.image_widget.setImage(self.state.original_image)
                self.process_button.setEnabled(True)
                self.results_widget.word_label.setText("")
                self.results_widget.confidence_label.setText("")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")

    def processImage(self):
        """Process the loaded image."""
        if not self.state.image_path or self.state.is_processing:
            return

        self.state.is_processing = True
        self.process_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress

        # Start recognition in worker thread
        self.worker = RecognitionWorker(self.recognizer, self.state.image_path)
        self.worker.finished.connect(self.onRecognitionFinished)
        self.worker.error.connect(self.onRecognitionError)
        self.worker.start()

    def onRecognitionFinished(self, result: WordRecognitionResult):
        """Handle completion of recognition task."""
        self.state.is_processing = False
        self.process_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.results_widget.showResults(result)

    def onRecognitionError(self, error_message: str):
        """Handle recognition errors."""
        self.state.is_processing = False
        self.process_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Recognition Error", error_message)


def main():
    """Main function to run the application."""
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle("Fusion")

    # Create and show the main window
    window = KyrgyzRecognitionGUI()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
