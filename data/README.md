# Data Directory Structure

This directory contains all datasets used for training and testing the Kyrgyz letter recognition model.

## Directory Structure
```
data/
├── README.md                   # This file
├── raw/                       # Original, unprocessed datasets
│   ├── cyrilic_words/        # Cyrillic handwriting dataset
│   ├── handwritten_kyrgyz_letters/  # Kyrgyz letters dataset
│   └── test_model/           # Test data for model evaluation
├── processed/                 # Processed and prepared datasets
│   ├── combined/             # Merged datasets after preprocessing
│   ├── cyrillic_words/       # Processed Cyrillic words
│   └── handwritten_kyrgyz_letters/  # Processed Kyrgyz letters
└── test/                     # Reserved for testing data
```

## Dataset Sources and Setup

### 1. Handwritten Kyrgyz Letters Dataset
- **Source**: [Kaggle - Database of 36 Handwritten Kyrgyz Letters](https://www.kaggle.com/datasets/ilgizzhumaev/database-of-36-handwritten-kyrgyz-letters/code)
- **Setup Instructions**:
  1. Download the dataset from Kaggle
  2. Extract the contents
  3. Place the extracted images in `raw/handwritten_kyrgyz_letters/`
  4. Expected structure:
     ```
     handwritten_kyrgyz_letters/
     ├── А/
     ├── Б/
     ├── В/
     └── ... (all 36 letter folders)
     ```

### 2. Cyrillic Handwriting Dataset
- **Source**: [Kaggle - Cyrillic Handwriting Dataset](https://www.kaggle.com/datasets/constantinwerner/cyrillic-handwriting-dataset/data)
- **Setup Instructions**:
  1. Download the dataset from Kaggle
  2. Extract the contents
  3. Place the extracted data in `raw/cyrilic_words/`
  4. Expected structure:
     ```
     cyrilic_words/
     ├── images/
     │   ├── word1.png
     │   ├── word2.png
     │   └── ...
     └── labels/
         ├── word1.txt
         └── ...
     ```

## Data Processing

After placing the raw data, run the preprocessing scripts:
```bash
# From project root
python src/data_preprocessing.py
```

This will:
1. Clean and normalize the images
2. Create the processed datasets in `processed/`
3. Prepare the combined dataset in `processed/combined/`

## Test Data

The `test/` directory is reserved for:
- Model evaluation data
- Validation sets
- Custom test cases

## Data Format Requirements

### Images
- Format: PNG
- Size: 28x28 pixels (will be resized during preprocessing)
- Color: Grayscale
- Background: White
- Foreground: Black text/letters

### Labels
- Kyrgyz letters: Directory name as label
- Cyrillic words: Corresponding txt file with UTF-8 encoding

## Notes
- Keep raw data intact in `raw/` directory
- All preprocessing outputs go to `processed/`
- Use `test/` for custom test cases
- Backup your raw data before running preprocessing scripts

---

For questions about the datasets or setup issues, please open an issue in the GitHub repository.