
# Image-Based Wind and Cloud Feature Extraction with Deep Learning Frame Interpolation

This project focuses on leveraging computer vision and deep learning techniques to analyze satellite imagery for extracting key features like wind direction, wind speed, and cloud density. It employs convolutional autoencoders for frame interpolation, providing smooth transitions between time-sequenced images. The repository includes Python scripts for preprocessing and feature extraction, as well as Jupyter notebooks for training and evaluating deep learning models.

---

## Main Components

### 1. **Feature Extraction**
- **`extract/app.py`**:
  - Extracts wind direction and magnitude from satellite images using OpenCV.
  - Outputs processed images and structured data files for further analysis.

### 2. **Feature Mapping**
- **`mapping/app.py`**:
  - Combines wind and cloud features into a unified dataset.
  - Saves extracted features in a CSV file for downstream tasks.

### 3. **Frame Interpolation**
- **`Bhot-sih-liya-main/Untitled1.ipynb`**:
  - Jupyter notebook implementing convolutional autoencoders.
  - Key functionalities include:
    - Loading and preparing data for frame interpolation.
    - Training the deep learning model on satellite image sequences.
    - Generating and visualizing interpolated frames between input images.

---

## Installation and Dependencies

Ensure you have Python installed. Then, install the required dependencies using pip:

```bash
pip install tensorflow==2.10.0 numpy pandas matplotlib opencv-python
```

---

## Usage

### 1. **Extract Wind Features**
Run the script to process wind images and extract direction and magnitude:
```bash
python extract/app.py
```

### 2. **Map Wind and Cloud Features**
Combine the processed wind and cloud features into a single dataset:
```bash
python mapping/app.py
```

### 3. **Train the Frame Interpolation Model**
Open the Jupyter notebook and execute the cells to train the autoencoder model:
```bash
jupyter notebook Bhot-sih-liya-main/Untitled1.ipynb
```

---

## Outputs

- **Processed Files**:
  - CSV files such as `wind_directions.csv` and `wind_cloud_features.csv`.
  - Processed images visualizing wind direction and cloud density.

- **Model Artifacts**:
  - Trained autoencoder models saved as `.h5` or `.keras` files.

- **Generated Images**:
  - Interpolated frames stored in directories like `GeneratedImages/`.

---

## License

Refer to the `LICENSE` file in the `Bhot-sih-liya-main` directory for details about usage and distribution.

Dataset
The images uesd are taken from INSAT-3DR Asia Sector Infrared2(BT) images and Wind data is taken from EOS-06 scatterometer. All the images are taken from ISRO's website https://mosdac.gov.in/gallery/

Preprocessing
Cloud Masking
Mapping EOS-06 to INSAT-3DR
Forming Wind Tensors
Model Definition
Variational AutoEncoders
ConvLSTM