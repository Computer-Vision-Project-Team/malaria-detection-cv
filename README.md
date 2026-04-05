# AI-Powered Malaria Parasite Screening Tool

![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/AI-TensorFlow_2.21-FF6F00?logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/CV-OpenCV-5C3EE8?logo=opencv&logoColor=white)

An automated, deep learning-based diagnostic support tool designed to identify malaria parasites in microscopic, Giemsa-stained red blood cell images. 

---

## Project Overview
Manual malaria diagnosis is highly time-consuming and prone to human error due to fatigue. This project provides a specialized **Classification Engine** that acts as a "second pair of eyes" for Medical Laboratory Technicians (MLTs). By uploading an image of a single red blood cell, the system instantly predicts whether the cell is **Parasitized** or **Uninfected** and provides visual evidence using computer vision segmentation.

### System Architecture & Methodology
This pipeline integrates three distinct computer vision and machine learning techniques:
1. **Image Enhancement (Method 1):** OpenCV `GaussianBlur` (3x3 kernel) is applied to all incoming images to reduce microscopic background noise while strictly preserving the sharp edges of the malaria parasites.
2. **Classification (Method 2):** A custom **MobileNetV2 Transfer Learning** architecture. The base model extracts spatial features, while a custom dense head outputs a binary classification. It achieved **94% accuracy** and a **93% recall** rate for parasitized cells on an unseen test set.
3. **Image Segmentation (Method 3):** If a cell is flagged as infected, an OpenCV HSV color-thresholding algorithm isolates the dark purple Giemsa stain and draws a bounding box directly around the parasite for visual interpretability.

---

## Repository Structure

```text
malaria-detection-cv/
├── .streamlit/             # UI Theme configurations
│   └── config.toml         
├── data/                   # SAMPLE IMAGES FOR TESTING 
├── models/                 # Pre-trained model weights
│   └── malaria_cnn.h5      # (See "Note on Model Weights" below)
├── notebooks/              # Jupyter notebooks for model training
│   └── model_train.ipynb   
├── src/                    # Source code for the application
│   ├── app.py              # Main Streamlit web dashboard
│   ├── image_utils.py      # OpenCV processing functions
│   ├── preprocess.py       # Data generator configuration
│   ├── train.py            # Model training pipeline
│   └── evaluate.py         # Model evaluation script
├── requirements.txt        # Project dependencies
└── README.md
```

## Note on Model Weights (.h5 file)
Due to GitHub file size limitations, the trained malaria_cnn.h5 model is not hosted in the GitHub repository.

If viewing on GitHub: The model weights and complete testing dataset have been submitted via a secure OneDrive link (included in the final coursework report).

If viewing from the OneDrive submission: The model is already included in the models/ directory, and the app is ready to run.

## Setup and Installation
### 1. Prerequisites
Python 3.9 - 3.12 installed on your system.

Windows Users - Long Path Fix (REQUIRED): Deep learning libraries have deeply nested folders that exceed the default Windows 260-character limit. Before installing, open Windows PowerShell as Administrator and run:

Windows PowerShell as administrator
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1

### 2. Installation Steps
Clone or extract the project, then open your terminal in the root directory (malaria-detection-cv).

### Step 1: Create a virtual environment

Bash
python -m venv venv

### Step 2: Activate the environment

Windows: venv\Scripts\activate

Mac/Linux: source venv/bin/activate

### Step 3: Install dependencies
(Note: We utilize tensorflow-cpu and opencv-python-headless for a lightweight, optimized web deployment).

Bash
pip install -r requirements.txt

### Running the Application
Once dependencies are installed, launch the Streamlit dashboard by running:

Bash
streamlit run src/app.py

The application will open automatically in your default web browser at http://localhost:8501.

## Testing the Application
Sample images for both Parasitized and Uninfected classes are provided in the data/ folder.
Steps:

Launch the Streamlit app using the command above.
Upload a sample image from the data/ folder.
View the classification result, confidence score, and parasite bounding boxes.


## Performance

Test Accuracy: 94%
Precision / Recall / F1-Score: 0.94 (balanced for both classes)
Validation Accuracy: 95.041%


## Team & Collaboration
Developed collaboratively using GitHub for version control. Detailed individual contributions are documented in the final project report.

## Additional Information
Training & Evaluation:
You can retrain or evaluate the model using the scripts in the src/ folder:

Bashpython src/train.py
python src/evaluate.py

## Deployment on Streamlit Community Cloud (Future Work)
The app is currently local-only. For cloud deployment, the model would need to be hosted externally due to size constraints.