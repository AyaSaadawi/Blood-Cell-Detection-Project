# Blood Cell Detection and Classification Using YOLOv12

## Project Overview

This project focuses on automating the detection and classification of blood cells in microscopic images using YOLOv12, a state-of-the-art object detection model. The goal is to streamline the analysis of blood smears, which is crucial for diagnosing blood disorders. By leveraging deep learning, this project aims to offer an efficient, accurate, and real-time solution for detecting red blood cells (RBCs), white blood cells (WBCs), and platelets. This method is particularly valuable in clinical settings, improving diagnostic speed and objectivity.

---

## Key Objectives

### Data Preprocessing and Augmentation
- **Data Cleaning**: Ensuring image quality and consistency by resizing, auto-orienting, and performing minor adjustments.
- **Data Augmentation**: Using techniques like flipping, rotation, and blur to create a more diverse and robust dataset.

### YOLOv12 Architecture
- **Object Detection**: Detecting and classifying RBCs, WBCs, and platelets using YOLOv12, which excels at small object detection.

### Real-Time Processing
- **Streamlit Interface**: Developing a real-time image and video upload feature for efficient clinical use.

---

## Project Features

### Data Preprocessing
- **Auto-Orientation**: Corrects the alignment of blood cells in images.
- **Resizing**: Standardizes image dimensions to 640x640 pixels.
  
### Data Augmentation Techniques
- Horizontal Flipping, Rotation, Shear, Brightness and Saturation Adjustments, and Blur were applied to improve model generalization.

### YOLOv12 Object Detection
- **YOLOv12 Model**: YOLOv12's advanced architecture makes it suitable for detecting small objects, such as platelets, which are often challenging in blood smear images.

### Model Evaluation Metrics
- Precision: 0.808
- Recall: 0.892
- mAP@50: 0.897

### Streamlit UI
- Real-time blood cell detection from image/video uploads with bounding boxes, labels, and confidence scores.

---

## Technologies Used

- **YOLOv12 (Ultralytics)**: For object detection.
- **Streamlit**: For developing the real-time web interface.
- **Google Colab**: For model training with NVIDIA Tesla T4 GPU.
- **OpenCV**: For image and video processing.
- **Python Libraries**: numpy, pandas, matplotlib, etc., for data manipulation and visualization.

---

## Dataset

The dataset used for training consists of blood cell images annotated for object detection:

- **Original Images**: 364 images of RBCs, WBCs, and platelets.
- **Augmented Dataset**: Expanded to 2,190 images using various augmentation techniques.

### Dataset Composition

- **RBCs**: Red Blood Cells.
- **WBCs**: White Blood Cells.
- **Platelets**: Small fragments aiding in clotting.

### Dataset Accessibility

The dataset is publicly available for academic and research purposes, accessible upon request.

---

## Key Files

### Preprocessing and Augmentation
- `yolo_train_eval.ipynb`: Notebook for training the YOLOv12 model on the dataset.
- `model_config.yaml`: Configuration file for YOLOv12 training.

### Real-Time Inference
- `app.py`: Streamlit app that handles image/video uploads and runs inference on YOLOv12.

---

## How to Run the Project

### Step 1: Set Up the Environment
Clone the repository:
```bash
git clone https://github.com/YourUsername/blood-cell-detection-yolov12.git
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Train the Model
Run the YOLOv12 training script:
```bash
python train_yolov12.py
```

### Step 4: Real-Time Inference
Start the Streamlit app to interact with the trained model:
```bash
streamlit run app.py
```

### Step 5: Upload Images or Videos
Upload blood smear images or video streams to detect and classify RBCs, WBCs, and platelets in real-time.

---

## Results and Evaluation

The YOLOv12 model has been trained and evaluated on a dataset of blood cell images. The evaluation metrics are:

- **Precision**: 0.808
- **Recall**: 0.892
- **mAP@50**: 0.897

The model excels at detecting platelets with high precision and recall. Challenges remain with overlapping or blurry cells, which can affect detection performance.

---

## Future Work

- **Dataset Expansion**: Increase the dataset size to improve model accuracy and robustness.
- **Real-Time Video Processing**: Improve the model’s capability to process video streams more efficiently.
- **Model Optimization**: Explore further optimization techniques for real-time inference on resource-constrained devices.

---

## Contributing

Contributions are welcome! Feel free to fork the repository, open an issue, or submit a pull request with your improvements.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Ultralytics YOLOv12 Documentation](https://docs.ultralytics.com)
- [Streamlit Documentation](https://streamlit.io)
- [Blood Cell Detection Dataset on Kaggle](https://www.kaggle.com/datasets/adhoppin/blood-cell-detection-dataset)
- [Roboflow](https://roboflow.com/)

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- Datasets sourced from Kaggle for educational purposes.
- Thanks to the open-source community for tools and libraries used in this project.


