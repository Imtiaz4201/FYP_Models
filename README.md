# UTILITY POLE RECOGNITION

## Description
In this folder, all the Python files are used for data preprocessing, training, and evaluating deep learning models. Each script is designed to handle specific stages of the workflow, ensuring efficient and accurate model development.

## Installation
1. **Install Visual Studio Code or Anaconda**: These tools are recommended for running `.ipynb` files (Jupyter Notebooks).
2. **Install Python 3.10**: Ensure you have Python 3.10 installed on your system.
3. **Install the Required Packages**: Use `pip` to install the necessary packages.
pip install torch==2.3.1
pip install ultralytics
pip install tensorflow==2.10
pip install opencv-python==4.10.0.82
4. **Install CUDA 11.0 and cuDNN 8.0**: Follow the official installation guides for CUDA and cuDNN to set up these dependencies on your system.

## Overview of Notebooks
1. **augmentation_for_pascal_voc.ipynb**: This notebook includes various data augmentation techniques specifically designed for the PASCAL VOC dataset to enhance the training data and improve model performance.
2. **evaluation.ipynb**: Contains the code for evaluating the performance of the trained models using different metrics and visualization tools to assess their accuracy and effectiveness.
3. **models_for_faster_rcnn.ipynb**: This notebook covers the training and testing of Faster R-CNN models, detailing the implementation, training process, and evaluation on the test data.
4. **pre_processing.ipynb**: Focuses on preprocessing the raw data, including steps such as data cleaning, normalization, and feature engineering to prepare the data for model training.
5. **pre_processing_for_tensorflow2_API.ipynb**: Provides preprocessing steps tailored for TensorFlow 2 API, ensuring the data is in the correct format and optimized for use with TensorFlow models.
6. **models_for_yolo.ipynb**: This notebook handles the training and testing of YOLO (You Only Look Once) models, including the implementation details, training procedures, and performance evaluation on test data.
7. **utils/confusion_matrix_tf2.py**: Generate confusion matrix for Faster R-CNN.
8. **utils/generate_tfrecords.py**: Generate tfrecords for Faster R-CNN models (train, test and validation dataset).
9. **utils/xml_to_csv.py**: Convert PASCAL VOC data/XML file to csv format which will be used to create tfrecords.
