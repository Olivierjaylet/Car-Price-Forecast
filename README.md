# Car Price Prediction using Machine Learning

This project applies multiple machine learning models to predict car prices based on numerical, categorical, and image data from the DVM-Car dataset. 
The goal is to compare the performance of various models, including both classical machine learning algorithms and deep learning models, in accurately predicting car prices.

## Project Overview

The project involves predicting car prices using several machine learning models, focusing on both classical tabular data (numerical and categorical) and image data. The models implemented include:
- **K-Nearest Neighbors (KNN)**
- **Extreme Gradient Boosting (XGBoost)**
- **Multilayer Perceptron (MLP)**
- **Convolutional Neural Network (CNN)**

## Roadmap of the project (see pdf : https://github.com/Olivierjaylet/Car-Price-Forecast/blob/1e337ce00b09a74b5e77ff2d0cf8b58183eb6844/Car_Price_Prediction.pdf)

1) Data Description
2) Feature engineering
3) Descriptive statistics
4) Optimization of different Machine Learning Models
5) Reduction of information methods
6) Agnostic methods
7) Data augmentation
8) Tuning of a pre-trained CNN model (Resnet-18)
9) Comparative results


### Key Features:
- **Data Integration**: Combining both tabular data and image data for price prediction.
- **Model Comparison**: Evaluating and comparing classical and deep learning models.
- **Model Interpretability**: Using model-agnostic methods such as SHAP values and ICE plots to interpret the model results.

## Dataset

The dataset used in this project is the **DVM-Car dataset**, which includes:
- **Numerical and Categorical Data**: Attributes like mileage, engine size, fuel type, and gearbox type.
- **Image Data**: Images of cars, including front-view images, which are used for CNN-based price predictions.

Data source : [https://github.com/zalandoresearch/fashion-mnist](https://deepvisualmarketing.github.io/)

### Data Preprocessing:
- **Data Cleaning**: Handling missing values, outliers, and standardizing the data.
- **One-Hot Encoding**: For categorical variables such as body type and fuel type.
- **Standardization**: Applied to numerical features for models like KNN.
- **Data Augmentation**: For image data, using random rotations and rescaling.

## Model Architecture

### K-Nearest Neighbors (KNN):
- Applied to the tabular data with hyperparameter tuning to select the optimal number of neighbors.

### Extreme Gradient Boosting (XGBoost):
- A powerful tree-based model used for tabular data with a step-by-step hyperparameter tuning process.

### Multilayer Perceptron (MLP):
- A feedforward neural network applied to the tabular data, with extensive hyperparameter tuning to find the optimal network structure.

### Convolutional Neural Network (CNN):
- ResNet-18, a pre-trained deep learning model, was fine-tuned for car price prediction based on images.

## Main Technologies

* Python
* Pytorch
* Shap
* Sklearn
* XGBoost
* Pandas
* Google colab








This project is an assignment made in the context of my Master studies
Aix-Marseille University
Degree : Master in Econometrics, Big-Data & Statistics





