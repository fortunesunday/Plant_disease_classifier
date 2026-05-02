# PLANT DISEASE CLASSIFICATION USING CONVOLUTED NEURAL NETWORKS (CNN)
## **Project Overview**
Plant Disease Classifier is a web application that detects crop leaf diseases from uploaded images and returns instant predictions to support early, informed decision-making. Built with a CNN model in PyTorch and deployed via Streamlit, it provides fast and accessible diagnostics through any web browser.
Plant diseases account for up to 40% of annual crop losses globally, contributing to food insecurity and its wider impacts. Currently, famers rely on visual inspection which often leads to misdiagnosis and improper treatment. This solution offers a faster, more reliable, and scalable approach to disease detection, helping improve yield and reduce costs.
## **Features**
- Image uupload Interface: Upload plant leaf images directly from your devic
- CNN-Based Classification: Powered by a trained PyTorch Convolutional Neural Network
- nstant Predictions: Get results within seconds
- Confidence Score Display: Shos Prediction Certainty
- User Friendly Interface: Simple to use
- Scalable Solution: Works without requiring expert intervention
- Supports Early Detection: Helps prevent disease spread and crop losses
## **Demo**
[Access the live app here](https://plantdiseaseclassifier-druby28cepgwq7fehcg7wr.streamlit.app/)
## Dataset
* Dataset used: [Link to dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
* classes: 38
* Total Images: 87,000
*  Data split:
    - Training: 70,295
    - Validation: 17,572
    - Test: 33
## **Model Architecture**
- Model Type:Convoluted Neural Network (CNN)
- Frame Work: PyTorch
- Input size: 224 * 224
- Layers:
    * Convolution Layers(Feature Extraction)
    * MaxPooling Layers
    * Fully connected (Dense) Layers
    * Dropout(to reduce overfitting)
- Activation Function: ELU
- Output Layer: softmax(multi-class classification)
- Loss Function: CrossEntropyLoss
- Optimizer: Adam
## WebApp Interface
<img width="1280" height="348" alt="app_interface1" src="https://github.com/user-attachments/assets/adb22abb-ec76-4512-84aa-c24a3732d903" />
<img width="1280" height="610" alt="app_interface2" src="https://github.com/user-attachments/assets/587c18f4-66d5-4a36-a8cc-08e5a7ad40a7" />


## **Training Process**
- Image preprocessing: Resizing, ToTensor
- Data Augmentation: RandomHorizontalFlip, RandomRotation
- Batch size:
- Epochs: 10
- Training Accuracy: 88.83%
- Validation Accuraacy: 88.98
## **Results**
- Model performs well on common plant disease and healthy leaves
- Predictions include class labels and confidence scores
## **Tech Stack**
-  Python
- Pytorch
- Streamlit
- Numpy
- Matplotlib
