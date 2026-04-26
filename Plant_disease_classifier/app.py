import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image


# Model definition
class PlantDisease(nn.Module):
  def __init__(self, num_classes, img_size = 128):
    super().__init__()

    self.feature_extractor = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
        nn.ELU(),
        nn.MaxPool2d(kernel_size = 2),

        nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
        nn.ELU(),
        nn.MaxPool2d(kernel_size = 2),

        nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
        nn.ELU(),
        nn.MaxPool2d(kernel_size = 2),
        nn.Flatten()
    )

    final_size = img_size // 8
    self.classifier = nn.Linear(128 * final_size * final_size, num_classes)

  def forward(self, x):
    x = self.feature_extractor(x)
    x = self.classifier(x)
    return x


# load model
@st.cache_resource
def load_model():
  device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = PlantDisease(num_classes = 38, img_size = 128)
  model.load_state_dict(torch.load('Plant_disease_classifier/best_model.pth', map_location = device, weights_only = True))
  model.eval()
  return model.to(device), device


# load class names
@st.cache_resource
def load_class_names():
  with open('Plant_disease_classifier/class_names.txt', 'r') as f:
    return [line.strip() for line in f.readlines()]


# transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


# UI
st.title('Plant Disease Classifier')
st.write('Upload a plant leaf image for disease classification')

uploaded_file = st.file_uploader('choose an image', type = ['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
  image = Image.open(uploaded_file).convert('RGB')
  st.image(image, caption = 'Uploaded image', use_container_width = True)

  with st.spinner('Classifying...'):
    model, device = load_model()
    class_names = load_class_names()

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
      output = model(input_tensor)
      probabilities = torch.softmax(output, dim = 1)
      predicted_idx = probabilities.argmax(dim=1).item()
      confidence = probabilities[0][predicted_idx].item()

    predicted_class = class_names[predicted_idx]


# results
  st.success(f'Prediction: {predicted_class}')
  st.metric(label = 'Confidence', value = f'{confidence * 100:.2f}%')

  st.subheader('Top 3 predictions')
  top3_probs, top3_indices = torch.topk(probabilities, 3, dim=1)
  for i in range(3):
    idx = top3_indices[0][i].item()
    prob = top3_probs[0][i].item()
    st.progress(prob)
    st.write(f'{i + 1}. {class_names[idx]} - {prob*100:.2f}%')




