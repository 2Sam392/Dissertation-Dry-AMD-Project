import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import io

# Define the model class
class HybridModel(nn.Module):
    def __init__(self, num_classes):
        super(HybridModel, self).__init__()
        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.vgg16.classifier = nn.Sequential(*list(self.vgg16.classifier.children())[:-1])
        self.resnet50.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(4096 + 2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        vgg_features = self.vgg16(x)
        resnet_features = self.resnet50(x)
        combined_features = torch.cat((vgg_features, resnet_features), dim=1)
        out = self.classifier(combined_features)
        return out

# Load model
num_classes = 2  # Adjust according to your model
model = HybridModel(num_classes=num_classes)
model.load_state_dict(torch.load('final_model_hybrid.pth', map_location=torch.device('cpu')))
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Streamlit app
st.title('Dry AMD Prediction')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image = transform(image).unsqueeze(0)

    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    ans = int(predicted.item())
    if ans == 1:
        st.success('No AMD Detected')
    elif ans == 0:
        st.error('Dry AMD Detected')
