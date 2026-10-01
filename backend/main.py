import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Real/Fake Detection API")

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    
    # Matching the exact 5-layer classification head from training
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(512, 2)
    )
    
    # Load your saved weights
    model.load_state_dict(torch.load("best_efficientnet.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()
class_names = ["Fake", "Real"]

# Transform pipeline matching evaluation notebook
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        conf, predicted = torch.max(probabilities, 1)
        
    class_name = class_names[predicted.item()]
    confidence = conf.item() * 100
    
    return {
        "prediction": class_name,
        "confidence": round(confidence, 2)
    }

@app.get("/")
def home():
    return {"message": "Real/Fake Detection Backend is Running"}
