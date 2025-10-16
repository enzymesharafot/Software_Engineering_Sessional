from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torchvision.transforms as transforms
import torchvision
import torch
import io
import logging

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (Allow frontend to communicate with backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load Model
model = torchvision.models.densenet201(weights=None)
model.classifier = torch.nn.Sequential(torch.nn.Linear(1920, 8))

try:
    state_dict = torch.load('Models/model_1.pt', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")

# Class labels
class_names = [
    'BA-cellulitis', 'BA-impetigo', 'FU-athlete-foot', 'FU-nail-fungus',
    'FU-ringworm', 'PA-cutaneous-larva-migrans', 'VI-chickenpox', 'VI-shingles'
]

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/prediction/")
async def predict(file: UploadFile = File(...)):
    try:
        logging.info(f"Received file: {file.filename}")

        # Read image file
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Transform image
        image = transform(image).unsqueeze(0).to('cpu')

        # Perform inference
        with torch.inference_mode():
            probabilities = torch.softmax(model(image), dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)
            predicted_class = class_names[predicted_idx.item()]
            confidence_score = confidence.item() * 100  # Convert to percentage

        logging.info(f"Prediction: {predicted_class}, Confidence: {confidence_score:.2f}%")

        return {
            "Predicted Disease": predicted_class,
            "Confidence": round(confidence_score, 2)
        }

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return {"error": "Failed to process the image."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)