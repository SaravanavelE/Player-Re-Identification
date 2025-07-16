# app.py
from ultralytics import YOLO
import cv2
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load YOLOv1 player detection model (trained for players only)
model = YOLO("best.pt")  # Replace with correct path if needed

# Load pre-trained ResNet for appearance embeddings
resnet = models.resnet50(pretrained=True)
resnet.eval()

# Image pre-processing for ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# Function to get embedding vector for a cropped player image
def get_embedding(image_crop):
    try:
        img = Image.fromarray(image_crop)
        img = transform(img).unsqueeze(0)
        with torch.no_grad():
            embedding = resnet(img)
        return embedding.squeeze().numpy()
    except Exception as e:
        print("Embedding error:", e)
        return np.zeros(1000)  # fallback embedding


# Function to detect and crop player boxes from video
def detect_and_crop_players(video_path, model):
    cap = cv2.VideoCapture(video_path)
    cropped_players = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO detection on each frame
        results = model(frame)[0]
        for box in results.boxes:
            if int(box.cls) == 0:  # class 0: player
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                if crop.size != 0:
                    cropped_players.append(crop)

    cap.release()
    return cropped_players


# Function to compute cosine similarity and find best matches
def match_players(embeddings_tacticam, embeddings_broadcast):
    matches = {}
    for i, tac_embed in enumerate(embeddings_tacticam):
        sims = cosine_similarity([tac_embed], embeddings_broadcast)[0]
        best_match = int(np.argmax(sims))
        matches[i] = best_match
    return matches


# MAIN PIPELINE
if __name__ == "__main__":
    print("🔍 Detecting and cropping players from broadcast.mp4...")
    broadcast_crops = detect_and_crop_players("broadcast.mp4", model)
    print(f"Broadcast players detected: {len(broadcast_crops)}")

    print("Detecting and cropping players from tacticam.mp4...")
    tacticam_crops = detect_and_crop_players("tacticam.mp4", model)
    print(f"Tacticam players detected: {len(tacticam_crops)}")

    print("Generating embeddings using ResNet...")
    broadcast_embeddings = [get_embedding(crop) for crop in broadcast_crops]
    tacticam_embeddings = [get_embedding(crop) for crop in tacticam_crops]

    print("Matching tacticam players to broadcast players...")
    matches = match_players(tacticam_embeddings, broadcast_embeddings)

    print("Player Mapping Results (Tacticam → Broadcast):")
    for tac_id, broad_id in matches.items():
        print(f"  Tacticam Player {tac_id} → Broadcast Player {broad_id}")
