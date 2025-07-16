# Player Re-Identification (Cross-Camera Mapping)

This project maps players across two camera views using:
- YOLOv1 for player detection
- ResNet for appearance embedding
- Cosine similarity for matching

## Files
- `app.py`: Main script
- `best.pt`: YOLOv1 model (not included in repo)
- `broadcast.mp4`, `tacticam.mp4`: Input videos (not included)

## Run
```bash
python app.py
