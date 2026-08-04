import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO

# Initialize FastAPI App
app = FastAPI(title="Pollen Counter API", description="YOLO Inference API for CBC-Apps")

# Allow local Laravel to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the i25 Model (Global instance to keep it in VRAM)
MODEL_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 
    '..', 
    'runs', 'detect', 'i24_260T_16V_YOLO11N-P2_94P_52B_96R_97A', 'weights', 'best.pt'
))

print(f"[INFO] Loading YOLO Model from {MODEL_PATH}")
try:
    model = YOLO(MODEL_PATH)
    # Perform a dummy prediction to warm up the GPU
    dummy_img = np.zeros((512, 512, 3), dtype=np.uint8)
    model.predict(dummy_img, verbose=False)
    print("[INFO] Model loaded and warmed up successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    model = None


class PredictResponse(BaseModel):
    count: int
    boxes: list[list[float]]  # [x1, y1, x2, y2]
    inference_time_ms: float


@app.get("/")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse)
async def predict(image: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
        
    try:
        # Read image bytes into numpy array
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format.")
            
        # Run inference
        # Conf=0.25 is default, max_det=5000 handles dense pollen
        results = model.predict(img, conf=0.25, max_det=5000, verbose=False)
        
        result = results[0]
        boxes_xyxy = result.boxes.xyxy.cpu().numpy().tolist()
        
        return PredictResponse(
            count=len(boxes_xyxy),
            boxes=boxes_xyxy,
            inference_time_ms=result.speed['inference']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    # Run the API locally on port 8001
    uvicorn.run("api:app", host="127.0.0.1", port=8001, reload=True)
