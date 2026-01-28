from ultralytics import YOLO
from roboflow import Roboflow

def train_model():
    # 1. Download Dataset from Roboflow
    # REPLACE THE LINES BELOW with the snippet you copied from Roboflow!
    rf = Roboflow(api_key="qRfASJGKsv7WNtvVPzTL")
    project = rf.workspace("sanchit-scmvi").project("medicine-bottle-gnk5a")
    version = project.version(1)
    dataset = version.download("yolov8")
                
    # 2. Initialize YOLOv8 Model
    # 'yolov8n.pt' is the "Nano" model. 
    # It is the fastest to train locally and sufficient for this prototype.
    model = YOLO('yolov8n.pt')

    # 3. Train the Model
    # epochs=25 is enough to prove the concept. 
    # img=640 matches your Roboflow resize settings.
    results = model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=25,
        imgsz=640,
        plots=True
    )

    print("Training Complete. Model saved to runs/detect/train/weights/best.pt")

if __name__ == "__main__":
    train_model()


