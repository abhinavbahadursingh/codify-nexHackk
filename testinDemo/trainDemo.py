from ultralytics import YOLO
import os

def main():
    model = YOLO(r"E:\codify_hackquanta\testinDemo\best.pt")

    # Check if a checkpoint exists before resuming
    run_dir = r"runs\detect\crash_detection_model"
    resume_flag = os.path.exists(os.path.join(run_dir, "weights", "last.pt"))

    results = model.train(
        data=r"C:\Users\abhin\Downloads\Traffic Density Prediction.v9-updateddataset.yolov11\data.yaml",
        epochs=50,
        imgsz=512,
        batch=2,
        name="crash_detection_model",
        device=0,
        resume= False
    )

if __name__ == "__main__":
    main()
