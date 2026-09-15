from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="C:/inspection/missing_component_detection.yolov8/data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        workers=0,
        name="pcb_v2"
    )

if __name__ == "__main__":
    main()