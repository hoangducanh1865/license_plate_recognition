from ultralytics import YOLO


class LicensePlateDetector:
    def __init__(self, model_path, device="cpu"):
        self.model = YOLO(model=model_path)
        self.device = device

    def detect(self, frame):

        # @QUESTION: Why [0]? How do we know to extract the license plate box list by this "boxes.data.tolist()"?
        # ANSWER: See in my notes
        return self.model(frame, device=self.device)[0].boxes.data.tolist()
