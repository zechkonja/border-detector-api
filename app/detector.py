from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import time
from app.config import Config

class VehicleDetector:
    """YOLO-based vehicle detector for traffic analysis"""

    # COCO dataset vehicle class IDs
    VEHICLE_CLASSES = {
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck'
    }

    def __init__(self, model_name=None):
        """Initialize YOLO model"""
        self.model_name = model_name or Config.YOLO_MODEL
        self.confidence_threshold = Config.CONFIDENCE_THRESHOLD

        print(f"Loading YOLO model: {self.model_name}")
        self.model = YOLO(self.model_name)
        print(f"Model loaded successfully!")

    def base64_to_image(self, base64_string):
        """
        Convert base64 string to OpenCV image

        Args:
            base64_string: Base64 encoded image string

        Returns:
            numpy.ndarray: OpenCV image in BGR format
        """
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]

            # Decode base64
            image_bytes = base64.b64decode(base64_string)

            # Convert to PIL Image
            image = Image.open(BytesIO(image_bytes))

            # Convert to OpenCV format (BGR)
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        except Exception as e:
            raise ValueError(f"Failed to decode image: {str(e)}")

    def detect_vehicles(self, image):
        """
        Run YOLO detection on image

        Args:
            image: OpenCV image (BGR format)

        Returns:
            dict: Detection results with vehicle counts
        """
        start_time = time.time()

        # Run inference
        results = self.model(image, conf=self.confidence_threshold, verbose=False)

        # Initialize counts
        counts = {
            'cars': 0,
            'trucks': 0,
            'buses': 0,
            'motorcycles': 0
        }

        total_confidence = 0
        detection_count = 0

        # Process detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])

                if cls_id in self.VEHICLE_CLASSES:
                    vehicle_type = self.VEHICLE_CLASSES[cls_id]
                    counts[vehicle_type + 's'] += 1
                    total_confidence += confidence
                    detection_count += 1

        # Calculate metrics
        total_vehicles = sum(counts.values())
        avg_confidence = total_confidence / detection_count if detection_count > 0 else 0
        processing_time = time.time() - start_time

        # Determine jam level
        jam_level = self.calculate_jam_level(total_vehicles)

        return {
            **counts,
            'total': total_vehicles,
            'jam_level': jam_level,
            'confidence': round(avg_confidence, 2),
            'processing_time': round(processing_time, 2)
        }

    def calculate_jam_level(self, total_vehicles):
        """
        Calculate traffic jam level based on vehicle count

        Args:
            total_vehicles: Total number of detected vehicles

        Returns:
            str: Jam level (low, medium, high, critical)
        """
        if total_vehicles < Config.JAM_LEVEL_LOW:
            return 'low'
        elif total_vehicles < Config.JAM_LEVEL_MEDIUM:
            return 'medium'
        elif total_vehicles < Config.JAM_LEVEL_HIGH:
            return 'high'
        else:
            return 'critical'

    def detect_from_base64(self, base64_string):
        """
        Detect vehicles from base64 encoded image

        Args:
            base64_string: Base64 encoded image

        Returns:
            dict: Detection results
        """
        image = self.base64_to_image(base64_string)
        return self.detect_vehicles(image)
