from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from io import BytesIO
import base64
import time
from app.config import Config

class VehicleDetector:
    """YOLO-based vehicle detector for traffic analysis"""

    # COCO dataset vehicle class IDs
    VEHICLE_CLASSES = {
        2: 'car',
        5: 'bus',
        7: 'truck',
        3: 'motorcycle'
    }

    # Proper plural mapping
    PLURAL_MAP = {
        'car': 'cars',
        'bus': 'buses',
        'truck': 'trucks',
        'motorcycle': 'motorcycles'
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
        Convert base64 string to OpenCV image with preprocessing

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

            # Enhance contrast (helps with distant vehicles)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.3)

            # Enhance sharpness (helps with small vehicles)
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.5)

            # Convert to OpenCV format (BGR)
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        except Exception as e:
            raise ValueError(f"Failed to decode image: {str(e)}")

    def detect_vehicles(self, image):
        """
        Run YOLO detection on image with optimized parameters

        Args:
            image: OpenCV image (BGR format)

        Returns:
            dict: Detection results with vehicle counts
        """
        start_time = time.time()

        results = self.model(
            image,
            conf=self.confidence_threshold,
            iou=0.3,
            imgsz=1280,
            max_det=300,
            verbose=False
        )

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
                    plural_key = self.PLURAL_MAP[vehicle_type]  # ← FIX
                    counts[plural_key] += 1
                    total_confidence += confidence
                    detection_count += 1

        # Calculate metrics
        total_vehicles = sum(counts.values())
        avg_confidence = total_confidence / detection_count if detection_count > 0 else 0
        processing_time = time.time() - start_time
        jam_level = self.calculate_jam_level(total_vehicles)

        return {
            **counts,
            'total': total_vehicles,
            'jam_level': jam_level,
            'confidence': round(avg_confidence, 2),
            'processing_time': round(processing_time, 2)
        }

    def detect_vehicles_tiled(self, image):
        """
        Tile-based detection for better small vehicle detection

        Args:
            image: OpenCV image (BGR format)

        Returns:
            dict: Detection results with vehicle counts
        """
        start_time = time.time()

        h, w = image.shape[:2]

        # Tile configuration
        tile_size = 640
        overlap = 0.2
        stride = int(tile_size * (1 - overlap))

        all_detections = []

        # Process each tile
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                # Extract tile
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                tile = image[y:y_end, x:x_end]

                # Skip tiny tiles
                if tile.shape[0] < 100 or tile.shape[1] < 100:
                    continue

                # Detect on tile
                results = self.model(
                    tile,
                    conf=self.confidence_threshold,
                    iou=0.3,
                    imgsz=640,
                    max_det=100,
                    verbose=False
                )

                # Adjust bounding box coordinates to full image
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        confidence = float(box.conf[0])

                        if cls_id in self.VEHICLE_CLASSES:
                            # Get box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                            # Adjust to full image coordinates
                            all_detections.append({
                                'class': cls_id,
                                'confidence': confidence,
                                'bbox': [x1 + x, y1 + y, x2 + x, y2 + y]
                            })

        # Remove duplicate detections
        unique_detections = self.remove_duplicates(all_detections)

        # Count vehicles
        counts = {
            'cars': 0,
            'trucks': 0,
            'buses': 0,
            'motorcycles': 0
        }

        total_confidence = 0

        for det in unique_detections:
            vehicle_type = self.VEHICLE_CLASSES[det['class']]
            plural_key = self.PLURAL_MAP[vehicle_type]  # ← FIX
            counts[plural_key] += 1
            total_confidence += det['confidence']

        total_vehicles = sum(counts.values())
        avg_confidence = total_confidence / len(unique_detections) if unique_detections else 0
        processing_time = time.time() - start_time
        jam_level = self.calculate_jam_level(total_vehicles)

        return {
            **counts,
            'total': total_vehicles,
            'jam_level': jam_level,
            'confidence': round(avg_confidence, 2),
            'processing_time': round(processing_time, 2)
        }

    def remove_duplicates(self, detections, iou_threshold=0.5):
        """
        Remove duplicate detections using Non-Maximum Suppression

        Args:
            detections: List of detection dictionaries
            iou_threshold: IoU threshold for considering detections as duplicates

        Returns:
            list: Filtered detections
        """
        if not detections:
            return []

        # Sort by confidence (highest first)
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

        keep = []

        while detections:
            best = detections.pop(0)
            keep.append(best)

            # Remove overlapping detections
            detections = [
                det for det in detections
                if self.calculate_iou(best['bbox'], det['bbox']) < iou_threshold
            ]

        return keep

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union of two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Intersection area
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # Union area
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

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
        Detect vehicles from base64 encoded image using tiled detection

        Args:
            base64_string: Base64 encoded image

        Returns:
            dict: Detection results
        """
        image = self.base64_to_image(base64_string)
        return self.detect_vehicles_tiled(image)
