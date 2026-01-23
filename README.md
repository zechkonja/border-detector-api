# Border Camera Detector API

AI-powered vehicle detection API for border traffic monitoring using YOLOv8.

## Features

- 🚗 Detects cars, trucks, buses, and motorcycles
- 📊 Calculates traffic jam levels
- ⚡ Fast inference with YOLOv8
- 🔄 Base64 image input support
- 📈 Confidence scores and processing time metrics

## API Endpoints

### `POST /detect`

Detect vehicles in an image.

**Request:**
```json  
{  
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."  
}  
