from flask import Flask, request, jsonify
from flask_cors import CORS
from app.detector import VehicleDetector
from app.config import Config
import os

app = Flask(__name__)
CORS(app)

# Load configuration
app.config.from_object(Config)

# Initialize detector
print("🚀 Initializing Vehicle Detector...")
detector = VehicleDetector()
print("✅ Detector ready!")

@app.route('/', methods=['GET'])
def home():
    """API info endpoint"""
    return jsonify({
        'name': 'Border Camera Vehicle Detector API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            'POST /detect': 'Detect vehicles from base64 image',
            'GET /health': 'Health check',
            'GET /': 'API info'
        }
    }), 200

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': detector.model_name,
        'ready': True
    }), 200

@app.route('/detect', methods=['POST'])
def detect():
    """
    Detect vehicles in image

    Request body:
    {
        "image": "base64_encoded_image_string"
    }

    Response:
    {
        "cars": 5,
        "trucks": 2,
        "buses": 1,
        "motorcycles": 0,
        "total": 8,
        "jam_level": "medium",
        "confidence": 0.85,
        "processing_time": 1.23
    }
    """
    try:
        # Validate request
        if not request.json:
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.json
        base64_image = data.get('image')

        if not base64_image:
            return jsonify({'error': 'Missing "image" field in request body'}), 400

        # Detect vehicles
        results = detector.detect_from_base64(base64_image)

        return jsonify(results), 200

    except ValueError as e:
        return jsonify({'error': f'Invalid image data: {str(e)}'}), 400
    except Exception as e:
        app.logger.error(f"Detection error: {str(e)}")
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
