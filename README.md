# D³ Deepfake Detection API

Flask REST API for D³ (Discrepancy Deepfake Detector) model with MinIO for model storage and PostgreSQL for data management.

## Features

- 🔍 **Deepfake Detection**: Detect AI-generated/manipulated images using state-of-the-art D³ model
- 📦 **MinIO Integration**: Store and manage model checkpoints in MinIO object storage
- 💾 **PostgreSQL Database**: Track inference history, results, and statistics
- 🚀 **REST API**: Easy-to-use HTTP endpoints for inference
- 📊 **Statistics & Analytics**: View detection statistics and history
- 🔄 **Batch Processing**: Process multiple images in a single request

## Architecture

```
api/
├── app.py                 # Main Flask application
├── config/
│   └── config.py         # Configuration management
├── models/
│   └── database.py       # SQLAlchemy models
├── routes/
│   ├── inference.py      # Inference endpoints
│   └── model.py          # Model management endpoints
├── services/
│   ├── minio_service.py  # MinIO client
│   └── inference_service.py  # Detection service
└── uploads/              # Temporary upload storage
```

## Prerequisites

- Python 3.8+
- PostgreSQL 12+
- MinIO server
- CUDA-capable GPU (recommended)

## Installation

### 1. Install Dependencies

```bash
cd api
pip install -r requirements.txt
```

### 2. Set Up PostgreSQL

```bash
# Create database
psql -U postgres
CREATE DATABASE d3_deepfake;
\q
```

### 3. Set Up MinIO

```bash
# Download and run MinIO (example with Docker)
docker run -p 9000:9000 -p 9001:9001 \
  -e "MINIO_ROOT_USER=minioadmin" \
  -e "MINIO_ROOT_PASSWORD=minioadmin" \
  quay.io/minio/minio server /data --console-address ":9001"
```

### 4. Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit .env with your settings
nano .env
```

### 5. Upload Model to MinIO

```bash
# Option 1: Use the API endpoint (after starting the server)
curl -X POST http://localhost:5000/api/model/upload \
  -F "file=@../ckpt/classifier.pth" \
  -F "name=classifier.pth"

# Option 2: Use MinIO web interface at http://localhost:9001
# Create bucket 'd3-models' and upload classifier.pth
```

### 6. Initialize Database

```bash
python app.py
# This will create all necessary tables
```

## Running the API

### Development Mode

```bash
cd api
python app.py
```

The API will be available at `http://localhost:5000`

### Production Mode (with Gunicorn)

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## API Endpoints

### Health Check

```bash
GET /health
```

### Inference Endpoints

#### 1. Predict Single Image

```bash
POST /api/inference/predict
Content-Type: multipart/form-data

Parameters:
- file: Image file (required)
- threshold: Classification threshold (optional, default: 0.5)

Response:
{
  "success": true,
  "result": {
    "prediction": 0.75,
    "is_fake": true,
    "confidence": 0.5,
    "processing_time": 0.123
  },
  "inference_id": 1
}
```

Example:
```bash
curl -X POST http://localhost:5000/api/inference/predict \
  -F "file=@test_image.jpg" \
  -F "threshold=0.5"
```

#### 2. Predict Multiple Images

```bash
POST /api/inference/predict-batch
Content-Type: multipart/form-data

Parameters:
- files: Multiple image files (required)
- threshold: Classification threshold (optional)

Response:
{
  "success": true,
  "count": 2,
  "results": [...]
}
```

Example:
```bash
curl -X POST http://localhost:5000/api/inference/predict-batch \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

#### 3. Get Inference History

```bash
GET /api/inference/history?limit=50&offset=0

Response:
{
  "success": true,
  "count": 50,
  "total": 1000,
  "results": [...]
}
```

#### 4. Get Specific Result

```bash
GET /api/inference/result/<inference_id>

Response:
{
  "success": true,
  "result": {...}
}
```

#### 5. Get Statistics

```bash
GET /api/inference/stats

Response:
{
  "success": true,
  "stats": {
    "total_inferences": 1000,
    "total_fake": 600,
    "total_real": 400,
    "fake_percentage": 60.0,
    "avg_processing_time": 0.15
  }
}
```

### Model Management Endpoints

#### 1. Get Model Info

```bash
GET /api/model/info

Response:
{
  "success": true,
  "info": {
    "loaded": true,
    "model_name": "classifier.pth",
    "architecture": "CLIP:ViT-L/14",
    "device": "cuda:1"
  }
}
```

#### 2. Load Model

```bash
POST /api/model/load

Response:
{
  "success": true,
  "message": "Model loaded successfully"
}
```

#### 3. Unload Model

```bash
POST /api/model/unload

Response:
{
  "success": true,
  "message": "Model unloaded successfully"
}
```

#### 4. List Models in MinIO

```bash
GET /api/model/list

Response:
{
  "success": true,
  "count": 3,
  "models": ["classifier.pth", "model_v2.pth"]
}
```

## Configuration

Edit `api/.env` file:

```env
# PostgreSQL
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=d3_deepfake

# MinIO
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_SECURE=False
MINIO_BUCKET_NAME=d3-models

# CUDA
CUDA_DEVICE=1
```

## Database Schema

### InferenceResult
- id: Primary key
- image_name: Original filename
- prediction: Probability score (0-1)
- is_fake: Boolean result
- confidence: Confidence score
- threshold: Used threshold
- processing_time: Inference time in seconds
- created_at: Timestamp
- user_id: Foreign key (optional)

### User
- id: Primary key
- username: Unique username
- email: Unique email
- api_key: Optional API key
- created_at: Timestamp
- is_active: Boolean status

### RequestLog
- id: Primary key
- endpoint: API endpoint
- method: HTTP method
- status_code: Response status
- ip_address: Client IP
- response_time: Response time
- created_at: Timestamp

## Python Client Example

```python
import requests

# Single image prediction
with open('test_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/inference/predict',
        files={'file': f},
        data={'threshold': 0.5}
    )
    result = response.json()
    print(f"Is Fake: {result['result']['is_fake']}")
    print(f"Confidence: {result['result']['confidence']}")

# Get statistics
response = requests.get('http://localhost:5000/api/inference/stats')
stats = response.json()
print(stats['stats'])
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in config
- Use FP16: Set `USE_FP16=True` in config
- Free GPU memory: `curl -X POST http://localhost:5000/api/model/unload`

### MinIO Connection Error
- Check MinIO is running: `curl http://localhost:9000/minio/health/live`
- Verify credentials in `.env`

### Database Connection Error
- Check PostgreSQL is running: `pg_isready`
- Verify credentials and database exists

## Performance Optimization

1. **Model Loading**: Load model once at startup
2. **FP16 Inference**: Enable half precision for 50% memory reduction
3. **Batch Processing**: Use batch endpoints for multiple images
4. **Database Indexing**: Add indexes on frequently queried columns
5. **Caching**: Implement Redis for result caching

## Security Considerations

- Use strong SECRET_KEY in production
- Implement API rate limiting
- Add authentication/authorization
- Validate file uploads (size, type)
- Use HTTPS in production
- Secure MinIO and PostgreSQL credentials

## License

Same as the main D³ project.

## Citation

```bibtex
@inproceedings{yang2025d3,
  title={D3: Scaling Up Deepfake Detection by Learning from Discrepancy},
  author={Yang, Yongqi and Qian, Zhihao and Zhu, Ye and Russakovsky, Olga and Wu, Yu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```
