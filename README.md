# 🎭 D³ Deepfake Detection API

Advanced deepfake detection system using D³ (Diverse Deepfake Detection) with CLIP architecture, PostgreSQL database, and MinIO object storage.

## Features

- **State-of-the-art Detection**: D³-CLIP model for accurate deepfake detection
- **Single & Batch Prediction**: Process one or multiple images at once
- **PostgreSQL Database**: Persistent storage for prediction results
- **MinIO Object Storage**: Efficient storage for result images
- **RESTful API**: Well-documented REST endpoints
- **Swagger Documentation**: Interactive API documentation
- **Production Ready**: Error handling, logging, and validation

## Prerequisites

- Python 3.8+
- PostgreSQL 12+
- Docker (for MinIO)
- CUDA-capable GPU (optional, CPU mode available)

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/hoangphat25092005/DeepFake-Project-Model.git
cd DeepFake-Project-Model
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup Environment

Create `.env` file in `api/` directory:

```env
# Database
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/DeepFake

# MinIO
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=your_bucket_name

# Model
MODEL_CHECKPOINT_PATH=your_model_checkpoint
DEVICE=cpu  # or cuda if your system support

# Flask
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=True
```

### 4. Start MinIO

```bash
docker run -d \
  -p 9000:9000 \
  -p 9001:9001 \
  --name minio \
  -e "MINIO_ROOT_USER=minioadmin" \
  -e "MINIO_ROOT_PASSWORD=minioadmin" \
  -v ~/minio/data:/data \
  quay.io/minio/minio server /data --console-address ":9001"
```

### 5. Initialize Database

```bash
cd api
bash init_database.sh
```

### 6. Run API

```bash
cd /path/to/D3
python api/app.py
```

API will be available at:
- **API**: http://localhost:5000
- **Swagger Docs**: http://localhost:5000/docs
- **MinIO Console**: http://localhost:9001

## API Documentation

### Health Check

```bash
GET /health
```

### Single Prediction

```bash
POST /predict
Content-Type: multipart/form-data

{
  "image": <file>
}
```

### Batch Prediction

```bash
POST /batch_predict
Content-Type: multipart/form-data

{
  "images": <file[]>
}
```

### Database Queries

```bash
GET /api/db/predictions/recent?limit=10
GET /api/db/predictions/{id}
GET /api/db/predictions/by-label/FAKE
GET /api/db/statistics
```

## Testing

```bash
# Health check
curl http://localhost:5000/health

# Single prediction
curl -X POST -F "image=@test.jpg" http://localhost:5000/predict

# Get recent predictions
curl http://localhost:5000/api/db/predictions/recent

# Get statistics
curl http://localhost:5000/api/db/statistics
```

## Project Structure

```
D3/
├── api/
│   ├── app.py                  # Main application
│   ├── config.py               # Configuration
│   ├── .env                    # Environment variables
│   │
│   ├── routes/                 # API routes
│   │   ├── health_check_route.py
│   │   ├── prediction_route.py
│   │   └── database_route.py
│   │
│   ├── services/               # Business logic
│   │   ├── minio_service.py
│   │   └── database_service.py
│   │
│   ├── models/                 # Database models
│   │   └── prediction.py
│   │
│   └── utils/                  # Utilities
│       └── model_loader.py
│
├── models/                     # ML model architecture
├── checkpoints/                # Model weights (not in git)
└── requirements.txt
```

## 🛠️ Development

### Install Development Dependencies

```bash
pip install -r requirements-dev.txt
```

### Run Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black api/
flake8 api/
```

## Database Schema

### predictions table
- `id`: Primary key
- `original_filename`: Original image name
- `result_filename`: Result image name
- `minio_url`: MinIO presigned URL
- `label`: REAL or FAKE
- `confidence`: Prediction confidence
- `created_at`: Timestamp

### prediction_batches table
- `id`: Primary key
- `batch_id`: Unique batch identifier
- `total_images`: Number of images
- `successful`: Successful predictions
- `failed`: Failed predictions

## Configuration

See [`.env.example`](api/.env.example) for all configuration options.

## License

MIT License

## Contributors

- Hoang Phat (@hoangphat25092005)

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## Contact

- GitHub: [@hoangphat25092005](https://github.com/hoangphat25092005)
- Project: [DeepFake-Project-Model](https://github.com/hoangphat25092005/DeepFake-Project-Model)

## Acknowledgments

- D³ (Diverse Deepfake Detection) research
- OpenAI CLIP architecture
- Flask framework
- MinIO object storage