# 🎭 D³ Deepfake Detection API# 🎭 D³ Deepfake Detection API



![Python](https://img.shields.io/badge/python-3.8+-blue.svg)![Python](https://img.shields.io/badge/python-3.8+-blue.svg)

![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-red.svg)![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-red.svg)

![CUDA](https://img.shields.io/badge/CUDA-11.8%20%7C%2012.8-green.svg)![CUDA](https://img.shields.io/badge/CUDA-11.8%20%7C%2012.8-green.svg)

![Flask](https://img.shields.io/badge/Flask-3.0.0-black.svg)![Flask](https://img.shields.io/badge/Flask-3.0.0-black.svg)

![License](https://img.shields.io/badge/license-MIT-green.svg)![License](https://img.shields.io/badge/license-MIT-green.svg)



Production-ready REST API for deepfake detection using the D³-CLIP ViT-L/14 model, featuring MongoDB Atlas storage, MinIO object storage, and GPU-accelerated inference on NVIDIA RTX A4000.Production-ready REST API for deepfake detection using the D³-CLIP ViT-L/14 model, featuring MongoDB Atlas storage, MinIO object storage, and GPU-accelerated inference.



---## 📋 Table of Contents



## 📋 Table of Contents- [Features](#-features)

- [Architecture](#-architecture)

- [Features](#-features)- [Requirements](#-requirements)

- [Architecture](#-architecture)- [Installation](#-installation)

- [Requirements](#-requirements)- [Configuration](#-configuration)

- [Quick Start](#-quick-start)- [Deployment](#-deployment)

- [Installation](#-installation)- [API Endpoints](#-api-endpoints)

- [Configuration](#-configuration)- [Usage Examples](#-usage-examples)

- [Deployment](#-deployment)- [Monitoring](#-monitoring)

- [API Endpoints](#-api-endpoints)- [Troubleshooting](#-troubleshooting)

- [Usage Examples](#-usage-examples)- [Contributing](#-contributing)

- [Monitoring](#-monitoring)

- [Troubleshooting](#-troubleshooting)## ✨ Features

- [Project Structure](#-project-structure)

- [Contributing](#-contributing)### Core Capabilities

- **State-of-the-art Detection**: D³-CLIP ViT-L/14 model fine-tuned on WildRF dataset- **Production Ready**: Error handling, validation, and health checks

- **GPU Acceleration**: NVIDIA RTX A4000 (16GB VRAM) with CUDA 11.8/12.8

- **Single & Batch Processing**: Efficient handling of individual or multiple images### API Features

- **Cloud-Ready**: MongoDB Atlas for scalable, distributed storage- **RESTful Design**: Clean, intuitive API endpoints

- **Object Storage**: MinIO for secure result image storage- **Swagger UI**: Interactive API documentation at `/apidocs`

- **Comprehensive Logging**: Detailed logs for debugging and monitoring- **Real-time Health Checks**: Monitor API status

- **Production Ready**: Error handling, validation, and health checks- **Prediction History**: Query past predictions

- **Statistics Dashboard**: API usage metrics

### API Features

- **RESTful Design**: Clean, intuitive API endpoints## 🏗️ Architecture

- **Swagger UI**: Interactive API documentation at `/apidocs`

- **Real-time Health Checks**: Monitor API status```

- **Prediction History**: Query and retrieve past predictions┌─────────────────────────────────────┐

- **Statistics Dashboard**: API usage metrics and analytics│          Client Application         │

- **Batch Processing**: Handle multiple images efficiently└──────────────┬──────────────────────┘

               │ HTTP/REST

---               ▼

┌─────────────────────────────────────┐

## 🏗️ Architecture│        Flask API (Port 5001)        │

│  ┌──────────────────────────────┐   │

```│  │   Route Handlers             │   │

┌─────────────────────────────────────┐│  │   - /predict                 │   │

│          Client Application         ││  │   - /predict/batch           │   │

└──────────────┬──────────────────────┘│  │   - /predictions/*           │   │

               │ HTTP/REST│  │   - /health                  │   │

               ▼│  └───────────┬──────────────────┘   │

┌─────────────────────────────────────┐│              ▼                       │

│        Flask API (Port 5001)        ││  ┌──────────────────────────────┐   │

│  ┌──────────────────────────────┐   ││  │   Service Layer              │   │

│  │   Route Handlers             │   ││  │   - ModelService (GPU)       │   │

│  │   - /predict                 │   ││  │   - DatabaseService          │   │

│  │   - /predict/batch           │   ││  │   - StorageService           │   │

│  │   - /predictions/*           │   ││  └───────────┬──────────────────┘   │

│  │   - /health                  │   ││              ▼                       │

│  └───────────┬──────────────────┘   ││  ┌──────────────────────────────┐   │

│              ▼                       ││  │   D³-CLIP Model              │   │

│  ┌──────────────────────────────┐   ││  │   ViT-L/14 Architecture      │   │

│  │   Service Layer              │   ││  │   Fine-tuned on WildRF       │   │

│  │   - ModelService (GPU)       │   ││  └──────────────────────────────┘   │

│  │   - DatabaseService          │   │└────────┬────────────────┬───────────┘

│  │   - StorageService           │   │         │                │

│  └───────────┬──────────────────┘   │         ▼                ▼

│              ▼                       │┌─────────────┐   ┌──────────────┐

│  ┌──────────────────────────────┐   ││  MongoDB    │   │    MinIO     │

│  │   D³-CLIP Model              │   ││  Atlas      │   │  (Port 9000) │

│  │   ViT-L/14 Architecture      │   ││  (Cloud)    │   │  Local/Cloud │

│  │   Fine-tuned on WildRF       │   │└─────────────┘   └──────────────┘

│  └──────────────────────────────┘   │```

└────────┬────────────────┬───────────┘

         │                │### Technology Stack

         ▼                ▼

┌─────────────┐   ┌──────────────┐| Component | Technology | Version |

│  MongoDB    │   │    MinIO     │|-----------|-----------|---------|

│  Atlas      │   │  (Port 9000) │| **Backend** | Flask | 3.0.0 |

│  (Cloud)    │   │  Local/Cloud │| **WSGI Server** | Gunicorn | 21.2.0 |

└─────────────┘   └──────────────┘| **Deep Learning** | PyTorch | 2.0.1 |

```| **Model** | CLIP ViT-L/14 | - |

| **Database** | MongoDB Atlas | 4.6+ |

### Technology Stack| **Object Storage** | MinIO | Latest |

| **API Docs** | Flasgger | 0.9.7.1 |

| Component | Technology | Version || **CUDA** | 11.8 / 12.8 | - |

|-----------|-----------|---------|

| **Backend** | Flask | 3.0.0 |## 💻 Requirements

| **WSGI Server** | Gunicorn | 21.2.0 |

| **Deep Learning** | PyTorch | 2.0.1 |### Hardware

| **Model** | CLIP ViT-L/14 | - |- **GPU**: NVIDIA RTX A4000 (16GB VRAM) or equivalent

| **Database** | MongoDB Atlas | 4.6+ |  - Minimum: GPU with 8GB+ VRAM

| **Object Storage** | MinIO | Latest |  - CPU mode available (slower inference)

| **API Docs** | Flasgger | 0.9.7.1 |- **RAM**: 16GB+ recommended

| **CUDA** | 11.8 / 12.8 | - |- **Storage**: 10GB+ for model and dependencies



---### Software

- **OS**: Linux (Ubuntu 22.04 LTS recommended)

## 💻 Requirements- **Python**: 3.8, 3.9, or 3.10

- **CUDA**: 11.8 or 12.8

### Hardware- **NVIDIA Drivers**: 470+ (for CUDA support)

- **GPU**: NVIDIA RTX A4000 (16GB VRAM) or equivalent

  - Minimum: GPU with 8GB+ VRAM### Accounts (Optional but Recommended)

  - CPU mode available (slower inference)- **MongoDB Atlas**: Free tier available at [mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas)

- **RAM**: 16GB+ recommended- **MinIO**: Can run locally (no account needed)

- **Storage**: 10GB+ for model and dependencies

## 📦 Installation

### Software

- **OS**: Linux (Ubuntu 22.04 LTS recommended)### Option 1: Manual Installation (Recommended for Development)

- **Python**: 3.8, 3.9, or 3.10

- **CUDA**: 11.8 or 12.8#### Step 1: Clone Repository

- **NVIDIA Drivers**: 470+ (for CUDA support)

```bash

### Servicesgit clone https://github.com/hoangphat25092005/DeepFake-Project-Model.git

- **MongoDB Atlas**: Free tier available ([Sign up](https://www.mongodb.com/cloud/atlas))cd DeepFake-Project-Model/api

- **MinIO**: Self-hosted object storage```



---#### Step 2: Create Python Environment



## 🚀 Quick Start```bash

# Using conda (recommended)

### 1. Install MinIOconda create -n d3_api python=3.10

conda activate d3_api

```bash

cd /mnt/mmlab2024nas/danh/phatlh/D3/api# Or using venv

./install_minio.shpython -m venv venv

source ~/.bashrcsource venv/bin/activate  # Linux/Mac

``````



### 2. Install Python Dependencies#### Step 3: Install Dependencies



```bash```bash

pip install --user gunicorn minio pymongo dnspython flask flask-cors flasgger python-dotenv Pillow opencv-python-headless# Core dependencies

```pip install flask==3.0.0

pip install flask-cors==4.0.0

### 3. Start Servicespip install flasgger==0.9.7.1

pip install gunicorn==21.2.0

```bashpip install python-dotenv==1.0.0

./start_manual.sh

# Choose GPU when prompted (0 or 1)# Database

```pip install pymongo==4.6.1

pip install dnspython==2.6.1

### 4. Test API

# Storage

```bashpip install minio==7.2.0

curl http://localhost:5001/health

```# Image processing

pip install Pillow==10.1.0

**Access Points:**pip install opencv-python==4.8.1.78

- API: http://localhost:5001

- API Docs: http://localhost:5001/apidocs# Deep Learning (CUDA 11.8)

- MinIO Console: http://localhost:9001pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118



---# Additional ML libraries

pip install timm==0.9.12

## 📦 Installationpip install ftfy==6.1.3

pip install regex==2023.10.3

### Detailed Installation Steps```



#### Step 1: Clone Repository#### Step 4: Install MinIO



```bash```bash

git clone https://github.com/hoangphat25092005/DeepFake-Project-Model.git# Download MinIO to user bin directory

cd DeepFake-Project-Model/apimkdir -p ~/bin

```wget https://dl.min.io/server/minio/release/linux-amd64/minio -O ~/bin/minio

chmod +x ~/bin/minio

#### Step 2: Create Python Environment

# Add to PATH

```bashecho 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc

# Using conda (recommended)source ~/.bashrc

conda create -n d3_api python=3.10

conda activate d3_api# Verify installation

minio --version

# Or using venvMINIO_ACCESS_KEY=minioadmin

python -m venv venvMINIO_SECRET_KEY=minioadmin

source venv/bin/activateMINIO_BUCKET=your_bucket_name

```

# Model

#### Step 3: Install DependenciesMODEL_CHECKPOINT_PATH=your_model_checkpoint

DEVICE=cpu  # or cuda if your system support

```bash

# Core dependencies# Flask

pip install flask==3.0.0 flask-cors==4.0.0 flasgger==0.9.7.1FLASK_HOST=0.0.0.0

pip install gunicorn==21.2.0 python-dotenv==1.0.0FLASK_PORT=5000

FLASK_DEBUG=True

# Database```

pip install pymongo==4.6.1 dnspython==2.6.1

### 4. Start MinIO

# Storage

pip install minio==7.2.0```bash

docker run -d \

# Image processing  -p 9000:9000 \

pip install Pillow==10.1.0 opencv-python==4.8.1.78  -p 9001:9001 \

  --name minio \

# Deep Learning (CUDA 11.8)  -e "MINIO_ROOT_USER=minioadmin" \

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \  -e "MINIO_ROOT_PASSWORD=minioadmin" \

  --index-url https://download.pytorch.org/whl/cu118  -v ~/minio/data:/data \

  quay.io/minio/minio server /data --console-address ":9001"

# Additional ML libraries```

pip install timm==0.9.12 ftfy==6.1.3 regex==2023.10.3

```### 5. Initialize Database



#### Step 4: Install MinIO (No Sudo Required)```bash

cd api

```bashbash init_database.sh

# Use installation script```

./install_minio.sh

### 6. Run API

# Or manually

mkdir -p ~/bin```bash

wget https://dl.min.io/server/minio/release/linux-amd64/minio -O ~/bin/miniocd /path/to/D3

chmod +x ~/bin/miniopython api/app.py

echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc```

source ~/.bashrc

```API will be available at:

- **API**: http://localhost:5000

---- **Swagger Docs**: http://localhost:5000/docs

- **MinIO Console**: http://localhost:9001

## ⚙️ Configuration

## API Documentation

### Environment Variables

### Health Check

The `.env` file contains all configuration. Key settings:

```bash

```envGET /health

# Flask Configuration```

FLASK_PORT=5001

FLASK_ENV=production### Single Prediction

FLASK_HOST=0.0.0.0

```bash

# MongoDB AtlasPOST /predict

MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/Content-Type: multipart/form-data

MONGODB_DATABASE=DeepFake

DATABASE_POOL_SIZE=50{

  "image": <file>

# MinIO}

MINIO_ENDPOINT=localhost:9000```

MINIO_ACCESS_KEY=yourkey
MINIO_SECRET_KEY=yoursecret#

MINIO_SECURE=False

MINIO_BUCKET_NAME=deepfake-results```bash

POST /batch_predict

# ModelContent-Type: multipart/form-data

MODEL_CHECKPOINT_PATH=../checkpoints/finetune_wildrf/model_epoch_best.pth

MODEL_TYPE=d3_clip{

DEVICE=cuda  "images": <file[]>

}


### Database Queries

# Upload

MAX_CONTENT_LENGTH=104857600  # 100MB```bash

```GET /api/db/predictions/recent?limit=10

GET /api/db/predictions/{id}

### MongoDB Atlas SetupGET /api/db/predictions/by-label/FAKE

GET /api/db/statistics

1. **Sign Up**: [mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas)```

2. **Create Cluster**: Choose M0 (free tier)

3. **Create Database**: `DeepFake`## Testing

4. **Get Connection String**: Replace `<password>` with your password

5. **Whitelist IP**: Add `0.0.0.0/0` or your server IP```bash

# Health check

### MinIO Configurationcurl http://localhost:5000/health



MinIO stores result images:# Single prediction

curl -X POST -F "image=@test.jpg" http://localhost:5000/predict

```bash

# Start MinIO# Get recent predictions

MINIO_ROOT_USER=username MINIO_ROOT_PASSWORD=password \curl http://localhost:5000/api/db/predictions/recent

  nohup ~/bin/minio server ~/minio-data \

  --address ":9000" --console-address ":9001" > ~/minio.log 2>&1 &# Get statistics

```curl http://localhost:5000/api/db/statistics

```

---

## Project Structure

## Deployment

```

### Automated Deployment (Recommended)D3/

├── api/

```bash│   ├── app.py                  # Main application

cd /mnt/mmlab2024nas/danh/phatlh/D3/api│   ├── config.py               # Configuration

│   ├── .env                    # Environment variables

# Start services│   │

./start_manual.sh│   ├── routes/                 # API routes

│   │   ├── health_check_route.py

# Stop services│   │   ├── prediction_route.py

./stop_manual.sh│   │   └── database_route.py

```│   │

│   ├── services/               # Business logic

### Manual Deployment│   │   ├── minio_service.py

│   │   └── database_service.py

#### Start MinIO│   │

│   ├── models/                 # Database models

```bash│   │   └── prediction.py

MINIO_ROOT_USER=HoangPhatCs MINIO_ROOT_PASSWORD="25092005=))" \│   │

  nohup ~/bin/minio server ~/minio-data \│   └── utils/                  # Utilities

  --address ":9000" --console-address ":9001" > ~/minio.log 2>&1 &│       └── model_loader.py

```│

├── models/                     # ML model architecture

#### Start API├── checkpoints/                # Model weights (not in git)

└── requirements.txt

```bash```

cd /mnt/mmlab2024nas/danh/phatlh/D3/api

conda activate diffuseinst## 🛠️ Development



# Production (with Gunicorn)### Install Development Dependencies

CUDA_VISIBLE_DEVICES=0 gunicorn --workers 4 --threads 2 --timeout 600 \

  --bind 0.0.0.0:5001 app:app```bash

pip install -r requirements-dev.txt

# Development (with Flask)```

CUDA_VISIBLE_DEVICES=0 python app.py

```### Run Tests



#### Background Deployment```bash

pytest tests/

```bash```

# Start in background

CUDA_VISIBLE_DEVICES=0 nohup gunicorn --workers 4 --threads 2 --timeout 600 \### Code Formatting

  --bind 0.0.0.0:5001 app:app > api.log 2>&1 &

echo $! > api.pid```bash

black api/

# View logsflake8 api/

tail -f api.log```

```

## Database Schema

### GPU Selection

### predictions table

```bash- `id`: Primary key

# Use GPU 0 (First RTX A4000)- `original_filename`: Original image name

CUDA_VISIBLE_DEVICES=0 ./start_manual.sh- `result_filename`: Result image name

- `minio_url`: MinIO presigned URL

# Use GPU 1 (Second RTX A4000)- `label`: REAL or FAKE

CUDA_VISIBLE_DEVICES=1 ./start_manual.sh- `confidence`: Prediction confidence

```- `created_at`: Timestamp



---### prediction_batches table

- `id`: Primary key

## 📡 API Endpoints- `batch_id`: Unique batch identifier

- `total_images`: Number of images

### Base URL- `successful`: Successful predictions

```- `failed`: Failed predictions

http://localhost:5001

```## Configuration



### Swagger DocumentationSee [`.env.example`](api/.env.example) for all configuration options.

```

http://localhost:5001/apidocs## License

```

MIT License

### Endpoints Reference

## Contributors

| Endpoint | Method | Description |

|----------|--------|-------------|- Hoang Phat (@hoangphat25092005)

| `/health` | GET | Health check |

| `/api/v1/status` | GET | API status |## Contributing

| `/api/v1/predict` | POST | Single image prediction |

| `/api/v1/predict/batch` | POST | Batch prediction |1. Fork the repository

| `/api/v1/predictions/recent` | GET | Recent predictions |2. Create feature branch (`git checkout -b feature/amazing-feature`)

| `/api/v1/predictions/<id>` | GET | Get prediction by ID |3. Commit changes (`git commit -m 'Add amazing feature'`)

| `/api/v1/statistics` | GET | Usage statistics |4. Push to branch (`git push origin feature/amazing-feature`)

5. Open Pull Request

### API Examples

## Contact

#### Health Check

- GitHub: [@hoangphat25092005](https://github.com/hoangphat25092005)

```bash- Project: [DeepFake-Project-Model](https://github.com/hoangphat25092005/DeepFake-Project-Model)

curl http://localhost:5001/health

```## Acknowledgments



**Response:**- D³ (Diverse Deepfake Detection) research

```json- OpenAI CLIP architecture

{- Flask framework

  "status": "ok",- MinIO object storage
  "timestamp": "2026-01-15T10:30:45.123Z"
}
```

#### Single Image Prediction

```bash
curl -X POST http://localhost:5001/api/v1/predict \
  -F "file=@image.jpg" \
  -F "model_type=d3_clip"
```

**Response:**
```json
{
  "success": true,
  "data": {
    "prediction_id": "507f1f77bcf86cd799439011",
    "filename": "image.jpg",
    "prediction": "FAKE",
    "confidence": 0.9234,
    "processing_time": 0.156,
    "model_type": "d3_clip",
    "timestamp": "2026-01-15T10:30:45.123Z",
    "result_url": "http://localhost:9000/deepfake-results/507f1f77bcf86cd799439011_result.jpg"
  }
}
```

#### Batch Prediction

```bash
curl -X POST http://localhost:5001/api/v1/predict/batch \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg" \
  -F "model_type=d3_clip"
```

**Response:**
```json
{
  "success": true,
  "data": {
    "batch_id": "batch_507f1f77bcf86cd799439011",
    "total_images": 3,
    "results": [
      {"filename": "image1.jpg", "prediction": "REAL", "confidence": 0.9521},
      {"filename": "image2.jpg", "prediction": "FAKE", "confidence": 0.8834},
      {"filename": "image3.jpg", "prediction": "FAKE", "confidence": 0.9156}
    ],
    "processing_time": 0.823
  }
}
```

#### Get Recent Predictions

```bash
curl "http://localhost:5001/api/v1/predictions/recent?limit=5"
```

#### Get Statistics

```bash
curl http://localhost:5001/api/v1/statistics
```

**Response:**
```json
{
  "total_predictions": 1523,
  "real_count": 876,
  "fake_count": 647,
  "average_confidence": 0.9123,
  "average_processing_time": 0.187
}
```

---

## 💡 Usage Examples

### Python

```python
import requests

# Single prediction
url = "http://localhost:5001/api/v1/predict"
files = {"file": open("test_image.jpg", "rb")}
data = {"model_type": "d3_clip"}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Prediction: {result['data']['prediction']}")
print(f"Confidence: {result['data']['confidence']:.2%}")
```

### JavaScript

```javascript
// Using Fetch API
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('model_type', 'd3_clip');

fetch('http://localhost:5001/api/v1/predict', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  console.log('Prediction:', data.data.prediction);
  console.log('Confidence:', data.data.confidence);
});
```

### cURL Batch Processing

```bash
# Batch processing multiple images
curl -X POST http://localhost:5001/api/v1/predict/batch \
  -F "files=@img1.jpg" \
  -F "files=@img2.jpg" \
  -F "files=@img3.jpg" \
  -F "model_type=d3_clip"
```

---

## 📊 Monitoring

### View Logs

```bash
# API logs
tail -f /mnt/mmlab2024nas/danh/phatlh/D3/api/api.log

# MinIO logs
tail -f ~/minio.log
```

### Monitor GPU

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Detailed GPU info
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
```

### Check Services

```bash
# Check if API is running
lsof -i :5001

# Check if MinIO is running
lsof -i :9000

# Check processes
ps aux | grep gunicorn
ps aux | grep minio
```

### Health Monitoring

```bash
# Simple health check
curl http://localhost:5001/health

# Detailed status
curl http://localhost:5001/api/v1/status
```

---

## 🔧 Troubleshooting

### Common Issues

#### Port Already in Use

```bash
# Find and kill process
lsof -i :5001
kill -9 <PID>

# Or use stop script
./stop_manual.sh
```

#### GPU Not Detected

```bash
# Check GPU
nvidia-smi

# Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Set correct GPU
export CUDA_VISIBLE_DEVICES=0
```

#### Model Checkpoint Not Found

```bash
# Verify path
ls -la /mnt/mmlab2024nas/danh/phatlh/D3/checkpoints/finetune_wildrf/model_epoch_best.pth

# Update .env with correct path
```

#### MongoDB Connection Failed

```bash
# Test connection
python -c "from pymongo import MongoClient; client = MongoClient('your_url'); print(client.server_info())"

# Check MongoDB Atlas IP whitelist
# Verify internet connection
```

#### MinIO Connection Failed

```bash
# Check if running
lsof -i :9000

# Restart MinIO
kill $(cat ~/minio.pid)
MINIO_ROOT_USER=HoangPhatCs MINIO_ROOT_PASSWORD="25092005=))" \
  nohup ~/bin/minio server ~/minio-data --address ":9000" --console-address ":9001" > ~/minio.log 2>&1 &
```

### Debug Mode

Enable detailed logging:

```bash
# In .env
FLASK_DEBUG=True
LOG_LEVEL=DEBUG

# Restart
./stop_manual.sh
./start_manual.sh
```

---

## 📚 Project Structure

```
api/
├── app.py                      # Main Flask application
├── .env                        # Environment configuration
├── models/                     # Data models
│   └── database.py            # MongoDB models
├── routes/                     # API routes
│   ├── prediction_route.py   # Prediction endpoints
│   └── status_route.py        # Status endpoints
├── services/                   # Business logic
│   ├── model_service.py       # Model inference
│   ├── db_service_mongodb.py # Database operations
│   └── storage_service.py     # MinIO operations
├── config/                     # Configuration
│   └── config.py              # App configuration
├── uploads/                    # Temporary uploads
├── logs/                       # Application logs
├── start_manual.sh            # Start script ⭐
├── stop_manual.sh             # Stop script ⭐
├── install_minio.sh           # MinIO installer ⭐
├── QUICKSTART.md              # Quick start guide
├── MANUAL_DEPLOYMENT.md       # Deployment guide
├── NO_SUDO_DEPLOYMENT.md      # No-sudo guide
└── README.md                  # This file
```

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## 📄 License

MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- **D³ Model**: Diverse Deepfake Detection research
- **CLIP**: OpenAI's CLIP architecture
- **WildRF Dataset**: Training dataset

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/hoangphat25092005/DeepFake-Project-Model/issues)
- **Documentation**: See `docs/` for detailed guides
- **Email**: hoangphatpy123@gmail.com

---

## 🔗 Quick Links

- **Repository**: [github.com/hoangphat25092005/DeepFake-Project-Model](https://github.com/hoangphat25092005/DeepFake-Project-Model)
- **MongoDB Atlas**: [mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas)
- **MinIO Docs**: [min.io/docs](https://min.io/docs)
- **API Docs** (after starting): http://localhost:5001/apidocs

---

**Made with ❤️ using D³-CLIP Model** | **Powered by NVIDIA RTX A4000**

**Current Status**: ✅ MongoDB Atlas Connected | ⚡ GPU Accelerated | 🚀 Production Ready
