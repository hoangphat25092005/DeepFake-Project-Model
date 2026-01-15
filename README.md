# 🎭 D³ Deepfake Detection API# 🎭 D³ Deepfake Detection API



![Python](https://img.shields.io/badge/python-3.8+-blue.svg)![Python](https://img.shields.io/badge/python-3.8+-blue.svg)

![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-red.svg)![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-red.svg)

![CUDA](https://img.shields.io/badge/CUDA-11.8%20%7C%2012.8-green.svg)![CUDA](https://img.shields.io/badge/CUDA-11.8%20%7C%2012.8-green.svg)

![Flask](https://img.shields.io/badge/Flask-3.0.0-black.svg)![Flask](https://img.shields.io/badge/Flask-3.0.0-black.svg)

![License](https://img.shields.io/badge/license-MIT-green.svg)![License](https://img.shields.io/badge/license-MIT-green.svg)



Production-ready REST API for deepfake detection using the D³-CLIP ViT-L/14 model, featuring MongoDB Atlas storage, MinIO (Docker) object storage, and GPU-accelerated inference.Production-ready REST API for deepfake detection using the D³-CLIP ViT-L/14 model, featuring MongoDB Atlas storage, MinIO (Docker) object storage, and GPU-accelerated inference.



------



## 📋 Table of Contents## 📋 Table of Contents



- [Features](#-features)- [Features](#-features)

- [Architecture](#-architecture)- [Architecture](#-architecture)

- [Requirements](#-requirements)- [Requirements](#-requirements)

- [Installation](#-installation)- [Installation](#-installation)

- [Configuration](#-configuration)- [Configuration](#-configuration)

- [API Endpoints](#-api-endpoints)- [API Endpoints](#-api-endpoints)

- [Usage Examples](#-usage-examples)- [Usage Examples](#-usage-examples)

- [Monitoring](#-monitoring)- [Monitoring](#-monitoring)

- [Troubleshooting](#-troubleshooting)- [Troubleshooting](#-troubleshooting)## ✨ Features

- [Project Structure](#-project-structure)

- [Contributing](#-contributing)- [Project Structure](#-project-structure)



---- [Contributing](#-contributing)### Core Capabilities



## ✨ Features- **State-of-the-art Detection**: D³-CLIP ViT-L/14 model fine-tuned on WildRF dataset- **Production Ready**: Error handling, validation, and health checks



### Core Capabilities- **GPU Acceleration**: NVIDIA RTX A4000 (16GB VRAM) with CUDA 11.8/12.8

- **State-of-the-art Detection**: D³-CLIP ViT-L/14 model fine-tuned on WildRF dataset

- **GPU Acceleration**: Optimized for NVIDIA GPUs with CUDA support- **Single & Batch Processing**: Efficient handling of individual or multiple images### API Features

- **Single & Batch Processing**: Efficient handling of individual or multiple images

- **Cloud-Ready**: MongoDB Atlas for scalable, distributed storage- **Cloud-Ready**: MongoDB Atlas for scalable, distributed storage- **RESTful Design**: Clean, intuitive API endpoints

- **Object Storage**: MinIO (Docker) for secure result image storage

- **Comprehensive Logging**: Detailed logs for debugging and monitoring- **Object Storage**: MinIO for secure result image storage- **Swagger UI**: Interactive API documentation at `/apidocs`

- **Production Ready**: Error handling, validation, and health checks

- **Comprehensive Logging**: Detailed logs for debugging and monitoring- **Real-time Health Checks**: Monitor API status

### API Features

- **RESTful Design**: Clean, intuitive API endpoints- **Production Ready**: Error handling, validation, and health checks- **Prediction History**: Query past predictions

- **Swagger UI**: Interactive API documentation at `/apidocs`

- **Real-time Health Checks**: Monitor API status- **Statistics Dashboard**: API usage metrics

- **Prediction History**: Query and retrieve past predictions

- **Statistics Dashboard**: API usage metrics and analytics### API Features

- **Batch Processing**: Handle multiple images efficiently

- **RESTful Design**: Clean, intuitive API endpoints## 🏗️ Architecture

---

- **Swagger UI**: Interactive API documentation at `/apidocs`

## 🏗️ Architecture

- **Real-time Health Checks**: Monitor API status```

```

┌─────────────────────────────────────┐- **Prediction History**: Query and retrieve past predictions┌─────────────────────────────────────┐

│          Client Application         │

└──────────────┬──────────────────────┘- **Statistics Dashboard**: API usage metrics and analytics│          Client Application         │

               │ HTTP/REST

               ▼- **Batch Processing**: Handle multiple images efficiently└──────────────┬──────────────────────┘

┌─────────────────────────────────────┐

│        Flask API (Port 5001)        │               │ HTTP/REST

│  ┌──────────────────────────────┐   │

│  │   Route Handlers             │   │---               ▼

│  │   - /predict                 │   │

│  │   - /predict/batch           │   │┌─────────────────────────────────────┐

│  │   - /predictions/*           │   │

│  │   - /health                  │   │## 🏗️ Architecture│        Flask API (Port 5001)        │

│  └───────────┬──────────────────┘   │

│              ▼                       ││  ┌──────────────────────────────┐   │

│  ┌──────────────────────────────┐   │

│  │   Service Layer              │   │```│  │   Route Handlers             │   │

│  │   - ModelService (GPU)       │   │

│  │   - DatabaseService          │   │┌─────────────────────────────────────┐│  │   - /predict                 │   │

│  │   - StorageService           │   │

│  └───────────┬──────────────────┘   ││          Client Application         ││  │   - /predict/batch           │   │

│              ▼                       │

│  ┌──────────────────────────────┐   │└──────────────┬──────────────────────┘│  │   - /predictions/*           │   │

│  │   D³-CLIP Model              │   │

│  │   ViT-L/14 Architecture      │   │               │ HTTP/REST│  │   - /health                  │   │

│  │   Fine-tuned on WildRF       │   │

│  └──────────────────────────────┘   │               ▼│  └───────────┬──────────────────┘   │

└────────┬────────────────┬───────────┘

         │                │┌─────────────────────────────────────┐│              ▼                       │

         ▼                ▼

┌─────────────┐   ┌──────────────┐│        Flask API (Port 5001)        ││  ┌──────────────────────────────┐   │

│  MongoDB    │   │    MinIO     │

│  Atlas      │   │   (Docker)   ││  ┌──────────────────────────────┐   ││  │   Service Layer              │   │

│  (Cloud)    │   │  Port 9000   │

└─────────────┘   └──────────────┘│  │   Route Handlers             │   ││  │   - ModelService (GPU)       │   │

```

│  │   - /predict                 │   ││  │   - DatabaseService          │   │

### Technology Stack

│  │   - /predict/batch           │   ││  │   - StorageService           │   │

| Component | Technology | Version |

|-----------|-----------|---------|│  │   - /predictions/*           │   ││  └───────────┬──────────────────┘   │

| **Backend** | Flask | 3.0.0 |

| **WSGI Server** | Gunicorn | 21.2.0 |│  │   - /health                  │   ││              ▼                       │

| **Deep Learning** | PyTorch | 2.0.1 |

| **Model** | CLIP ViT-L/14 | - |│  └───────────┬──────────────────┘   ││  ┌──────────────────────────────┐   │

| **Database** | MongoDB Atlas | 4.6+ |

| **Object Storage** | MinIO (Docker) | Latest |│              ▼                       ││  │   D³-CLIP Model              │   │

| **API Docs** | Flasgger | 0.9.7.1 |

| **CUDA** | 11.8 / 12.8 | - |│  ┌──────────────────────────────┐   ││  │   ViT-L/14 Architecture      │   │



---│  │   Service Layer              │   ││  │   Fine-tuned on WildRF       │   │



## 📦 Requirements│  │   - ModelService (GPU)       │   ││  └──────────────────────────────┘   │



### Hardware│  │   - DatabaseService          │   │└────────┬────────────────┬───────────┘

- **GPU**: NVIDIA GPU with CUDA support (recommended: 8GB+ VRAM)

- **RAM**: 16GB minimum│  │   - StorageService           │   │         │                │

- **Storage**: 10GB for model + dependencies

│  └───────────┬──────────────────┘   │         ▼                ▼

### Software

- **Python**: 3.8, 3.9, or 3.10│              ▼                       │┌─────────────┐   ┌──────────────┐

- **CUDA**: 11.8 or 12.8

- **Docker**: Latest version (for MinIO)│  ┌──────────────────────────────┐   ││  MongoDB    │   │    MinIO     │

- **MongoDB Atlas Account**: Free M0 tier available

│  │   D³-CLIP Model              │   ││  Atlas      │   │  (Port 9000) │

### External Services

- **MongoDB Atlas**: Cloud database (free tier available)│  │   ViT-L/14 Architecture      │   ││  (Cloud)    │   │  Local/Cloud │

- **MinIO**: Object storage (runs in Docker container)

│  │   Fine-tuned on WildRF       │   │└─────────────┘   └──────────────┘

---

│  └──────────────────────────────┘   │```

## 🚀 Installation

└────────┬────────────────┬───────────┘

### 1. Clone Repository

         │                │### Technology Stack

```bash

cd /mnt/mmlab2024nas/danh/phatlh/D3/api         ▼                ▼

```

┌─────────────┐   ┌──────────────┐| Component | Technology | Version |

### 2. Set Up Python Environment

│  MongoDB    │   │    MinIO     │|-----------|-----------|---------|

```bash

# Using conda (recommended)│  Atlas      │   │  (Port 9000) │| **Backend** | Flask | 3.0.0 |

conda create -n deepfake python=3.10

conda activate deepfake│  (Cloud)    │   │  Local/Cloud │| **WSGI Server** | Gunicorn | 21.2.0 |



# Or using venv└─────────────┘   └──────────────┘| **Deep Learning** | PyTorch | 2.0.1 |

python3.10 -m venv venv

source venv/bin/activate```| **Model** | CLIP ViT-L/14 | - |

```

| **Database** | MongoDB Atlas | 4.6+ |

### 3. Install Dependencies

### Technology Stack| **Object Storage** | MinIO | Latest |

```bash

pip install -r requirements.txt| **API Docs** | Flasgger | 0.9.7.1 |

```

| Component | Technology | Version || **CUDA** | 11.8 / 12.8 | - |

**Key Dependencies:**

```|-----------|-----------|---------|

Flask==3.0.0

gunicorn==21.2.0| **Backend** | Flask | 3.0.0 |## 💻 Requirements

torch==2.0.1

torchvision==0.15.2| **WSGI Server** | Gunicorn | 21.2.0 |

pymongo==4.6.1

minio==7.2.0| **Deep Learning** | PyTorch | 2.0.1 |### Hardware

flasgger==0.9.7.1

opencv-python==4.8.1.78| **Model** | CLIP ViT-L/14 | - |- **GPU**: NVIDIA RTX A4000 (16GB VRAM) or equivalent

Pillow==10.1.0

timm==0.9.12| **Database** | MongoDB Atlas | 4.6+ |  - Minimum: GPU with 8GB+ VRAM

ftfy==6.1.1

python-dotenv==1.0.0| **Object Storage** | MinIO | Latest |  - CPU mode available (slower inference)

```

| **API Docs** | Flasgger | 0.9.7.1 |- **RAM**: 16GB+ recommended

### 4. Download Model Checkpoint

| **CUDA** | 11.8 / 12.8 | - |- **Storage**: 10GB+ for model and dependencies

Ensure the D³-CLIP model is available:



```bash

ls ../checkpoints/finetune_wildrf/model_epoch_best.pth---### Software

```

- **OS**: Linux (Ubuntu 22.04 LTS recommended)

---

## 💻 Requirements- **Python**: 3.8, 3.9, or 3.10

## ⚙️ Configuration

- **CUDA**: 11.8 or 12.8

### 1. Environment Variables

### Hardware- **NVIDIA Drivers**: 470+ (for CUDA support)

Create or update `.env` file:

- **GPU**: NVIDIA RTX A4000 (16GB VRAM) or equivalent

```bash

# Flask Configuration  - Minimum: GPU with 8GB+ VRAM### Accounts (Optional but Recommended)

FLASK_ENV=production

FLASK_PORT=5001  - CPU mode available (slower inference)- **MongoDB Atlas**: Free tier available at [mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas)



# GPU Configuration- **RAM**: 16GB+ recommended- **MinIO**: Can run locally (no account needed)

CUDA_VISIBLE_DEVICES=0  # Use GPU 0 (change to 1 for GPU 1, or 0,1 for both)

- **Storage**: 10GB+ for model and dependencies

# MongoDB Atlas Configuration

MONGODB_URI=mongodb+srv://username:password@cluster0.ahuga.mongodb.net/## 📦 Installation

MONGODB_DATABASE=DeepFake

MONGODB_COLLECTION=predictions### Software



# MinIO Configuration (Docker)- **OS**: Linux (Ubuntu 22.04 LTS recommended)### Option 1: Manual Installation (Recommended for Development)

MINIO_ENDPOINT=localhost:9000

MINIO_ACCESS_KEY=HoangPhatCs- **Python**: 3.8, 3.9, or 3.10

MINIO_SECRET_KEY=your_secure_password

MINIO_BUCKET=deepfake-results- **CUDA**: 11.8 or 12.8#### Step 1: Clone Repository

MINIO_SECURE=False

- **NVIDIA Drivers**: 470+ (for CUDA support)

# Model Configuration

MODEL_PATH=../checkpoints/finetune_wildrf/model_epoch_best.pth```bash

BATCH_SIZE=8

IMAGE_SIZE=224### Servicesgit clone https://github.com/hoangphat25092005/DeepFake-Project-Model.git



# Upload Limits- **MongoDB Atlas**: Free tier available ([Sign up](https://www.mongodb.com/cloud/atlas))cd DeepFake-Project-Model/api

MAX_CONTENT_LENGTH=104857600  # 100MB

```- **MinIO**: Self-hosted object storage```



### 2. MongoDB Atlas Setup



1. **Sign Up**: [mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas)---#### Step 2: Create Python Environment

2. **Create Cluster**: Choose M0 (free tier)

3. **Create Database**: `DeepFake`

4. **Get Connection String**: Replace `<password>` with your password

5. **Whitelist IP**: Add `0.0.0.0/0` or your server IP## 🚀 Quick Start```bash



### 3. MinIO Docker Setup# Using conda (recommended)



MinIO runs in a Docker container for object storage:### 1. Install MinIOconda create -n d3_api python=3.10



```bashconda activate d3_api

# Pull MinIO image

docker pull minio/minio```bash



# Run MinIO containercd /mnt/mmlab2024nas/danh/phatlh/D3/api# Or using venv

docker run -d \

  --name minio \./install_minio.shpython -m venv venv

  -p 9000:9000 \

  -p 9001:9001 \source ~/.bashrcsource venv/bin/activate  # Linux/Mac

  -e "MINIO_ROOT_USER=HoangPhatCs" \

  -e "MINIO_ROOT_PASSWORD=your_secure_password" \``````

  -v ~/minio-data:/data \

  minio/minio server /data --console-address ":9001"

```

### 2. Install Python Dependencies#### Step 3: Install Dependencies

**Access MinIO Console:**

- Console: http://localhost:9001

- API: http://localhost:9000

```bash```bash

**Create Bucket:**

```bashpip install --user gunicorn minio pymongo dnspython flask flask-cors flasgger python-dotenv Pillow opencv-python-headless# Core dependencies

# Using mc (MinIO Client)

docker run -it --entrypoint=/bin/sh minio/mc```pip install flask==3.0.0

mc alias set myminio http://host.docker.internal:9000 HoangPhatCs your_secure_password

mc mb myminio/deepfake-resultspip install flask-cors==4.0.0

```

### 3. Start Servicespip install flasgger==0.9.7.1

---

pip install gunicorn==21.2.0

## 📡 API Endpoints

```bashpip install python-dotenv==1.0.0

### Health Check

./start_manual.sh

**GET** `/health`

# Choose GPU when prompted (0 or 1)# Database

Check API status and GPU availability.

```pip install pymongo==4.6.1

**Response:**

```jsonpip install dnspython==2.6.1

{

  "status": "healthy",### 4. Test API

  "model_loaded": true,

  "gpu_available": true,# Storage

  "gpu_count": 2,

  "cuda_version": "12.8"```bashpip install minio==7.2.0

}

```curl http://localhost:5001/health



**Example:**```# Image processing

```bash

curl http://localhost:5001/healthpip install Pillow==10.1.0

```

**Access Points:**pip install opencv-python==4.8.1.78

---

- API: http://localhost:5001

### Single Image Prediction

- API Docs: http://localhost:5001/apidocs# Deep Learning (CUDA 11.8)

**POST** `/predict`

- MinIO Console: http://localhost:9001pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

Predict if a single image is fake or real.



**Parameters:**

- `image` (file): Image file (JPEG, PNG)---# Additional ML libraries



**Response:**pip install timm==0.9.12

```json

{## 📦 Installationpip install ftfy==6.1.3

  "prediction": "FAKE",

  "confidence": 0.9342,pip install regex==2023.10.3

  "probabilities": {

    "FAKE": 0.9342,### Detailed Installation Steps```

    "REAL": 0.0658

  },

  "model": "D³-CLIP ViT-L/14",

  "timestamp": "2026-01-15T10:30:45Z",#### Step 1: Clone Repository#### Step 4: Install MinIO

  "result_image_url": "http://localhost:9000/deepfake-results/pred_123.jpg",

  "prediction_id": "65a7b8c9d1e2f3a4b5c6d7e8"

}

``````bash```bash



**Example:**git clone https://github.com/hoangphat25092005/DeepFake-Project-Model.git# Download MinIO to user bin directory

```bash

curl -X POST -F "image=@test.jpg" http://localhost:5001/predictcd DeepFake-Project-Model/apimkdir -p ~/bin

```

```wget https://dl.min.io/server/minio/release/linux-amd64/minio -O ~/bin/minio

---

chmod +x ~/bin/minio

### Batch Prediction

#### Step 2: Create Python Environment

**POST** `/predict/batch`

# Add to PATH

Predict multiple images at once.

```bashecho 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc

**Parameters:**

- `images` (files): Multiple image files# Using conda (recommended)source ~/.bashrc



**Response:**conda create -n d3_api python=3.10

```json

{conda activate d3_api# Verify installation

  "results": [

    {minio --version

      "filename": "image1.jpg",

      "prediction": "FAKE",# Or using venvMINIO_ACCESS_KEY=minioadmin

      "confidence": 0.9234,

      "prediction_id": "65a7b8c9d1e2f3a4b5c6d7e8"python -m venv venvMINIO_SECRET_KEY=minioadmin

    },

    {source venv/bin/activateMINIO_BUCKET=your_bucket_name

      "filename": "image2.jpg",

      "prediction": "REAL",```

      "confidence": 0.8765,

      "prediction_id": "65a7b8c9d1e2f3a4b5c6d7e9"# Model

    }

  ],#### Step 3: Install DependenciesMODEL_CHECKPOINT_PATH=your_model_checkpoint

  "total": 2,

  "processing_time": 1.234DEVICE=cpu  # or cuda if your system support

}

``````bash



**Example:**# Core dependencies# Flask

```bash

curl -X POST \pip install flask==3.0.0 flask-cors==4.0.0 flasgger==0.9.7.1FLASK_HOST=0.0.0.0

  -F "images=@image1.jpg" \

  -F "images=@image2.jpg" \pip install gunicorn==21.2.0 python-dotenv==1.0.0FLASK_PORT=5000

  http://localhost:5001/predict/batch

```FLASK_DEBUG=True



---# Database```



### Get Recent Predictionspip install pymongo==4.6.1 dnspython==2.6.1



**GET** `/api/db/predictions/recent?limit=10`### 4. Start MinIO



Retrieve recent predictions from database.# Storage



**Response:**pip install minio==7.2.0```bash

```json

{docker run -d \

  "predictions": [

    {# Image processing  -p 9000:9000 \

      "id": "65a7b8c9d1e2f3a4b5c6d7e8",

      "prediction": "FAKE",pip install Pillow==10.1.0 opencv-python==4.8.1.78  -p 9001:9001 \

      "confidence": 0.9342,

      "timestamp": "2026-01-15T10:30:45Z",  --name minio \

      "result_image_url": "http://localhost:9000/deepfake-results/pred_123.jpg"

    }# Deep Learning (CUDA 11.8)  -e "MINIO_ROOT_USER=minioadmin" \

  ],

  "count": 10pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \  -e "MINIO_ROOT_PASSWORD=minioadmin" \

}

```  --index-url https://download.pytorch.org/whl/cu118  -v ~/minio/data:/data \



**Example:**  quay.io/minio/minio server /data --console-address ":9001"

```bash

curl http://localhost:5001/api/db/predictions/recent?limit=10# Additional ML libraries```

```

pip install timm==0.9.12 ftfy==6.1.3 regex==2023.10.3

---

```### 5. Initialize Database

### Get Prediction by ID



**GET** `/api/db/predictions/{id}`

#### Step 4: Install MinIO (No Sudo Required)```bash

Retrieve a specific prediction by ID.

cd api

**Example:**

```bash```bashbash init_database.sh

curl http://localhost:5001/api/db/predictions/65a7b8c9d1e2f3a4b5c6d7e8

```# Use installation script```



---./install_minio.sh



### Get Predictions by Label### 6. Run API



**GET** `/api/db/predictions/by-label/{label}`# Or manually



Get all predictions for a specific label (FAKE or REAL).mkdir -p ~/bin```bash



**Example:**wget https://dl.min.io/server/minio/release/linux-amd64/minio -O ~/bin/miniocd /path/to/D3

```bash

curl http://localhost:5001/api/db/predictions/by-label/FAKEchmod +x ~/bin/miniopython api/app.py

```

echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc```

---

source ~/.bashrc

### Get Statistics

```API will be available at:

**GET** `/api/db/statistics`

- **API**: http://localhost:5000

Get API usage statistics.

---- **Swagger Docs**: http://localhost:5000/docs

**Response:**

```json- **MinIO Console**: http://localhost:9001

{

  "total_predictions": 1523,## ⚙️ Configuration

  "fake_count": 892,

  "real_count": 631,## API Documentation

  "average_confidence": 0.8745,

  "predictions_today": 45### Environment Variables

}

```### Health Check



**Example:**The `.env` file contains all configuration. Key settings:

```bash

curl http://localhost:5001/api/db/statistics```bash

```

```envGET /health

---

# Flask Configuration```

## 💡 Usage Examples

FLASK_PORT=5001

### Python Client

FLASK_ENV=production### Single Prediction

```python

import requestsFLASK_HOST=0.0.0.0



# Single prediction```bash

with open('test.jpg', 'rb') as f:

    files = {'image': f}# MongoDB AtlasPOST /predict

    response = requests.post('http://localhost:5001/predict', files=files)

    result = response.json()MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/Content-Type: multipart/form-data

    print(f"Prediction: {result['prediction']}")

    print(f"Confidence: {result['confidence']:.2%}")MONGODB_DATABASE=DeepFake



# Batch predictionDATABASE_POOL_SIZE=50{

files = [

    ('images', open('image1.jpg', 'rb')),  "image": <file>

    ('images', open('image2.jpg', 'rb'))

]# MinIO}

response = requests.post('http://localhost:5001/predict/batch', files=files)

results = response.json()MINIO_ENDPOINT=localhost:9000```

for r in results['results']:

    print(f"{r['filename']}: {r['prediction']} ({r['confidence']:.2%})")MINIO_ACCESS_KEY=yourkey

```MINIO_SECRET_KEY=yoursecret#



### JavaScript ClientMINIO_SECURE=False



```javascriptMINIO_BUCKET_NAME=deepfake-results```bash

// Single prediction

const formData = new FormData();POST /batch_predict

formData.append('image', imageFile);

# ModelContent-Type: multipart/form-data

fetch('http://localhost:5001/predict', {

  method: 'POST',MODEL_CHECKPOINT_PATH=../checkpoints/finetune_wildrf/model_epoch_best.pth

  body: formData

})MODEL_TYPE=d3_clip{

  .then(res => res.json())

  .then(data => {DEVICE=cuda  "images": <file[]>

    console.log(`Prediction: ${data.prediction}`);

    console.log(`Confidence: ${(data.confidence * 100).toFixed(2)}%`);}

  });



// Batch prediction### Database Queries

const formData = new FormData();

imageFiles.forEach(file => formData.append('images', file));# Upload



fetch('http://localhost:5001/predict/batch', {MAX_CONTENT_LENGTH=104857600  # 100MB```bash

  method: 'POST',

  body: formData```GET /api/db/predictions/recent?limit=10

})

  .then(res => res.json())GET /api/db/predictions/{id}

  .then(data => {

    data.results.forEach(r => {### MongoDB Atlas SetupGET /api/db/predictions/by-label/FAKE

      console.log(`${r.filename}: ${r.prediction} (${(r.confidence * 100).toFixed(2)}%)`);

    });GET /api/db/statistics

  });

```1. **Sign Up**: [mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas)```



### cURL Examples2. **Create Cluster**: Choose M0 (free tier)



```bash3. **Create Database**: `DeepFake`## Testing

# Health check

curl http://localhost:5001/health4. **Get Connection String**: Replace `<password>` with your password



# Single prediction5. **Whitelist IP**: Add `0.0.0.0/0` or your server IP```bash

curl -X POST -F "image=@test.jpg" http://localhost:5001/predict

# Health check

# Batch prediction

curl -X POST \### MinIO Configurationcurl http://localhost:5000/health

  -F "images=@image1.jpg" \

  -F "images=@image2.jpg" \

  http://localhost:5001/predict/batch

MinIO stores result images:# Single prediction

# Get recent predictions

curl http://localhost:5001/api/db/predictions/recent?limit=10curl -X POST -F "image=@test.jpg" http://localhost:5000/predict



# Get statistics```bash

curl http://localhost:5001/api/db/statistics

```# Start MinIO# Get recent predictions



---MINIO_ROOT_USER=username MINIO_ROOT_PASSWORD=password \curl http://localhost:5000/api/db/predictions/recent



## 📊 Monitoring  nohup ~/bin/minio server ~/minio-data \



### View Logs  --address ":9000" --console-address ":9001" > ~/minio.log 2>&1 &# Get statistics



```bash```curl http://localhost:5000/api/db/statistics

# Application logs

tail -f logs/app.log```



# Error logs---

tail -f logs/error.log

## Project Structure

# MinIO logs

docker logs minio## Deployment

```

```

### Monitor GPU Usage

### Automated Deployment (Recommended)D3/

```bash

# Watch GPU in real-time├── api/

watch -n 1 nvidia-smi

```bash│   ├── app.py                  # Main application

# Check CUDA availability

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"cd /mnt/mmlab2024nas/danh/phatlh/D3/api│   ├── config.py               # Configuration

```

│   ├── .env                    # Environment variables

### Check Service Status

# Start services│   │

```bash

# Check API./start_manual.sh│   ├── routes/                 # API routes

curl http://localhost:5001/health

│   │   ├── health_check_route.py

# Check MongoDB

python check_mongodb_data.py# Stop services│   │   ├── prediction_route.py



# Check MinIO./stop_manual.sh│   │   └── database_route.py

docker ps | grep minio

curl http://localhost:9000/minio/health/live```│   │

```

│   ├── services/               # Business logic

---

### Manual Deployment│   │   ├── minio_service.py

## 🔧 Troubleshooting

│   │   └── database_service.py

### 1. CUDA Out of Memory

#### Start MinIO│   │

**Issue**: `RuntimeError: CUDA out of memory`

│   ├── models/                 # Database models

**Solution**:

```bash```bash│   │   └── prediction.py

# Reduce batch size in .env

BATCH_SIZE=4 MINIO_ROOT_USER=username MINIO_ROOT_PASSWORD=password \│   │



# Use a specific GPU  nohup ~/bin/minio server ~/minio-data \│   └── utils/                  # Utilities

CUDA_VISIBLE_DEVICES=0

  --address ":9000" --console-address ":9001" > ~/minio.log 2>&1 &│       └── model_loader.py

# Clear GPU cache

python -c "import torch; torch.cuda.empty_cache()"```│

```

├── models/                     # ML model architecture

### 2. MongoDB Connection Failed

#### Start API├── checkpoints/                # Model weights (not in git)

**Issue**: Cannot connect to MongoDB Atlas

└── requirements.txt

**Solution**:

- Check internet connection```bash```

- Verify MongoDB URI in `.env`

- Whitelist your IP in MongoDB Atlascd /mnt/mmlab2024nas/danh/phatlh/D3/api

- Check database and collection names

conda activate diffuseinst## 🛠️ Development

### 3. MinIO Container Not Running



**Issue**: MinIO container stopped or not accessible

# Production (with Gunicorn)### Install Development Dependencies

**Solution**:

```bashCUDA_VISIBLE_DEVICES=0 gunicorn --workers 4 --threads 2 --timeout 600 \

# Check container status

docker ps -a | grep minio  --bind 0.0.0.0:5001 app:app```bash



# Restart containerpip install -r requirements-dev.txt

docker restart minio

# Development (with Flask)```

# View logs

docker logs minioCUDA_VISIBLE_DEVICES=0 python app.py



# Recreate container```### Run Tests

docker rm minio

docker run -d --name minio -p 9000:9000 -p 9001:9001 \

  -e "MINIO_ROOT_USER=your_username" \

  -e "MINIO_ROOT_PASSWORD=your_password" \#### Background Deployment```bash

  -v ~/minio-data:/data \

  minio/minio server /data --console-address ":9001"pytest tests/

```

```bash```

### 4. Model Loading Failed

# Start in background

**Issue**: Cannot load model checkpoint

CUDA_VISIBLE_DEVICES=0 nohup gunicorn --workers 4 --threads 2 --timeout 600 \### Code Formatting

**Solution**:

```bash  --bind 0.0.0.0:5001 app:app > api.log 2>&1 &

# Verify model file exists

ls -lh ../checkpoints/finetune_wildrf/model_epoch_best.pthecho $! > api.pid```bash



# Check file permissionsblack api/

chmod 644 ../checkpoints/finetune_wildrf/model_epoch_best.pth



```

```

### 5. Port Already in Use

## Database Schema

**Issue**: Port 5001 or 9000 already in use


# Find process using port

lsof -i :5001```bash- `id`: Primary key

lsof -i :9000


FLASK_PORT=5002




### 6. Image Upload FailedCUDA_VISIBLE_DEVICES=1 ./start_manual.sh- `confidence`: Prediction confidence



**Issue**: File too large or format not supported```- `created_at`: Timestamp



**Solution**:

- Ensure image is under 100MB

- Supported formats: JPEG, JPG, PNG---### prediction_batches table

- Check MAX_CONTENT_LENGTH in `.env`

- `id`: Primary key

---

## 📡 API Endpoints- `batch_id`: Unique batch identifier

## 📁 Project Structure

- `total_images`: Number of images

```

D3/### Base URL- `successful`: Successful predictions

├── api/

│   ├── app.py                  # Main Flask application```- `failed`: Failed predictions

│   ├── config.py               # Configuration management

│   ├── .env                    # Environment variableshttp://localhost:5001

│   │

│   ├── routes/                 # API route handlers```## Configuration

│   │   ├── health_check_route.py

│   │   ├── prediction_route.py

│   │   └── database_route.py

│   │### Swagger DocumentationSee [`.env.example`](api/.env.example) for all configuration options.

│   ├── services/               # Business logic layer

│   │   ├── model_service.py    # Model inference```

│   │   ├── minio_service.py    # Object storage

│   │   └── db_service_mongodb.py  # Database operationshttp://localhost:5001/apidocs## License

│   │

│   ├── models/                 # Database models```

│   │   └── database.py

│   │MIT License

│   └── utils/                  # Utility functions

│       └── model_loader.py### Endpoints Reference

│

├── models/                     # ML model architectures## Contributors

│   ├── clip_models.py

│   ├── vision_transformer.py| Endpoint | Method | Description |

│   └── ...

│|----------|--------|-------------|- Hoang Phat (@hoangphat25092005)

├── checkpoints/                # Model weights (not in git)

│   └── finetune_wildrf/| `/health` | GET | Health check |

│       └── model_epoch_best.pth

│| `/api/v1/status` | GET | API status |## Contributing

└── data/                       # Dataset utilities

    └── datasets.py| `/api/v1/predict` | POST | Single image prediction |

```

| `/api/v1/predict/batch` | POST | Batch prediction |1. Fork the repository

---

| `/api/v1/predictions/recent` | GET | Recent predictions |2. Create feature branch (`git checkout -b feature/amazing-feature`)

## Contributing

| `/api/v1/predictions/<id>` | GET | Get prediction by ID |3. Commit changes (`git commit -m 'Add amazing feature'`)

Contributions are welcome! Please follow these guidelines:

| `/api/v1/statistics` | GET | Usage statistics |4. Push to branch (`git push origin feature/amazing-feature`)

1. **Fork the repository**

2. **Create a feature branch**: `git checkout -b feature/new-feature`5. Open Pull Request

3. **Commit changes**: `git commit -am 'Add new feature'`

4. **Push to branch**: `git push origin feature/new-feature`### API Examples

5. **Submit a pull request**

## Contact

### Code Style

- Follow PEP 8 guidelines#### Health Check

- Add docstrings to functions

- Include type hints where appropriate- GitHub: [@hoangphat25092005](https://github.com/hoangphat25092005)

- Write unit tests for new features

```bash- Project: [DeepFake-Project-Model](https://github.com/hoangphat25092005/DeepFake-Project-Model)

---

curl http://localhost:5001/health

## 📄 License

```## Acknowledgments

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



---

**Response:**- D³ (Diverse Deepfake Detection) research

## Support

```json- OpenAI CLIP architecture

For issues, questions, or contributions:

{- Flask framework

- **GitHub Issues**: [Create an issue](https://github.com/hoangphat25092005/DeepFake-Project-Model/issues)

- **Email**: hoangphatpy123@gmail.com  "status": "ok",- MinIO object storage

- **Documentation**: See additional guides in `/api/docs/`  "timestamp": "2026-01-15T10:30:45.123Z"

}

---```



## Acknowledgments
#### Single Image Prediction


- **D³ Model**: Based on the D³ framework for deepfake detection```bash

- **CLIP**: OpenAI's Contrastive Language-Image Pre-trainingcurl -X POST http://localhost:5001/api/v1/predict \

- **WildRF Dataset**: Fine-tuning dataset for robust detection  -F "file=@image.jpg" \

- **MongoDB Atlas**: Cloud database solution  -F "model_type=d3_clip"

- **MinIO**: High-performance object storage```



---**Response:**

```json

**Built for deepfake detection research**{

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
