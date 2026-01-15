# D³ Deepfake Detection API - Deployment Guide

Complete guide for deploying the D³ API on your school GPU server with production-ready configuration.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Deployment](#quick-deployment)
3. [Manual Deployment](#manual-deployment)
4. [GPU Optimization](#gpu-optimization)
5. [SSL/HTTPS Setup](#ssl-https-setup)
6. [Monitoring & Logging](#monitoring--logging)
7. [Testing](#testing)
8. [Troubleshooting](#troubleshooting)

---

## 🔧 Prerequisites

### System Requirements
- **OS**: Ubuntu 18.04+ or similar Linux distribution
- **GPU**: NVIDIA GPU with CUDA support
- **RAM**: Minimum 16GB (32GB recommended for video processing)
- **Storage**: 50GB+ free space
- **Network**: Static IP or domain name

### Software Requirements
```bash
# Check GPU
nvidia-smi

# Check CUDA
nvcc --version

# Check Python
python --version  # Should be 3.8+

# Check conda
conda --version
```

### Required Services
- **MinIO**: Object storage (should be running on port 9000)
- **PostgreSQL**: Database (should be accessible)

---

## 🚀 Quick Deployment (Automated)

The easiest way to deploy is using the automated deployment script:

### Step 1: Make script executable
```bash
cd /mnt/mmlab2024nas/danh/phatlh/D3
chmod +x deploy.sh
```

### Step 2: Run deployment
```bash
./deploy.sh
```

The script will:
- ✅ Create logs directory
- ✅ Verify conda environment
- ✅ Install Gunicorn
- ✅ Set up systemd service
- ✅ Configure Nginx
- ✅ Set up firewall rules
- ✅ Start the API

### Step 3: Verify deployment
```bash
# Check service status
sudo systemctl status d3-api

# Test API
curl http://localhost/health
```

---

## 🛠️ Manual Deployment

If you prefer step-by-step manual deployment or need to customize:

### Step 1: Prepare Environment

```bash
cd /mnt/mmlab2024nas/danh/phatlh/D3

# Create logs directory
mkdir -p logs

# Activate conda environment
conda activate d3

# Install Gunicorn
pip install gunicorn gevent
```

### Step 2: Configure Gunicorn

The `gunicorn_config.py` is already created. Key settings:
- **Workers**: 2 (limited for GPU memory)
- **Threads**: 2 per worker
- **Timeout**: 600s (for long video processing)
- **Bind**: 0.0.0.0:6000

Test Gunicorn:
```bash
cd api
gunicorn --config ../gunicorn_config.py app:app
# Press Ctrl+C to stop
```

### Step 3: Install Systemd Service

```bash
# Copy service file
sudo cp d3-api.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable service (start on boot)
sudo systemctl enable d3-api

# Start service
sudo systemctl start d3-api

# Check status
sudo systemctl status d3-api
```

**Important**: Edit `/etc/systemd/system/d3-api.service` if your paths differ:
- User/Group
- Conda environment path
- Project directory path

### Step 4: Configure Nginx

```bash
# Install Nginx
sudo apt-get update
sudo apt-get install nginx

# Edit nginx config and replace YOUR_DOMAIN_OR_IP
nano d3-api-nginx.conf
# Replace YOUR_DOMAIN_OR_IP with your server IP (e.g., 192.168.1.100)

# Copy to Nginx sites
sudo cp d3-api-nginx.conf /etc/nginx/sites-available/d3-api

# Create symlink
sudo ln -s /etc/nginx/sites-available/d3-api /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx
```

### Step 5: Configure Firewall

```bash
# Allow HTTP and HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Check status
sudo ufw status
```

### Step 6: Verify Deployment

```bash
# Check Gunicorn service
sudo systemctl status d3-api

# Check Nginx
sudo systemctl status nginx

# Test API locally
curl http://localhost/health

# Test from external (replace with your IP)
curl http://YOUR_SERVER_IP/health
```

---

## ⚡ GPU Optimization

### 1. Limit GPU Memory Growth

Add to `api/services/inference_service.py` (already configured):
```python
import torch
torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% max
```

### 2. Monitor GPU Usage

```bash
# Watch GPU in real-time
watch -n 1 nvidia-smi

# Check GPU memory
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
```

### 3. Multiple GPU Support

If you have multiple GPUs, specify which one to use in `.env`:
```env
CUDA_VISIBLE_DEVICES=0  # Use GPU 0
# Or CUDA_VISIBLE_DEVICES=0,1 for multiple GPUs
```

Also update `d3-api.service`:
```ini
Environment="CUDA_VISIBLE_DEVICES=0"
```

### 4. Batch Processing Optimization

For video processing, adjust frame batch size in `inference_service.py`:
```python
FRAME_BATCH_SIZE = 8  # Adjust based on GPU memory
```

### 5. Mixed Precision (Optional)

For faster inference, enable mixed precision:
```python
# In api/utils/model_loader.py
with torch.cuda.amp.autocast():
    outputs = model(image_tensor)
```

---

## 🔒 SSL/HTTPS Setup (Recommended)

### Option 1: Let's Encrypt (Free SSL)

```bash
# Install Certbot
sudo apt-get install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal is set up automatically
sudo certbot renew --dry-run
```

### Option 2: Self-Signed Certificate (Development)

```bash
# Generate certificate
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout /etc/ssl/private/d3-api.key \
  -out /etc/ssl/certs/d3-api.crt

# Update Nginx config to use the certificate
# Uncomment HTTPS section in d3-api-nginx.conf
```

After SSL setup:
- Uncomment HTTPS server block in `d3-api-nginx.conf`
- Update certificate paths
- Restart Nginx: `sudo systemctl restart nginx`

---

## 📊 Monitoring & Logging

### View Logs

```bash
# Systemd service logs (Gunicorn)
sudo journalctl -u d3-api -f

# Last 100 lines
sudo journalctl -u d3-api -n 100

# Nginx access logs
sudo tail -f /var/log/nginx/d3-api-access.log

# Nginx error logs
sudo tail -f /var/log/nginx/d3-api-error.log

# Application logs
tail -f /mnt/mmlab2024nas/danh/phatlh/D3/logs/error.log
tail -f /mnt/mmlab2024nas/danh/phatlh/D3/logs/access.log
```

### Set Up Log Rotation

Create `/etc/logrotate.d/d3-api`:
```bash
sudo nano /etc/logrotate.d/d3-api
```

Add:
```
/mnt/mmlab2024nas/danh/phatlh/D3/logs/*.log {
    daily
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 danh danh
    sharedscripts
    postrotate
        systemctl reload d3-api
    endscript
}
```

### Monitor GPU

Create systemd service to log GPU usage:
```bash
# Create monitoring script
cat > /home/danh/monitor_gpu.sh << 'EOF'
#!/bin/bash
while true; do
    nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.free --format=csv >> /var/log/gpu_usage.log
    sleep 60
done
EOF

chmod +x /home/danh/monitor_gpu.sh
```

---

## 🧪 Testing

### 1. Health Check

```bash
curl http://localhost/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_available": true
}
```

### 2. Image Prediction

```bash
curl -X POST http://localhost/predict \
  -F "image=@/path/to/test/image.jpg"
```

### 3. Video Prediction

```bash
curl -X POST http://localhost/predict/video \
  -F "video=@/path/to/test/video.mp4" \
  -o response.json
```

### 4. Load Testing

Install Apache Bench:
```bash
sudo apt-get install apache2-utils
```

Run load test:
```bash
ab -n 100 -c 10 http://localhost/health
```

---

## 🐛 Troubleshooting

### Issue 1: Service Won't Start

```bash
# Check logs
sudo journalctl -u d3-api -n 50

# Common causes:
# - Wrong conda path
# - Missing dependencies
# - Port already in use

# Check if port 6000 is free
sudo netstat -tulpn | grep 6000
```

**Fix**: Edit `/etc/systemd/system/d3-api.service` paths and restart:
```bash
sudo systemctl daemon-reload
sudo systemctl restart d3-api
```

### Issue 2: 502 Bad Gateway (Nginx)

```bash
# Check if Gunicorn is running
sudo systemctl status d3-api

# Check Gunicorn logs
sudo journalctl -u d3-api -f

# Test Gunicorn directly
curl http://localhost:6000/health
```

**Fix**: Restart services:
```bash
sudo systemctl restart d3-api
sudo systemctl restart nginx
```

### Issue 3: GPU Out of Memory

```bash
# Check GPU memory
nvidia-smi

# Monitor in real-time
watch -n 1 nvidia-smi
```

**Fix**: Reduce workers or batch size in `gunicorn_config.py`:
```python
workers = 1  # Reduce to 1
```

### Issue 4: Slow Video Processing

**Fix**: Adjust frame sampling in `inference_service.py`:
```python
# Sample fewer frames
fps = max(1, int(total_frames / 30))  # Process 30 frames max
```

### Issue 5: MinIO Connection Failed

```bash
# Check MinIO service
curl http://localhost:9000

# Check MinIO credentials in .env
cat api/.env | grep MINIO
```

**Fix**: Start MinIO or update credentials in `.env`

### Issue 6: Database Connection Error

```bash
# Check PostgreSQL
sudo systemctl status postgresql

# Test connection
psql -U your_user -d d3_results -c "SELECT 1"
```

**Fix**: Update `DATABASE_URL` in `.env`

---

## 📱 API Usage Examples

### Python Client

```python
import requests

# Health check
response = requests.get("http://YOUR_SERVER_IP/health")
print(response.json())

# Image prediction
with open("image.jpg", "rb") as f:
    files = {"image": f}
    response = requests.post("http://YOUR_SERVER_IP/predict", files=files)
    print(response.json())

# Video prediction
with open("video.mp4", "rb") as f:
    files = {"video": f}
    response = requests.post("http://YOUR_SERVER_IP/predict/video", files=files)
    result = response.json()
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Frames analyzed: {len(result['frame_predictions'])}")
```

### cURL Examples

```bash
# Image prediction
curl -X POST http://YOUR_SERVER_IP/predict \
  -F "image=@test.jpg" \
  | jq '.'

# Video prediction
curl -X POST http://YOUR_SERVER_IP/predict/video \
  -F "video=@test.mp4" \
  | jq '.frame_predictions | length'

# Get all predictions from database
curl http://YOUR_SERVER_IP/predictions | jq '.'
```

---

## 🎯 Performance Tuning

### 1. Optimize Nginx

Add to `/etc/nginx/nginx.conf`:
```nginx
worker_processes auto;
worker_rlimit_nofile 65535;

events {
    worker_connections 4096;
    use epoll;
}
```

### 2. Enable HTTP/2

Update your Nginx server block:
```nginx
listen 443 ssl http2;
```

### 3. Enable Gzip Compression

```nginx
gzip on;
gzip_vary on;
gzip_types application/json;
```

### 4. Add Caching

```nginx
location ~* \.(jpg|jpeg|png|gif|ico)$ {
    expires 30d;
    add_header Cache-Control "public, immutable";
}
```

---

## 🔄 Maintenance

### Update Code

```bash
cd /mnt/mmlab2024nas/danh/phatlh/D3
git pull  # If using git

# Restart service
sudo systemctl restart d3-api
```

### Update Dependencies

```bash
conda activate d3
pip install --upgrade -r requirements.txt
sudo systemctl restart d3-api
```

### Backup Database

```bash
pg_dump -U your_user d3_results > backup_$(date +%Y%m%d).sql
```

---

## 📞 Support

If you encounter issues:

1. Check logs: `sudo journalctl -u d3-api -n 100`
2. Verify GPU: `nvidia-smi`
3. Test services: `sudo systemctl status d3-api nginx`
4. Check ports: `sudo netstat -tulpn | grep -E '80|6000'`

---

## ✅ Deployment Checklist

- [ ] Conda environment activated
- [ ] Gunicorn installed
- [ ] Systemd service configured and running
- [ ] Nginx installed and configured
- [ ] Firewall ports opened (80, 443)
- [ ] SSL certificate obtained (optional but recommended)
- [ ] MinIO service running
- [ ] PostgreSQL database accessible
- [ ] GPU accessible (`nvidia-smi` works)
- [ ] Health check responds successfully
- [ ] Test image prediction works
- [ ] Test video prediction works
- [ ] Logs directory created and writable
- [ ] Log rotation configured

---

**🎉 Congratulations! Your D³ Deepfake Detection API is now deployed and production-ready!**
