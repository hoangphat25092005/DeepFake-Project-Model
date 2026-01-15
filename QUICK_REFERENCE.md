# D³ API - Quick Reference Card

## 🚀 Quick Start

```bash
# Automated deployment (recommended)
./deploy.sh

# Or manual start
sudo systemctl start d3-api
```

## 📊 Common Commands

### Service Management
```bash
# Start service
sudo systemctl start d3-api

# Stop service
sudo systemctl stop d3-api

# Restart service
sudo systemctl restart d3-api

# Check status
sudo systemctl status d3-api

# Enable auto-start on boot
sudo systemctl enable d3-api
```

### View Logs
```bash
# Application logs (live)
sudo journalctl -u d3-api -f

# Last 100 lines
sudo journalctl -u d3-api -n 100

# Today's logs
sudo journalctl -u d3-api --since today

# Nginx logs
sudo tail -f /var/log/nginx/d3-api-error.log
sudo tail -f /var/log/nginx/d3-api-access.log
```

### GPU Monitoring
```bash
# Current GPU status
nvidia-smi

# Watch GPU (updates every 1s)
watch -n 1 nvidia-smi

# GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
```

### Test API
```bash
# Health check
curl http://localhost/health

# Test with image
curl -X POST http://localhost/predict \
  -F "image=@test.jpg"

# Test with video
curl -X POST http://localhost/predict/video \
  -F "video=@test.mp4"

# Get all predictions
curl http://localhost/predictions
```

## 🔧 Configuration Files

| File | Purpose | Location |
|------|---------|----------|
| `.env` | Environment variables | `api/.env` |
| `gunicorn_config.py` | Gunicorn settings | Root directory |
| `d3-api.service` | Systemd service | `/etc/systemd/system/` |
| `d3-api-nginx.conf` | Nginx config | `/etc/nginx/sites-available/` |

## 📍 Important Paths

```bash
# Project directory
/mnt/mmlab2024nas/danh/phatlh/D3

# Model checkpoint
/mnt/mmlab2024nas/danh/phatlh/D3/checkpoints/finetune_wildrf/model_epoch_best.pth

# Logs
/mnt/mmlab2024nas/danh/phatlh/D3/logs/

# API code
/mnt/mmlab2024nas/danh/phatlh/D3/api/
```

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Predict single image |
| `/predict/batch` | POST | Predict multiple images |
| `/predict/video` | POST | Predict video |
| `/predictions` | GET | Get all predictions |
| `/predictions/<id>` | GET | Get specific prediction |

## 🐛 Quick Troubleshooting

### Service won't start
```bash
sudo journalctl -u d3-api -n 50
sudo systemctl status d3-api
```

### 502 Bad Gateway
```bash
curl http://localhost:6000/health  # Test Gunicorn directly
sudo systemctl restart d3-api
sudo systemctl restart nginx
```

### GPU Out of Memory
```bash
nvidia-smi  # Check GPU usage
# Reduce workers in gunicorn_config.py
```

### Slow performance
```bash
# Check if GPU is being used
nvidia-smi
# Should show GPU processes when making predictions
```

## ⚙️ Configuration Adjustments

### Increase timeout for large videos
Edit `gunicorn_config.py`:
```python
timeout = 1200  # 20 minutes
```

### Reduce memory usage
Edit `gunicorn_config.py`:
```python
workers = 1  # Use only 1 worker
```

### Change port
Edit `gunicorn_config.py`:
```python
bind = "0.0.0.0:7000"  # Use port 7000
```

Then update Nginx upstream in `/etc/nginx/sites-available/d3-api`:
```nginx
upstream d3_backend {
    server 127.0.0.1:7000;
}
```

## 🔄 Update & Restart Workflow

```bash
# 1. Update code
cd /mnt/mmlab2024nas/danh/phatlh/D3
git pull  # or edit files

# 2. Update dependencies (if needed)
conda activate d3
pip install -r requirements.txt

# 3. Restart service
sudo systemctl restart d3-api

# 4. Check status
sudo systemctl status d3-api

# 5. Test
curl http://localhost/health
```

## 📱 Remote Access

Replace `localhost` with your server IP:
```bash
# From another machine
curl http://192.168.1.100/health

# With authentication (if you add it later)
curl -H "Authorization: Bearer YOUR_TOKEN" http://192.168.1.100/predict
```

## 🔒 Security Tips

1. **Add authentication** to API endpoints
2. **Set up SSL** with Let's Encrypt
3. **Use firewall** rules to limit access
4. **Rotate logs** regularly
5. **Monitor** for unusual activity

## 📊 Performance Metrics

```bash
# Check response time
time curl http://localhost/health

# Load test
ab -n 100 -c 10 http://localhost/health

# Check Nginx connections
sudo netstat -an | grep :80 | wc -l
```

## 🎯 Production Checklist

- [ ] Service running and enabled
- [ ] Nginx configured and running
- [ ] Firewall ports opened
- [ ] SSL certificate installed
- [ ] Logs rotating
- [ ] GPU accessible
- [ ] MinIO service running
- [ ] Database accessible
- [ ] Health check passing
- [ ] Test predictions working

---

**Need more help?** Check the full guide: `DEPLOYMENT.md`
