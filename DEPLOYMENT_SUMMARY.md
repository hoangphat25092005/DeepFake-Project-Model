# 🚀 D³ Deepfake Detection API - Deployment Package

## 📦 What's Included

Your D³ API is now ready for production deployment with the following files:

### Configuration Files
1. **`api/.env`** - Production environment configuration
   - Flask production mode enabled
   - GPU and model paths configured
   - MinIO and PostgreSQL credentials

2. **`gunicorn_config.py`** - WSGI server configuration
   - 2 workers optimized for GPU
   - 600s timeout for video processing
   - Comprehensive logging setup

3. **`d3-api.service`** - Systemd service definition
   - Auto-start on boot
   - Automatic restart on failure
   - GPU environment variables

4. **`d3-api-nginx.conf`** - Nginx reverse proxy config
   - Load balancing
   - SSL/HTTPS ready
   - Large file upload support (500MB)

### Deployment Tools
1. **`check_deployment.sh`** - Pre-deployment verification
   - Checks all requirements
   - Validates GPU, Python, dependencies
   - Tests model loading
   - **Run this first!**

2. **`deploy.sh`** - Automated deployment script
   - One-command deployment
   - Sets up all services
   - Configures firewall
   - Tests everything

### Documentation
1. **`DEPLOYMENT.md`** - Complete deployment guide
   - Step-by-step manual deployment
   - GPU optimization tips
   - SSL/HTTPS setup
   - Troubleshooting guide

2. **`QUICK_REFERENCE.md`** - Quick command reference
   - Common commands
   - Configuration locations
   - Testing examples

3. **`README.md`** - Project overview and API usage

---

## 🎯 Deployment Steps

### Option 1: Automated (Recommended)

```bash
# 1. Verify everything is ready
./check_deployment.sh

# 2. Deploy automatically
./deploy.sh

# 3. Test the API
curl http://YOUR_SERVER_IP/health
```

### Option 2: Manual

Follow the detailed guide in `DEPLOYMENT.md`

---

## 📋 Pre-Deployment Checklist

Before deploying, ensure you have:

- ✅ NVIDIA GPU with CUDA support
- ✅ Conda environment 'd3' created and working
- ✅ Model checkpoint file exists
- ✅ MinIO service running (for result storage)
- ✅ PostgreSQL database accessible
- ✅ Static IP or domain name for your server
- ✅ Sudo/root access for installing services

---

## 🔧 Quick Configuration

### Change API Port
Edit `gunicorn_config.py`:
```python
bind = "0.0.0.0:YOUR_PORT"
```

### Adjust GPU Memory
Edit `api/services/inference_service.py`:
```python
torch.cuda.set_per_process_memory_fraction(0.6)  # Use 60% of GPU
```

### Limit Workers
Edit `gunicorn_config.py`:
```python
workers = 1  # Use only 1 worker to save memory
```

### Increase Timeout
Edit `gunicorn_config.py`:
```python
timeout = 1200  # 20 minutes for very long videos
```

---

## 🌐 API Endpoints

Once deployed, your API will have these endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Single image prediction |
| `/predict/batch` | POST | Batch image prediction |
| `/predict/video` | POST | Video prediction |
| `/predictions` | GET | Get all predictions |
| `/predictions/<id>` | GET | Get specific prediction |

---

## 🧪 Testing After Deployment

```bash
# 1. Health check
curl http://localhost/health

# 2. Test with sample image
curl -X POST http://localhost/predict \
  -F "image=@data/WildRF/test/facebook/00000.jpg"

# 3. Check service status
sudo systemctl status d3-api

# 4. Watch logs
sudo journalctl -u d3-api -f

# 5. Monitor GPU
watch -n 1 nvidia-smi
```

---

## 📊 Architecture Overview

```
Internet
    ↓
[Port 80/443]
    ↓
Nginx (Reverse Proxy)
    ↓
[Port 6000]
    ↓
Gunicorn (WSGI Server)
    ↓
Flask API (app.py)
    ↓
┌─────────────┬──────────────┬────────────┐
│             │              │            │
D3 Model   MinIO       PostgreSQL    GPU
(PyTorch)  (Storage)   (Database)   (CUDA)
```

---

## 🛡️ Security Recommendations

1. **SSL/HTTPS**: Use Let's Encrypt for free SSL certificate
   ```bash
   sudo certbot --nginx -d your-domain.com
   ```

2. **Firewall**: Only open necessary ports
   ```bash
   sudo ufw allow 80/tcp
   sudo ufw allow 443/tcp
   sudo ufw enable
   ```

3. **Authentication**: Add API key authentication (future enhancement)

4. **Rate Limiting**: Configure Nginx rate limiting
   ```nginx
   limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
   ```

5. **CORS**: Configure allowed origins in Flask
   ```python
   from flask_cors import CORS
   CORS(app, origins=["https://your-frontend.com"])
   ```

---

## 📈 Performance Tips

1. **Batch Processing**: Use `/predict/batch` for multiple images
2. **Video Sampling**: Adjust frame sampling rate in `inference_service.py`
3. **Caching**: Enable Nginx caching for repeated requests
4. **Load Balancing**: Add multiple Gunicorn instances for high traffic
5. **GPU Optimization**: Use mixed precision inference

---

## 🔄 Maintenance

### Daily
- Check service status: `sudo systemctl status d3-api`
- Monitor GPU: `nvidia-smi`
- Check disk space: `df -h`

### Weekly
- Review logs: `sudo journalctl -u d3-api --since "1 week ago"`
- Check database size: `psql -c "SELECT pg_size_pretty(pg_database_size('d3_results'))"`
- Verify backups

### Monthly
- Update dependencies: `pip install --upgrade -r requirements.txt`
- Rotate logs: Automatic if logrotate is configured
- Review performance metrics

---

## 🐛 Common Issues & Solutions

### Issue: Service won't start
```bash
# Check logs
sudo journalctl -u d3-api -n 50

# Verify conda path
which conda

# Test manually
conda activate d3
cd api
gunicorn --config ../gunicorn_config.py app:app
```

### Issue: GPU not detected
```bash
# Check GPU
nvidia-smi

# Verify CUDA in PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# Check CUDA_VISIBLE_DEVICES
echo $CUDA_VISIBLE_DEVICES
```

### Issue: 502 Bad Gateway
```bash
# Test Gunicorn directly
curl http://localhost:6000/health

# Restart services
sudo systemctl restart d3-api
sudo systemctl restart nginx
```

### Issue: Out of Memory
```bash
# Reduce workers in gunicorn_config.py
workers = 1

# Reduce GPU memory fraction
torch.cuda.set_per_process_memory_fraction(0.5)
```

---

## 📞 Support & Resources

- **Full Documentation**: `DEPLOYMENT.md`
- **Quick Commands**: `QUICK_REFERENCE.md`
- **Check System**: `./check_deployment.sh`
- **Deploy**: `./deploy.sh`

---

## 📝 Next Steps After Deployment

1. ✅ Verify all endpoints work
2. ✅ Set up SSL certificate with Let's Encrypt
3. ✅ Configure monitoring (Prometheus/Grafana)
4. ✅ Set up automated backups
5. ✅ Add authentication layer
6. ✅ Create API documentation (Swagger/OpenAPI)
7. ✅ Build frontend dashboard
8. ✅ Set up continuous deployment (CI/CD)

---

## 🎉 You're Ready!

Your D³ Deepfake Detection API is production-ready with:
- ✅ GPU-accelerated inference
- ✅ Production-grade WSGI server (Gunicorn)
- ✅ Reverse proxy (Nginx)
- ✅ Systemd service management
- ✅ Comprehensive logging
- ✅ Video and image support
- ✅ MinIO storage integration
- ✅ PostgreSQL database
- ✅ Error handling and validation

**Start with:**
```bash
./check_deployment.sh && ./deploy.sh
```

**Good luck with your deployment! 🚀**
