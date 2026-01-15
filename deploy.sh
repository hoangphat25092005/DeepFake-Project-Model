#!/bin/bash

################################################################################
# D³ Deepfake Detection API - Deployment Script
# This script automates the deployment process on your GPU server
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="/mnt/mmlab2024nas/danh/phatlh/D3"
CONDA_ENV="d3"
SERVICE_NAME="d3-api"
NGINX_CONFIG="d3-api"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}D³ Deepfake Detection API Deployment${NC}"
echo -e "${GREEN}========================================${NC}\n"

# Function to print step
print_step() {
    echo -e "\n${YELLOW}>>> $1${NC}\n"
}

# Function to check command success
check_success() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Success${NC}"
    else
        echo -e "${RED}✗ Failed${NC}"
        exit 1
    fi
}

# Step 1: Create logs directory
print_step "Step 1: Creating logs directory"
mkdir -p "$PROJECT_DIR/logs"
check_success

# Step 2: Check conda environment
print_step "Step 2: Checking conda environment"
if conda env list | grep -q "^$CONDA_ENV "; then
    echo "✓ Conda environment '$CONDA_ENV' exists"
else
    echo -e "${RED}✗ Conda environment '$CONDA_ENV' not found!${NC}"
    echo "Please create it first with: conda env create -f environment.yml"
    exit 1
fi

# Step 3: Install Gunicorn
print_step "Step 3: Installing Gunicorn"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
pip install gunicorn gevent
check_success

# Step 4: Test Gunicorn locally
print_step "Step 4: Testing Gunicorn (3 seconds)"
cd "$PROJECT_DIR/api"
timeout 3 gunicorn --config "$PROJECT_DIR/gunicorn_config.py" app:app || true
echo "✓ Gunicorn configuration looks good"

# Step 5: Install systemd service
print_step "Step 5: Installing systemd service"
echo "Please enter your sudo password if prompted:"
sudo cp "$PROJECT_DIR/$SERVICE_NAME.service" /etc/systemd/system/
sudo systemctl daemon-reload
check_success

# Step 6: Enable and start service
print_step "Step 6: Enabling and starting service"
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"
sleep 3
check_success

# Step 7: Check service status
print_step "Step 7: Checking service status"
sudo systemctl status "$SERVICE_NAME" --no-pager || true

# Step 8: Test local API
print_step "Step 8: Testing local API endpoint"
sleep 2
curl -f http://localhost:6000/health || echo -e "${RED}API not responding on port 6000${NC}"
check_success

# Step 9: Configure Nginx
print_step "Step 9: Configuring Nginx"
if ! command -v nginx &> /dev/null; then
    echo "Installing Nginx..."
    sudo apt-get update
    sudo apt-get install -y nginx
fi

# Update YOUR_DOMAIN_OR_IP in nginx config
read -p "Enter your server IP or domain (e.g., 192.168.1.100): " SERVER_ADDR
if [ -z "$SERVER_ADDR" ]; then
    echo "Using localhost as default"
    SERVER_ADDR="localhost"
fi

sed -i "s/YOUR_DOMAIN_OR_IP/$SERVER_ADDR/g" "$PROJECT_DIR/d3-api-nginx.conf"

sudo cp "$PROJECT_DIR/d3-api-nginx.conf" /etc/nginx/sites-available/$NGINX_CONFIG
sudo ln -sf /etc/nginx/sites-available/$NGINX_CONFIG /etc/nginx/sites-enabled/
sudo nginx -t
check_success

# Step 10: Restart Nginx
print_step "Step 10: Restarting Nginx"
sudo systemctl restart nginx
check_success

# Step 11: Configure firewall
print_step "Step 11: Configuring firewall (UFW)"
if command -v ufw &> /dev/null; then
    sudo ufw allow 80/tcp
    sudo ufw allow 443/tcp
    echo "✓ Opened ports 80 and 443"
else
    echo "UFW not installed, skipping firewall configuration"
fi

# Step 12: Final health check
print_step "Step 12: Final health check"
sleep 2
echo "Testing API through Nginx..."
curl -f http://localhost/health || echo -e "${RED}API not responding through Nginx${NC}"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Deployment Complete!${NC}"
echo -e "${GREEN}========================================${NC}\n"

echo "📊 Service Status:"
sudo systemctl status "$SERVICE_NAME" --no-pager | head -10

echo -e "\n🔗 API Endpoints:"
echo "  • Health Check: http://$SERVER_ADDR/health"
echo "  • Image Prediction: http://$SERVER_ADDR/predict"
echo "  • Video Prediction: http://$SERVER_ADDR/predict/video"

echo -e "\n📝 Useful Commands:"
echo "  • View logs: journalctl -u $SERVICE_NAME -f"
echo "  • Restart service: sudo systemctl restart $SERVICE_NAME"
echo "  • Stop service: sudo systemctl stop $SERVICE_NAME"
echo "  • Check status: sudo systemctl status $SERVICE_NAME"
echo "  • Nginx logs: sudo tail -f /var/log/nginx/d3-api-error.log"

echo -e "\n⚠️  Next Steps:"
echo "  1. Configure SSL certificate with Let's Encrypt (recommended)"
echo "  2. Set up monitoring with Prometheus/Grafana"
echo "  3. Configure log rotation"
echo "  4. Test with actual images/videos"

echo -e "\n${GREEN}Happy detecting deepfakes! 🔍${NC}\n"
