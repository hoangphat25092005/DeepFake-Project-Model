#!/bin/bash

################################################################################
# D³ Deepfake Detection API - Pre-Deployment Checker
# Run this script before deployment to verify all requirements are met
################################################################################

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
PASSED=0
FAILED=0
WARNINGS=0

echo -e "${BLUE}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   D³ API Pre-Deployment Checklist                 ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════╝${NC}\n"

# Function to check and print result
check() {
    local name="$1"
    local command="$2"
    local required="$3"  # "required" or "optional"
    
    printf "%-50s" "Checking $name..."
    
    if eval "$command" &>/dev/null; then
        echo -e "[${GREEN}✓${NC}]"
        ((PASSED++))
        return 0
    else
        if [ "$required" = "required" ]; then
            echo -e "[${RED}✗${NC}]"
            ((FAILED++))
        else
            echo -e "[${YELLOW}⚠${NC}]"
            ((WARNINGS++))
        fi
        return 1
    fi
}

# Function to check file exists
check_file() {
    local name="$1"
    local filepath="$2"
    local required="$3"
    
    check "$name" "test -f '$filepath'" "$required"
}

# Function to check directory exists
check_dir() {
    local name="$1"
    local dirpath="$2"
    local required="$3"
    
    check "$name" "test -d '$dirpath'" "$required"
}

# Function to get info
get_info() {
    local name="$1"
    local command="$2"
    
    printf "%-50s" "$name:"
    result=$(eval "$command" 2>/dev/null)
    if [ -n "$result" ]; then
        echo -e "${GREEN}$result${NC}"
    else
        echo -e "${YELLOW}N/A${NC}"
    fi
}

echo -e "${BLUE}━━━ System Requirements ━━━${NC}\n"

check "NVIDIA GPU" "nvidia-smi" "required"
check "CUDA" "nvcc --version" "required"
check "Python 3" "python3 --version" "required"
check "Conda" "conda --version" "required"
check "Git" "git --version" "optional"

echo -e "\n${BLUE}━━━ System Information ━━━${NC}\n"

get_info "GPU" "nvidia-smi --query-gpu=name --format=csv,noheader | head -1"
get_info "CUDA Version" "nvcc --version | grep release | awk '{print \$5}' | cut -d',' -f1"
get_info "Python Version" "python3 --version | awk '{print \$2}'"
get_info "Total RAM" "free -h | grep Mem | awk '{print \$2}'"
get_info "Available Disk Space" "df -h /mnt/mmlab2024nas | tail -1 | awk '{print \$4}'"
get_info "CPU Cores" "nproc"

echo -e "\n${BLUE}━━━ Conda Environment ━━━${NC}\n"

check "Conda env 'd3' exists" "conda env list | grep -q '^d3 '" "required"

if conda env list | grep -q '^d3 '; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate d3
    
    check "PyTorch installed" "python -c 'import torch'" "required"
    check "CUDA available in PyTorch" "python -c 'import torch; assert torch.cuda.is_available()'" "required"
    check "timm installed" "python -c 'import timm'" "required"
    check "Flask installed" "python -c 'import flask'" "required"
    check "OpenCV installed" "python -c 'import cv2'" "required"
    
    get_info "PyTorch Version" "python -c 'import torch; print(torch.__version__)' 2>/dev/null"
    get_info "CUDA Device" "python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null"
fi

echo -e "\n${BLUE}━━━ Project Files ━━━${NC}\n"

PROJECT_DIR="/mnt/mmlab2024nas/danh/phatlh/D3"

check_file "API app.py" "$PROJECT_DIR/api/app.py" "required"
check_file "Environment file (.env)" "$PROJECT_DIR/api/.env" "required"
check_file "Model checkpoint" "$PROJECT_DIR/checkpoints/finetune_wildrf/model_epoch_best.pth" "required"
check_file "Gunicorn config" "$PROJECT_DIR/gunicorn_config.py" "required"
check_file "Systemd service" "$PROJECT_DIR/d3-api.service" "required"
check_file "Nginx config" "$PROJECT_DIR/d3-api-nginx.conf" "required"
check_file "Deployment script" "$PROJECT_DIR/deploy.sh" "required"

check_dir "API routes" "$PROJECT_DIR/api/routes" "required"
check_dir "API services" "$PROJECT_DIR/api/services" "required"
check_dir "Models directory" "$PROJECT_DIR/models" "required"

echo -e "\n${BLUE}━━━ External Services ━━━${NC}\n"

check "MinIO reachable" "curl -s -o /dev/null -w '%{http_code}' http://localhost:9000 | grep -q 200" "required"
check "PostgreSQL running" "sudo systemctl is-active postgresql" "required"

# Check MinIO credentials from .env if file exists
if [ -f "$PROJECT_DIR/api/.env" ]; then
    source "$PROJECT_DIR/api/.env"
    if [ -n "$MINIO_ENDPOINT" ]; then
        check "MinIO endpoint accessible" "curl -s -o /dev/null -w '%{http_code}' $MINIO_ENDPOINT | grep -q 200" "required"
    fi
fi

echo -e "\n${BLUE}━━━ Network & Ports ━━━${NC}\n"

check "Port 6000 available" "! sudo netstat -tulpn 2>/dev/null | grep -q ':6000'" "required"
check "Port 80 available" "! sudo netstat -tulpn 2>/dev/null | grep -q ':80'" "optional"
check "Nginx installed" "nginx -v" "optional"

echo -e "\n${BLUE}━━━ Permissions ━━━${NC}\n"

check_dir "Logs directory exists" "$PROJECT_DIR/logs" "optional"
check "Deploy script executable" "test -x '$PROJECT_DIR/deploy.sh'" "optional"

if [ -d "$PROJECT_DIR/logs" ]; then
    check "Logs directory writable" "test -w '$PROJECT_DIR/logs'" "optional"
fi

echo -e "\n${BLUE}━━━ Test Model Loading ━━━${NC}\n"

printf "%-50s" "Testing model load..."
cd "$PROJECT_DIR"
if conda env list | grep -q '^d3 '; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate d3
    
    test_output=$(python -c "
import sys
sys.path.append('$PROJECT_DIR')
try:
    from api.utils.model_loader import D3ModelPredictor
    predictor = D3ModelPredictor()
    print('SUCCESS')
except Exception as e:
    print(f'ERROR: {e}')
    sys.exit(1)
" 2>&1)
    
    if echo "$test_output" | grep -q "SUCCESS"; then
        echo -e "[${GREEN}✓${NC}]"
        ((PASSED++))
    else
        echo -e "[${RED}✗${NC}]"
        echo "  Error: $test_output"
        ((FAILED++))
    fi
else
    echo -e "[${RED}✗${NC}]"
    echo "  Cannot activate conda environment"
    ((FAILED++))
fi

# Summary
echo -e "\n${BLUE}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Results Summary                                  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════╝${NC}\n"

TOTAL=$((PASSED + FAILED + WARNINGS))
echo -e "Total checks: ${BLUE}$TOTAL${NC}"
echo -e "Passed:       ${GREEN}$PASSED${NC}"
echo -e "Failed:       ${RED}$FAILED${NC}"
echo -e "Warnings:     ${YELLOW}$WARNINGS${NC}"

echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}╔════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║   ✓ All critical checks passed!                   ║${NC}"
    echo -e "${GREEN}║   Ready for deployment!                           ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "Next step: Run ${GREEN}./deploy.sh${NC} to deploy the API"
    echo ""
    exit 0
else
    echo -e "${RED}╔════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║   ✗ Some critical checks failed                   ║${NC}"
    echo -e "${RED}║   Please fix the issues before deploying         ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Common fixes:"
    echo "  • Install missing packages: conda activate d3 && pip install -r requirements.txt"
    echo "  • Start MinIO: docker start minio (or your MinIO command)"
    echo "  • Start PostgreSQL: sudo systemctl start postgresql"
    echo "  • Check GPU: nvidia-smi"
    echo ""
    exit 1
fi
