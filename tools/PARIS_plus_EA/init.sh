#!/bin/bash

# GPU自动检测函数
detect_gpu_count() {
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        if [ $GPU_COUNT -gt 0 ]; then
            echo "检测到 $GPU_COUNT 个GPU"
            return $GPU_COUNT
        fi
    fi
    echo "未检测到GPU，使用CPU模式"
    return 0
}
 
# 检测GPU数量
detect_gpu_count
gpu_count=$?
 
# 设置CUDA_VISIBLE_DEVICES
if [ $gpu_count -gt 0 ]; then
    export CUDA_VISIBLE_DEVICES=$(seq 0 $((gpu_count-1)) | tr '\n' ',' | sed 's/,$//')
    echo "CUDA_VISIBLE_DEVICES设置为: $CUDA_VISIBLE_DEVICES"
else
    export CUDA_VISIBLE_DEVICES="-1"
    echo "使用CPU模式"
fi

# 设置默认值
DEFAULT_DATA_DIR="/kgbnu3/dgw/AgentEA/data/EA/dbp15k_zh_en"
DEFAULT_SEED_ALIGNS="/kgbnu3/dgw/AgentEA/data/EA/dbp15k_zh_en/seed_aligns"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --data_dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --seed_aligns)
      SEED_ALIGNS="$2"
      shift 2
      ;;
    *)
      echo "未知参数: $1"
      exit 1
      ;;
  esac
done

# 使用默认值（如果参数未提供）
DATA_DIR=${DATA_DIR:-$DEFAULT_DATA_DIR}
SEED_ALIGNS=${SEED_ALIGNS:-$DEFAULT_SEED_ALIGNS}

# 激活conda环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate agentea

# 运行Python脚本
python ./tools/PARIS_plus_EA/run.py --data_dir "$DATA_DIR" --seed_aligns "$SEED_ALIGNS"