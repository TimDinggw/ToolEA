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

DEFAULT_DATA_DIR="./data/openea_v2_EN_DE_15K_V1/"

while [[ $# -gt 0 ]]; do
  case $1 in
    --data_dir)
      DATA_DIR="$2"
      shift 2
      ;;
    *)
      echo "未知参数: $1"
      exit 1
      ;;
  esac
done

DATA_DIR=${DATA_DIR:-$DEFAULT_DATA_DIR}

start_time=$SECONDS

echo "=========================================="
echo "RDGCN EA 预处理开始 (使用OpenEA框架)"
echo "数据目录: $DATA_DIR"
echo "=========================================="

source ~/anaconda3/etc/profile.d/conda.sh
conda activate openea

#cd ./tools/RDGCN_EA

python ./tools/RDGCN_EA/run_rdgcn.py ./tools/OpenEA/run/args/rdgcn_args_15K.json "$DATA_DIR"

#cd ../..

OUTPUT_EMBEDDINGS="./tools/RDGCN_EA/output/embeddings.npy"

OPENEA_OUTPUT="./tools/OpenEA/output/results/"

if [ -d "$OPENEA_OUTPUT" ]; then
    find "$OPENEA_OUTPUT" -name "*embeddings*.npy" -exec cp {} "$OUTPUT_EMBEDDINGS" \; 2>/dev/null
    if [ -f "$OUTPUT_EMBEDDINGS" ]; then
        echo "嵌入文件已复制到: $OUTPUT_EMBEDDINGS"
    else
        echo "警告: 未找到OpenEA生成的嵌入文件"
        find "$OPENEA_OUTPUT" -name "*.npy" -type f 2>/dev/null | head -5
    fi
fi

end_time=$SECONDS
execution_time=$((end_time - start_time))

echo "=========================================="
echo "RDGCN EA 预处理完成"
echo "总执行时间: ${execution_time} 秒"
echo "=========================================="