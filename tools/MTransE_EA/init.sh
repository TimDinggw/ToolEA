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

DEFAULT_DATA_DIR="/wangzhichun_gc/dgw/AgentEA/data/EA/dbp15k_zh_en"
DEFAULT_SEED_ALIGNS="/wangzhichun_gc/dgw/AgentEA/data/EA/dbp15k_zh_en/seed_aligns"

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

DATA_DIR=${DATA_DIR:-$DEFAULT_DATA_DIR}
SEED_ALIGNS=${SEED_ALIGNS:-$DEFAULT_SEED_ALIGNS}

start_time=$SECONDS

echo "=========================================="
echo "MTransE EA 预处理开始"
echo "数据目录: $DATA_DIR"
echo "种子对齐: $SEED_ALIGNS"
echo "=========================================="

source ~/anaconda3/etc/profile.d/conda.sh
conda activate eakit

#CMD="python ./tools/EAkit/run.py --log mtranse --data_dir $DATA_DIR --save ./tools/MTransE_EA/output --rate 0.3 --epoch 20 --check 10 --update 10 --train_batch_size 1024 --share --encoder '' --hiddens '100' --decoder 'TransE,MTransE_Align' --sampling '.,.' --k '0,0' --margin '0,0' --alpha '1,50' --feat_drop 0.0 --lr 0.01 --train_dist 'euclidean' --test_dist 'euclidean' --early"
CMD="python ./tools/EAkit/run.py --log mtranse --data_dir $DATA_DIR --seed_aligns $SEED_ALIGNS --rate 0.3 --epoch 400 --check 10 --update 10 --train_batch_size 5000 --encoder '' --hiddens '100' --decoder 'TransE,MTransE_Align' --sampling '.,.' --k '0,0' --margin '0,0' --alpha '1,50' --feat_drop 0.0 --lr 0.01 --train_dist 'euclidean' --test_dist 'euclidean' --early"

if [ -n "$SEED_ALIGNS" ]; then
  CMD="$CMD --seed_aligns $SEED_ALIGNS"
fi

echo "执行命令: $CMD"
eval $CMD


end_time=$SECONDS
execution_time=$((end_time - start_time))

echo "=========================================="
echo "MTransE EA 预处理完成"
echo "总执行时间: ${execution_time} 秒"
echo "=========================================="