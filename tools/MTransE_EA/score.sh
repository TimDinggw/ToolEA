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

QUERY_FILE=""
OUTPUT_FILE="./tools/MTransE_EA/results.txt"
EMBEDDINGS_FILE="./tools/MTransE_EA/output/embeddings.npy"

while [[ $# -gt 0 ]]; do
    case $1 in
        --query_file)
            QUERY_FILE="$2"
            shift 2
            ;;
        --output_file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

if [ -z "$QUERY_FILE" ]; then
    echo "错误: 必须指定对齐文件路径 --query_file"
    echo "用法: ./score.sh --query_file <对齐文件路径> [--output_file <输出文件路径>]"
    exit 1
fi

if [ ! -f "$QUERY_FILE" ]; then
    echo "错误: 对齐文件不存在: $QUERY_FILE"
    exit 1
fi

if [ ! -f "$EMBEDDINGS_FILE" ]; then
    echo "错误: 嵌入文件不存在: $EMBEDDINGS_FILE"
    echo "请先运行 init.sh 进行训练"
    exit 1
fi

echo "=========================================="
echo "MTransE EA 批量评分"
echo "查询文件: $QUERY_FILE"
echo "输出文件: $OUTPUT_FILE"
echo "=========================================="

source ~/anaconda3/etc/profile.d/conda.sh
conda activate eakit

python -c "
import numpy as np

embeddings_file = '$EMBEDDINGS_FILE'
query_file = '$QUERY_FILE'
output_file = '$OUTPUT_FILE'

embeddings = np.load(embeddings_file)

n_entities = embeddings.shape[0]
n1 = n_entities // 2

# 首先收集所有距离，用于归一化
all_distances = []
valid_pairs = []

with open(query_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) >= 2:
            id_a = int(parts[0])
            id_b = int(parts[1])
            
            vec_a = embeddings[id_a]
            if id_b < n1:
                vec_b = embeddings[n1 + id_b]
            elif id_b < n_entities:
                vec_b = embeddings[id_b]
            else:
                continue
                
            # 计算两个实体向量之间的欧几里得距离
            distance = np.linalg.norm(vec_a - vec_b)
            all_distances.append(distance)
            valid_pairs.append((id_a, id_b))

# 计算距离的最小值和最大值，用于归一化
if all_distances:
    min_distance = min(all_distances)
    max_distance = max(all_distances)
    distance_range = max_distance - min_distance if max_distance > min_distance else 1.0
else:
    min_distance = 0
    max_distance = 1
    distance_range = 1.0

# 计算每个对齐对的相似度并保存结果
results = []
for (id_a, id_b), distance in zip(valid_pairs, all_distances):
    # 将距离归一化到[0,1]区间，然后用1减去得到相似度
    normalized_distance = (distance - min_distance) / distance_range
    similarity = 1 - normalized_distance
    results.append((id_a, id_b, similarity))

with open(output_file, 'w', encoding='utf-8') as f:
    for id_a, id_b, score in results:
        f.write(f'{id_a}\t{id_b}\t{score}\n')

print(f'成功评分 {len(results)} 个对齐对')
"

echo "批量评分完成"