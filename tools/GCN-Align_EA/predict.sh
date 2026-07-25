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
top_k=10
OUTPUT_FILE="./tools/GCN-Align_EA/results.txt"
EMBEDDINGS_FILE="./tools/GCN-Align_EA/output/embeddings.npy"

while [[ $# -gt 0 ]]; do
    case $1 in
        --query_file)
            QUERY_FILE="$2"
            shift 2
            ;;
        --top_k)
            top_k="$2"
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
    echo "错误: 必须指定查询文件路径 --query_file"
    exit 1
fi

if [ ! -f "$QUERY_FILE" ]; then
    echo "错误: 查询文件不存在: $QUERY_FILE"
    exit 1
fi

if [ ! -f "$EMBEDDINGS_FILE" ]; then
    echo "错误: 嵌入文件不存在: $EMBEDDINGS_FILE"
    exit 1
fi

if ! [[ "$top_k" =~ ^[0-9]+$ ]] || [ "$top_k" -le 0 ]; then
    echo "错误: top_k必须是正整数"
    exit 1
fi

echo "=========================================="
echo "GCN-Align EA 批量预测"
echo "查询文件: $QUERY_FILE"
echo "top_k: $top_k"
echo "输出文件: $OUTPUT_FILE"
echo "=========================================="

source ~/anaconda3/etc/profile.d/conda.sh
conda activate eakit

python -c "
import numpy as np

embeddings_file = '$EMBEDDINGS_FILE'
query_file = '$QUERY_FILE'
output_file = '$OUTPUT_FILE'
top_k = int('$top_k')

embeddings = np.load(embeddings_file)

n_entities = embeddings.shape[0]
n1 = n_entities // 2

src_embeddings = embeddings[:n1]
tgt_embeddings = embeddings[n1:]

query_entities = []
with open(query_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) >= 1:
            query_entities.append(int(parts[0]))

# 收集所有查询实体的距离，用于归一化
all_distances = []
for src_id in query_entities:
    if src_id >= n1:
        continue
    
    src_vec = src_embeddings[src_id:src_id+1]
    distances = np.linalg.norm(src_vec - tgt_embeddings, axis=1)
    all_distances.extend(distances)

# 计算距离的最小值和最大值，用于归一化
if all_distances:
    min_distance = min(all_distances)
    max_distance = max(all_distances)
    # 避免除以零
    distance_range = max_distance - min_distance if max_distance > min_distance else 1.0
else:
    min_distance = 0
    max_distance = 1
    distance_range = 1.0

results = {}
for src_id in query_entities:
    if src_id >= n1:
        continue
    
    src_vec = src_embeddings[src_id:src_id+1]
    
    # 使用欧几里得距离，与GCN-Align模型一致
    distances = np.linalg.norm(src_vec - tgt_embeddings, axis=1)
    
    # 将距离归一化到[0,1]区间，然后用1减去得到相似度（距离越小，相似度越高）
    normalized_distances = (distances - min_distance) / distance_range
    similarities = 1 - normalized_distances
    
    top_indices = np.argsort(-similarities)[:top_k]
    top_scores = similarities[top_indices]
    
    results[str(src_id)] = []
    for tgt_idx, score in zip(top_indices, top_scores):
        results[str(src_id)].append((str(tgt_idx), float(score)))

with open(output_file, 'w', encoding='utf-8') as f:
    for entity_id, targets in results.items():
        for target_id, score in targets:
            f.write(f'{entity_id}\t{target_id}\t{score}\n')

print(f'成功预测 {len(results)} 个实体的top-{top_k}结果')
"

echo "批量预测完成"