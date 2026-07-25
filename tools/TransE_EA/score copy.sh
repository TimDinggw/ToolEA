#!/bin/bash

QUERY_FILE=""
OUTPUT_FILE="./tools/TransE_EA/results.txt"
EMBEDDINGS_FILE="./tools/TransE_EA/embeddings.npy"

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
echo "TransE EA 批量评分"
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
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

n_entities = embeddings.shape[0]
n1 = n_entities // 2

results = []
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
                
            cosine_sim = float(np.dot(vec_a, vec_b))
            results.append((id_a, id_b, cosine_sim))

with open(output_file, 'w', encoding='utf-8') as f:
    for id_a, id_b, score in results:
        f.write(f'{id_a}\t{id_b}\t{score}\n')

print(f'成功计算 {len(results)} 个对齐对的分数')
"

echo "批量评分完成"