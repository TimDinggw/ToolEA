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
# 激活conda环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate agentea

# 初始化变量
QUERY_FILE=""
top_k=10
OUTPUT_FILE="./tools/DualAMN_EA/results.txt"

# 解析命令行参数
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
        --entity_id)
            echo "警告: --entity_id参数已弃用，请使用--query_file进行批量查询"
            shift 2
            ;;
        --entity_source)
            echo "警告: --entity_source参数已弃用，请使用--query_file进行批量查询"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 检查查询文件是否存在
if [ -z "$QUERY_FILE" ]; then
    echo "错误: 必须指定查询文件路径 --query_file"
    exit 1
fi

if [ ! -f "$QUERY_FILE" ]; then
    echo "错误: 查询文件不存在: $QUERY_FILE"
    exit 1
fi

# 检查top_k是否为正整数
if ! [[ "$top_k" =~ ^[0-9]+$ ]] || [ "$top_k" -le 0 ]; then
    echo "错误: top_k必须是正整数"
    exit 1
fi

# 执行Python脚本进行批量查询
python ./tools/DualAMN_EA/predict.py --query_file "$QUERY_FILE" --output_file "$OUTPUT_FILE" --top_k "$top_k"

echo "批量预测完成，返回前 $top_k 个结果，请查看 $OUTPUT_FILE 文件"