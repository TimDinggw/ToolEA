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
OUTPUT_FILE="./tools/BERTINT_EA/results.txt"

# 解析命令行参数
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
        --saved_file)
            SAVED_FILE="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 检查对齐文件是否存在
if [ -z "$QUERY_FILE" ]; then
    echo "错误: 必须指定对齐文件路径 --QUERY_FILE"
    echo "用法: ./score.sh --QUERY_FILE <对齐文件路径> [--output_file <输出文件路径>] [--saved_file <嵌入文件路径>]"
    echo ""
    echo "对齐文件格式:"
    echo "  每行一个对齐对，格式为: source_entity_id target_entity_id"
    echo "  例如:"
    echo "      123 456"
    echo "      789 101"
    echo "      112 131"
    exit 1
fi

if [ ! -f "$QUERY_FILE" ]; then
    echo "错误: 对齐文件不存在: $QUERY_FILE"
    exit 1
fi


# 执行Python脚本进行批量评分
python ./tools/BERTINT_EA/interaction_model/predictor.py --query_file "$QUERY_FILE" --output_file "$OUTPUT_FILE" --mode "score"

echo "批量评分完成，请查看 $OUTPUT_FILE 文件"