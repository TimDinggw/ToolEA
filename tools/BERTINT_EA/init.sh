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
DEFAULT_DATA_DIR="./dgw/AgentEA/data/EA/dbp15k_zh_en"
DEFAULT_SEED_ALIGNS="./dgw/AgentEA/data/EA/dbp15k_zh_en/seed_aligns"

#DEFAULT_DATA_DIR="./dgw/AgentEA/data/EA/openea_v1.1_D_Y_15K_V1"
#DEFAULT_SEED_ALIGNS="./dgw/AgentEA/data/EA/openea_v1.1_D_Y_15K_V1/seed_aligns"

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

# 记录开始时间
start_time=$SECONDS

# 激活conda环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate agentea

# 运行Python脚本
# cd basic_bert_unit/
# python main.py --data_dir "$DATA_DIR" --seed_aligns "$SEED_ALIGNS" > output1.log 2>&1
# cd ../interaction_model/
# python -u clean_attribute_data.py --data_dir "$DATA_DIR" --seed_aligns "$SEED_ALIGNS" > output1.log 2>&1
# python -u get_entity_embedding.py > output2.log 2>&1
# python -u get_attributeValue_embedding.py --data_dir "$DATA_DIR" --seed_aligns "$SEED_ALIGNS" > output3.log 2>&1
# python -u get_neighView_and_desView_interaction_feature.py --data_dir "$DATA_DIR" --seed_aligns "$SEED_ALIGNS" > output4.log 2>&1
# python -u get_attributeView_interaction_feature.py --data_dir "$DATA_DIR" --seed_aligns "$SEED_ALIGNS" > output5.log 2>&1
# python -u interaction_model.py > output6.log 2>&1

# python -u predictor.py > output7.log 2>&1



# 在 "# 运行Python脚本" 之前添加以下代码

# 更新 Param.py 中的 DATA_PATH
PARAM_FILE="./tools/BERTINT_EA/interaction_model/Param.py"
if [ -f "$PARAM_FILE" ]; then
    # 使用 sed 替换 DATA_PATH 行
    sed -i "s|^DATA_PATH =.*|DATA_PATH = \"$DATA_DIR/\"|" "$PARAM_FILE"
    echo "已更新 Param.py 中的 DATA_PATH 为: $DATA_DIR"
fi



python ./tools/BERTINT_EA/basic_bert_unit/main.py --data_dir "$DATA_DIR" --seed_aligns "$SEED_ALIGNS" > output1.log 2>&1
python ./tools/BERTINT_EA/interaction_model/clean_attribute_data.py --data_dir "$DATA_DIR" --seed_aligns "$SEED_ALIGNS" > output1.log 2>&1
python ./tools/BERTINT_EA/interaction_model/get_entity_embedding.py > output2.log 2>&1
python ./tools/BERTINT_EA/interaction_model/get_attributeValue_embedding.py --data_dir "$DATA_DIR" --seed_aligns "$SEED_ALIGNS" > output3.log 2>&1
python ./tools/BERTINT_EA/interaction_model/get_neighView_and_desView_interaction_feature.py --data_dir "$DATA_DIR" --seed_aligns "$SEED_ALIGNS" > output4.log 2>&1
python ./tools/BERTINT_EA/interaction_model/get_attributeView_interaction_feature.py --data_dir "$DATA_DIR" --seed_aligns "$SEED_ALIGNS" > output5.log 2>&1
python ./tools/BERTINT_EA/interaction_model/interaction_model.py


# 计算并显示执行时间
end_time=$SECONDS
execution_time=$((end_time - start_time))

echo "BERT-INT总执行时间: ${execution_time} 秒"