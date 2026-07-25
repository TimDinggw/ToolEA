#!/bin/bash

# 设置默认值
DEFAULT_DATA_DIR="/kgbnu3/dgw/AgentEA/data/EA/dbp15k_zh_en"
DEFAULT_SEED_ALIGNS="/kgbnu3/dgw/AgentEA/data/EA/dbp15k_zh_en/seed_aligns.txt"

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
conda activate tf1

# 运行Python脚本
python ./tools/Dual-AMN_EA/run.py --data_dir "$DATA_DIR" --seed_aligns "$SEED_ALIGNS"