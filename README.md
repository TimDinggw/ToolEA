# Code for "ToolEA: A Multi-Tool Agentic Framework for Entity Alignment"

## Requirements
- python 3.10.19
- torch 2.9.0

## Datasets
- [DBP15K](https://github.com/nju-websoft/JAPE) 
- [SRPRS](https://github.com/nju-websoft/RSN)
- [OpenEA](https://github.com/nju-websoft/OpenEA)

## Data Preprocessing

- first run ./data_preprocess/openea_process.py or srprs_process.py
- then run ./data_preprocess/process_attr_triples_dbp.py or process_attr_triples_openea.py

## Alignment Models Implementation

- MTransE, TransEdge from [EAkit](https://github.com/THU-KEG/EAkit)
- [Dual-AMN](https://github.com/MaoXinn/Dual-AMN)
- [GCN-Align](https://github.com/1049451037/GCN-Align)
- [BERT-INT](https://github.com/kosugi11037/bert-int)
- [PAIRS](https://github.com/dig-team/PARIS)
- NameSimilarityTool from [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)

## Quick Run

```bash
nohup python -u main.py --task_type EA --data_path ./data/dbp15k_zh_en --percentage_per_round 0.2 > output_dbp15k_20%.log 2>&1 &
```