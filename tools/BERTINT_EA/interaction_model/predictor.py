import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.nn import init
from Param import *
import argparse

# 重新定义MlP类以匹配训练时的定义
class MlP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MlP, self).__init__()
        self.dense1 = nn.Linear(input_dim, hidden_dim, True)
        self.dense2 = nn.Linear(hidden_dim, 1, True)
        init.xavier_normal_(self.dense1.weight)
        init.xavier_normal_(self.dense2.weight)
    
    def forward(self, features):
        x = self.dense1(features)  # [B,h]
        x = F.relu(x)
        x = self.dense2(x)  # [B,1]
        x = F.tanh(x)
        x = torch.squeeze(x, 1)  # [B]
        return x

def load_interaction_model(model_path=None, features_path=None, entity_pairs_path=None):
    """
    加载交互模型和相关数据
    
    Args:
        model_path: 模型路径
        features_path: 特征路径
        entity_pairs_path: 实体对路径
    
    Returns:
        model, features, entity_pairs, entpair2f_idx, test_candidate
    """
    if model_path is None:
        model_path = INTERACTION_MODEL_SAVE_PATH
    
    print(f"Loading model from: {model_path}")
    
    # 使用weights_only=False来加载模型（注意安全风险）
    model = torch.load(model_path, map_location='cpu', weights_only=False)
    model.eval()
    
    # 加载特征
    if features_path is None:
        # 重建特征向量（根据原代码中的特征构建方式）
        entity_pairs = pickle.load(open(ENT_PAIRS_PATH, "rb"))
        
        # 加载各种特征
        nei_features = pickle.load(open(NEIGHBORVIEW_SIMILARITY_FEATURE_PATH, "rb"))
        att_features = pickle.load(open(ATTRIBUTEVIEW_SIMILARITY_FEATURE_PATH, 'rb'))
        des_features = pickle.load(open(DESVIEW_SIMILARITY_FEATURE_PATH, "rb"))
        
        # 合并特征
        all_features = []
        for i in range(len(entity_pairs)):
            all_features.append(nei_features[i] + att_features[i] + des_features[i])
        
        features = np.array(all_features, dtype=np.float32)
    else:
        features = pickle.load(open(features_path, "rb"))
        entity_pairs = pickle.load(open(entity_pairs_path, "rb"))
    
    # 创建实体对到特征索引的映射
    entpair2f_idx = {entpair: feature_idx for feature_idx, entpair in enumerate(entity_pairs)}
    
    # 加载候选集
    test_candidate = pickle.load(open(TEST_CANDIDATES_PATH, "rb"))
    
    print(f"Model loaded successfully. Feature shape: {features.shape}")
    print(f"Total entity pairs: {len(entity_pairs)}")
    print(f"Test candidates loaded for {len(test_candidate)} entities")
    
    return model, features, entity_pairs, entpair2f_idx, test_candidate


def predict_topk_alignments(entity_id, model, features, entpair2f_idx, test_candidate, k=10):
    """
    对单个实体预测top-k对齐结果
    
    Args:
        entity_id: 输入实体ID
        model: 训练好的交互模型
        features: 所有特征向量
        entpair2f_idx: 实体对到特征索引的映射
        test_candidate: 测试候选集
        k: top-k数量
    
    Returns:
        List of tuples: [(aligned_entity_id, similarity_score), ...]
    """
    if entity_id not in test_candidate:
        #print(f"Entity {entity_id} not found in test candidates.")
        return []
    
    # 获取该实体的候选对齐实体列表
    candidate_entities = test_candidate[entity_id]
    
    # 准备特征向量
    pairs_to_evaluate = [(entity_id, cand_id) for cand_id in candidate_entities]
    valid_pairs = []
    feature_indices = []
    
    for pair in pairs_to_evaluate:
        if pair in entpair2f_idx:
            valid_pairs.append(pair)
            feature_indices.append(entpair2f_idx[pair])
    
    if len(feature_indices) == 0:
        print(f"No valid pairs found for entity {entity_id}")
        return []
    
    # 提取特征
    features_tensor = torch.FloatTensor([features[idx] for idx in feature_indices])
    
    # 使用模型预测相似度分数
    with torch.no_grad():
        scores = model(features_tensor)
        # 使用sigmoid将分数归一化到0-1范围，或者保持原始tanh输出（-1到1之间）
        # 这里我们保持原始输出，因为tanh已经将值限制在-1到1之间
        scores_normalized = torch.sigmoid(scores)  # 转换到0-1区间
    
    # 获取分数列表
    score_list = scores_normalized.cpu().numpy().tolist()
    
    # 组合实体对和分数
    results = [(pair[1], score) for pair, score in zip(valid_pairs, score_list)]
    
    # 按分数排序并返回top-k
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k]


def calculate_pair_similarity(entity1_id, entity2_id, model, features, entpair2f_idx):
    """
    计算两个实体之间的对齐相似度
    
    Args:
        entity1_id: 第一个实体ID
        entity2_id: 第二个实体ID
        model: 训练好的交互模型
        features: 所有特征向量
        entpair2f_idx: 实体对到特征索引的映射
    
    Returns:
        float: 相似度分数
    """
    pair = (entity1_id, entity2_id)
    
    if pair not in entpair2f_idx:
        # 如果实体对不存在，尝试反向对
        rev_pair = (entity2_id, entity1_id)
        if rev_pair not in entpair2f_idx:
            #print(f"Entity pair ({entity1_id}, {entity2_id}) not found in entity pairs.")
            return 0.0  # 无法计算相似度
        pair = rev_pair
    
    feature_idx = entpair2f_idx[pair]
    feature_vector = torch.FloatTensor([features[feature_idx]])
    
    with torch.no_grad():
        score = model(feature_vector)
        # 使用sigmoid将分数归一化到0-1范围
        similarity = torch.sigmoid(score).item()
    
    return similarity


def process_single_entity_file(input_file, output_file, k=10):
    """
    处理每行一个实体ID的文件，输出top-k对齐结果
    
    Args:
        input_file: 输入文件路径（每行一个实体ID）
        output_file: 输出文件路径
        k: top-k数量
    """
    print("Loading model and data...")
    # 加载模型和数据
    model, features, entity_pairs, entpair2f_idx, test_candidate = load_interaction_model()
    
    print(f"Processing single entity file: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        processed_count = 0
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                entity_id = int(line)
                topk_results = predict_topk_alignments(
                    entity_id, model, features, entpair2f_idx, test_candidate, k
                )
                
                # 写入结果
                for i, (eid, score) in enumerate(topk_results):
                    result_line = f"{entity_id}\t{eid}\t{score:.6f}"
                    outfile.write(result_line + '\n')
                processed_count += 1
                
                # if processed_count % 100 == 0:
                #     print(f"Processed {processed_count} entities...")
                
            # except ValueError:
            #     print(f"Warning: Invalid entity ID at line {line_num}: {line}")
            # except Exception as e:
            #     print(f"Error processing line {line_num}: {line}, Error: {str(e)}")
            except Exception as e:
                pass

    print(f"Finished processing. Total entities processed: {processed_count}")


def process_entity_pair_file(input_file, output_file):
    """
    处理每行两个实体ID的文件，输出对齐相似度
    
    Args:
        input_file: 输入文件路径（每行两个实体ID，用空格或制表符分隔）
        output_file: 输出文件路径
    """
    print("Loading model and data...")
    # 加载模型和数据
    model, features, entity_pairs, entpair2f_idx, _ = load_interaction_model()
    
    print(f"Processing entity pair file: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        processed_count = 0
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                parts = line.split()
                if len(parts) >= 2:
                    entity1_id = int(parts[0])
                    entity2_id = int(parts[1])
                    similarity = calculate_pair_similarity(
                        entity1_id, entity2_id, model, features, entpair2f_idx
                    )
                    
                    # 写入结果
                    result_line = f"{entity1_id}\t{entity2_id}\t{similarity:.6f}"
                    outfile.write(result_line + '\n')
                    processed_count += 1
                    
                    # if processed_count % 100 == 0:
                    #     print(f"Processed {processed_count} entity pairs...")
                else:
                    print(f"Warning: Line {line_num} doesn't contain two entities: {line}")
            
            except ValueError:
                print(f"Warning: Invalid entity IDs at line {line_num}: {line}")
            except Exception as e:
                print(f"Error processing line {line_num}: {line}, Error: {str(e)}")
    
    print(f"Finished processing. Total entity pairs processed: {processed_count}")


def main():
    """
    主函数示例 - 你可以根据需要调用相应的函数
    """
    # print("Entity Alignment Processing Tool")
    # print("1. Process single entity file for top-k alignments")
    # print("2. Process entity pair file for similarity scores")
    # print("3. Example usage")
    
    # # 示例：展示如何使用函数（但不实际运行，除非提供了输入文件）
    # print("\nTo use this tool:")
    # print("- For single entity processing: process_single_entity_file('input_entities.txt', 'output_topk_alignments.txt', k=10)")
    # print("- For entity pair processing: process_entity_pair_file('input_entity_pairs.txt', 'output_similarities.txt)")
    # process_single_entity_file('input_entities.txt', 'output_topk_alignments.txt', k=10)
    # process_entity_pair_file('input_entity_pairs.txt', 'output_similarities.txt')
    
    parser = argparse.ArgumentParser(description='BERT_INT模型参数')
    parser.add_argument('--query_file', type=str, default='/kgbnu3/dgw/AgentEA/data/EA/dbp15k_zh_en', help='数据目录路径')
    parser.add_argument('--top_k', type=int, default=10, help='返回的top-k结果数量')
    parser.add_argument('--output_file', type=str, default='results.txt', help='输出结果文件路径')
    parser.add_argument('--mode', type=str, default='predict', help='模式')

    args = parser.parse_args()
    
    if args.mode == 'predict':
        process_single_entity_file(args.query_file, args.output_file, args.top_k)
    else:
        process_entity_pair_file(args.query_file, args.output_file)

if __name__ == '__main__':
    main()