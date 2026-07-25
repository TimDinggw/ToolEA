import pickle
import numpy as np
import argparse
import os

def bidirectional_csls_topk_batch(query_entities, saved_file="embs.pkl", top_k=10):
    """
    批量双向CSLS预测，根据实体ID判断其属于源语言还是目标语言，然后进行批量预测
    
    参数:
        query_entities: 查询实体ID列表
        saved_file: 保存的嵌入文件
        top_k: 返回的top-k结果
        
    返回:
        字典: {entity_id: [(target_id, score), ...]}
    """
    with open(saved_file, 'rb') as f:
        data = pickle.load(f)

    src_ents = data['src_entity_list']   # (N1,)
    tgt_ents = data['tgt_entity_list']   # (N2,)
    Lvec = data['Lvec']                  # (N1, d)
    Rvec = data['Rvec']                  # (N2, d)

    results = {}
    
    # 分离源实体和目标实体
    src_queries = []
    tgt_queries = []
    
    for entity_id in query_entities:
        if entity_id in src_ents:
            src_queries.append(entity_id)
        elif entity_id in tgt_ents:
            tgt_queries.append(entity_id)
        else:
            print(f"警告: 实体 {entity_id} 不在源语言或目标语言集合中")
    
    # 批量处理源语言到目标语言的预测
    if src_queries:
        print(f"处理 {len(src_queries)} 个源实体查询...")
        src_indices = [np.where(src_ents == entity_id)[0][0] for entity_id in src_queries]
        
        # 计算相似度矩阵
        sim_mat = np.dot(Lvec, Rvec.T)  # (N1, N2)
        k = 10
        LR = np.mean(np.partition(sim_mat, -k, axis=1)[:, -k:], axis=1)  # (N1,)
        RL = np.mean(np.partition(sim_mat.T, -k, axis=1)[:, -k:], axis=1)  # (N2,)
        
        for i, entity_id in enumerate(src_queries):
            query_idx = src_indices[i]
            query_sim = sim_mat[query_idx]  # (N2,)
            csls_scores = 2 * query_sim - LR[query_idx] - RL  # (N2,)
            top_indices = np.argsort(-csls_scores)[:top_k]
            results[entity_id] = [(int(tgt_ents[j]), float(csls_scores[j])) for j in top_indices]
    
    # 批量处理目标语言到源语言的预测
    if tgt_queries:
        print(f"处理 {len(tgt_queries)} 个目标实体查询...")
        tgt_indices = [np.where(tgt_ents == entity_id)[0][0] for entity_id in tgt_queries]
        
        # 计算相似度矩阵
        sim_mat = np.dot(Rvec, Lvec.T)  # (N2, N1)
        k = 10
        RL = np.mean(np.partition(sim_mat, -k, axis=1)[:, -k:], axis=1)  # (N2,)
        LR = np.mean(np.partition(sim_mat.T, -k, axis=1)[:, -k:], axis=1)  # (N1,)
        
        for i, entity_id in enumerate(tgt_queries):
            query_idx = tgt_indices[i]
            query_sim = sim_mat[query_idx]  # (N1,)
            csls_scores = 2 * query_sim - RL[query_idx] - LR  # (N1,)
            top_indices = np.argsort(-csls_scores)[:top_k]
            results[entity_id] = [(int(src_ents[j]), float(csls_scores[j])) for j in top_indices]
    
    return results

def read_query_file(query_file_path):
    """读取查询文件，格式为每行一个实体ID"""
    query_entities = []
    try:
        with open(query_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entity_id = int(line)
                        query_entities.append(entity_id)
                    except ValueError:
                        print(f"警告: 跳过无效的实体ID: {line}")
    except Exception as e:
        print(f"读取查询文件失败: {e}")
        return []
    
    return query_entities

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Entity Alignment Prediction - Batch Mode')
    parser.add_argument('--query_file', type=str, required=True, help='查询文件路径，每行一个实体ID')
    parser.add_argument('--saved_file', type=str, default='embs.pkl', help='保存的嵌入文件路径')
    parser.add_argument('--top_k', type=int, default=10, help='返回的top-k结果数量')
    parser.add_argument('--output_file', type=str, default='results.txt', help='输出结果文件路径')
    
    args = parser.parse_args()
    
    try:
        # 读取查询文件
        query_entities = read_query_file(args.query_file)
        if not query_entities:
            print("错误: 查询文件中没有有效的实体ID")
            return
        
        print(f"读取到 {len(query_entities)} 个查询实体")
        
        # 调用批量预测函数
        batch_results = bidirectional_csls_topk_batch(
            query_entities, 
            saved_file=args.saved_file, 
            top_k=args.top_k
        )
        
        # 准备输出结果
        output_lines = []
        for entity_id, results in batch_results.items():
            for target_id, score in results:
                output_lines.append(f"{entity_id}\t{target_id}\t{score:.6f}")
        
        # 写入结果到文件
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for line in output_lines:
                f.write(line + '\n')
        
        print(f"批量预测完成，共生成 {len(output_lines)} 个预测结果")
        print(f"结果已保存到 {args.output_file}")
        
    except ValueError as e:
        print(f"错误: {e}")
        # 写入错误信息到文件
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(f"错误: {e}\n")
    except Exception as e:
        print(f"发生未预期的错误: {e}")
        # 写入错误信息到文件
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(f"发生未预期的错误: {e}\n")

if __name__ == "__main__":
    main()