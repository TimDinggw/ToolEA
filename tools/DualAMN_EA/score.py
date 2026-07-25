import pickle
import numpy as np
import argparse
import os

def calculate_alignment_scores(alignment_pairs, saved_file="embs.pkl"):
    """
    计算批量对齐对的相似度分数
    
    参数:
        alignment_pairs: 对齐对列表，每个元素为(source_id, target_id)
        saved_file: 保存的嵌入文件
        
    返回:
        列表: [(source_id, target_id, score), ...]
    """
    with open(saved_file, 'rb') as f:
        data = pickle.load(f)

    src_ents = data['src_entity_list']   # (N1,)
    tgt_ents = data['tgt_entity_list']   # (N2,)
    Lvec = data['Lvec']                  # (N1, d)
    Rvec = data['Rvec']                  # (N2, d)

    results = []
    
    # 构建实体ID到索引的映射
    src_id_to_idx = {int(id_): idx for idx, id_ in enumerate(src_ents)}
    tgt_id_to_idx = {int(id_): idx for idx, id_ in enumerate(tgt_ents)}
    
    # 计算全局相似度矩阵用于CSLS计算
    sim_mat = np.dot(Lvec, Rvec.T)  # (N1, N2)
    k = 10
    LR = np.mean(np.partition(sim_mat, -k, axis=1)[:, -k:], axis=1)  # (N1,)
    RL = np.mean(np.partition(sim_mat.T, -k, axis=1)[:, -k:], axis=1)  # (N2,)
    
    valid_count = 0
    invalid_count = 0
    
    for source_id, target_id in alignment_pairs:
        # 检查实体是否在对应的集合中
        if source_id not in src_id_to_idx:
            #print(f"警告: 源实体 {source_id} 不在源语言集合中")
            invalid_count += 1
            continue
            
        if target_id not in tgt_id_to_idx:
            #print(f"警告: 目标实体 {target_id} 不在目标语言集合中")
            invalid_count += 1
            continue
        
        # 获取实体索引
        src_idx = src_id_to_idx[source_id]
        tgt_idx = tgt_id_to_idx[target_id]
        
        # 计算余弦相似度
        cosine_sim = sim_mat[src_idx, tgt_idx]
        
        # 计算CSLS分数
        csls_score = 2 * cosine_sim - LR[src_idx] - RL[tgt_idx]
        
        results.append((source_id, target_id, float(csls_score)))
        valid_count += 1
    
    print(f"成功计算 {valid_count} 个对齐对的分数")
    if invalid_count > 0:
        print(f"跳过 {invalid_count} 个无效的对齐对")
    
    return results

def read_alignment_file(alignment_file_path):
    """读取对齐文件，格式为每行: source_id target_id"""
    alignment_pairs = []
    try:
        with open(alignment_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) != 2:
                    print(f"警告: 第{line_num}行格式错误，跳过: {line}")
                    continue
                
                try:
                    source_id = int(parts[0])
                    target_id = int(parts[1])
                    alignment_pairs.append((source_id, target_id))
                except ValueError:
                    print(f"警告: 第{line_num}行包含无效的实体ID，跳过: {line}")
    except Exception as e:
        print(f"读取对齐文件失败: {e}")
        return []
    
    return alignment_pairs

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Entity Alignment Scoring - Batch Mode')
    parser.add_argument('--query_file', type=str, required=True, help='对齐文件路径，每行格式: source_id target_id')
    parser.add_argument('--saved_file', type=str, default='embs.pkl', help='保存的嵌入文件路径')
    parser.add_argument('--output_file', type=str, default='scores.txt', help='输出结果文件路径')
    
    args = parser.parse_args()
    
    try:
        # 读取对齐文件
        alignment_pairs = read_alignment_file(args.query_file)
        if not alignment_pairs:
            print("错误: 对齐文件中没有有效的对齐对")
            return
        
        print(f"读取到 {len(alignment_pairs)} 个对齐对")
        
        # 调用批量评分函数
        scores = calculate_alignment_scores(
            alignment_pairs, 
            saved_file=args.saved_file
        )
        
        if not scores:
            print("错误: 没有成功计算任何对齐对的分数")
            return
        
        # 准备输出结果
        output_lines = []
        for source_id, target_id, score in scores:
            output_lines.append(f"{source_id}\t{target_id}\t{score:.6f}")
        
        # 写入结果到文件
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for line in output_lines:
                f.write(line + '\n')
        
        print(f"批量评分完成，共计算 {len(scores)} 个对齐对的分数")
        print(f"结果已保存到 {args.output_file}")
        
        # 显示一些统计信息
        scores_array = np.array([score for _, _, score in scores])
        print(f"分数统计:")
        print(f"  平均值: {np.mean(scores_array):.6f}")
        print(f"  标准差: {np.std(scores_array):.6f}")
        print(f"  最大值: {np.max(scores_array):.6f}")
        print(f"  最小值: {np.min(scores_array):.6f}")
        
    except Exception as e:
        print(f"发生错误: {e}")
        # 写入错误信息到文件
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(f"错误: {e}\n")

if __name__ == "__main__":
    main()