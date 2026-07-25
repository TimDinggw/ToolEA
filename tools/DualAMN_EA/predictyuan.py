import pickle
import numpy as np

def bidirectional_csls_topk(query_entity_id, saved_file="embs.pkl", top_k=10, batch_size=1024):
    """
    双向CSLS预测，根据实体ID判断其属于源语言还是目标语言，然后进行双向预测
    
    参数:
        query_entity_id: 查询实体ID
        saved_file: 保存的嵌入文件
        top_k: 返回的top-k结果
        batch_size: 批处理大小
        
    返回:
        如果是源语言实体: (direction, results) direction为"src_to_tgt", results为预测结果
        如果是目标语言实体: (direction, results) direction为"tgt_to_src", results为预测结果
    """
    with open(saved_file, 'rb') as f:
        data = pickle.load(f)

    src_ents = data['src_entity_list']   # (N1,)
    tgt_ents = data['tgt_entity_list']   # (N2,)
    Lvec = data['Lvec']                  # (N1, d)
    Rvec = data['Rvec']                  # (N2, d)

    # 判断实体属于哪个语言集
    if query_entity_id in src_ents:
        direction = "src_to_tgt"
        # 源语言到目标语言的预测
        query_idx = np.where(src_ents == query_entity_id)[0][0]
        sim_mat = np.dot(Lvec, Rvec.T)  # (N1, N2)
        k = 10
        LR = np.mean(np.partition(sim_mat, -k, axis=1)[:, -k:], axis=1)  # (N1,)
        RL = np.mean(np.partition(sim_mat.T, -k, axis=1)[:, -k:], axis=1)  # (N2,)
        query_sim = sim_mat[query_idx]  # (N2,)
        csls_scores = 2 * query_sim - LR[query_idx] - RL  # (N2,)
        top_indices = np.argsort(-csls_scores)[:top_k]
        results = [(int(tgt_ents[i]), float(csls_scores[i])) for i in top_indices]
    elif query_entity_id in tgt_ents:
        direction = "tgt_to_src"
        # 目标语言到源语言的预测
        query_idx = np.where(tgt_ents == query_entity_id)[0][0]
        sim_mat = np.dot(Rvec, Lvec.T)  # (N2, N1)
        k = 10
        RL = np.mean(np.partition(sim_mat, -k, axis=1)[:, -k:], axis=1)  # (N2,)
        LR = np.mean(np.partition(sim_mat.T, -k, axis=1)[:, -k:], axis=1)  # (N1,)
        query_sim = sim_mat[query_idx]  # (N1,)
        csls_scores = 2 * query_sim - RL[query_idx] - LR  # (N1,)
        top_indices = np.argsort(-csls_scores)[:top_k]
        results = [(int(src_ents[i]), float(csls_scores[i])) for i in top_indices]
    else:
        raise ValueError(f"Entity {query_entity_id} is not in either source or target language set.")

    return direction, results


# 示例用法
if __name__ == "__main__":
    # 测试源语言实体
    src_entity_id = 0
    try:
        direction, topk = bidirectional_csls_topk(src_entity_id, saved_file="embs.pkl", top_k=5)
        print(f"\n{direction} predictions for source entity {src_entity_id}:")
        for ent, score in topk:
            print(f"  Target Entity: {ent}, CSLS Score: {score:.6f}")
    except Exception as e:
        print("Error:", e)
    
    # 测试目标语言实体
    tgt_entity_id = 10500
    try:
        direction, topk = bidirectional_csls_topk(tgt_entity_id, saved_file="embs.pkl", top_k=5)
        print(f"\n{direction} predictions for target entity {tgt_entity_id}:")
        for ent, score in topk:
            print(f"  Source Entity: {ent}, CSLS Score: {score:.6f}")
    except Exception as e:
        print("Error:", e)
