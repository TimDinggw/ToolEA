from typing import List, Set, Tuple

from data_loader import AlignmentPair


def evaluate_alignments(predictions: List[Tuple[str, str, float]], seed_aligns: List[Tuple[str, str, float]], ground_truth: List[AlignmentPair]) -> Tuple[float, float, float]:
    """评估对齐结果（排除种子对齐）
    
    Args:
        predictions: 预测的对齐结果列表 [(id_a, id_b, confidence)]
        seed_aligns: 种子对齐结果列表 [(id_a, id_b, confidence)]
        ground_truth: 真实对齐结果列表
        
    Returns:
        (Precision, Recall, F1-Score)
    """
    # 构建种子对齐集合
    seed_set = set((seed[0], seed[1]) for seed in seed_aligns)
    
    # 构建预测集合（排除种子对齐）
    pred_set = set((pred[0], pred[1]) for pred in predictions) - seed_set
    
    # 构建真实集合
    truth_set = set((truth.id_a, truth.id_b) for truth in ground_truth)
    
    # 计算TP、FP、FN
    true_positives = pred_set & truth_set
    false_positives = pred_set - truth_set
    false_negatives = truth_set - pred_set
    
    # 计算精确率
    precision = len(true_positives) / (len(true_positives) + len(false_positives)) if (len(true_positives) + len(false_positives)) > 0 else 0.0
    
    # 计算召回率
    recall = len(true_positives) / (len(true_positives) + len(false_negatives)) if (len(true_positives) + len(false_negatives)) > 0 else 0.0
    
    # 计算F1分数
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1_score

def print_evaluation_results(precision: float, recall: float, f1_score: float):
    """打印评估结果
    
    Args:
        precision: 精确率
        recall: 召回率
        f1_score: F1分数
    """
    print("\n===== 评估结果 =====")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数 (F1-Score): {f1_score:.4f}")
    print("===================\n")


if __name__ == "__main__":
    # 示例用法
    pass