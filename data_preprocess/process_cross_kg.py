import os
import shutil
from collections import defaultdict

def convert_dbp_fb_to_ea_format():
    """
    将DBP-FB数据集转换为EA数据格式，使用统一的实体和关系编号
    """
    # 输入目录
    input_base_dir = "./DBP-FB"
    dbp_dir = os.path.join(input_base_dir, "DBP")
    fb_dir = os.path.join(input_base_dir, "FB")
    cross_dir = os.path.join(input_base_dir, "cross")
    
    # 输出目录
    output_base_dir = input_base_dir + "_processed"
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 创建输出文件
    ent_ids_1_path = os.path.join(output_base_dir, "ent_ids_1")  # DBP实体
    ent_ids_2_path = os.path.join(output_base_dir, "ent_ids_2")  # FB实体
    rel_ids_1_path = os.path.join(output_base_dir, "rel_ids_1")  # DBP关系
    rel_ids_2_path = os.path.join(output_base_dir, "rel_ids_2")  # FB关系
    rel_triples_1_path = os.path.join(output_base_dir, "rel_triples_1")  # DBP三元组
    rel_triples_2_path = os.path.join(output_base_dir, "rel_triples_2")  # FB三元组
    seed_aligns_path = os.path.join(output_base_dir, "seed_aligns")  # 已知对齐
    test_aligns_path = os.path.join(output_base_dir, "test_aligns")  # 测试对齐
    cross_test_path = os.path.join(output_base_dir, "cross_test")  # cross_test转换后的文件
    
    # 存储所有实体和关系的集合
    all_entities = set()
    all_relations = set()
    
    # 读取DBP训练数据，提取实体和关系
    dbp_train_path = os.path.join(dbp_dir, "train.txt")
    with open(dbp_train_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                head, relation, tail = parts[0], parts[1], parts[2]
                all_entities.add(head)
                all_entities.add(tail)
                all_relations.add(relation)
    
    # 读取FB训练数据，提取实体和关系
    fb_train_path = os.path.join(fb_dir, "train.txt")
    with open(fb_train_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                head, relation, tail = parts[0], parts[1], parts[2]
                all_entities.add(head)
                all_entities.add(tail)
                all_relations.add(relation)
    
    # 创建统一的实体ID映射
    entity_to_id = {entity: f"{i}" for i, entity in enumerate(sorted(list(all_entities)))}
    
    # 创建统一的关系ID映射
    relation_to_id = {rel: f"{i}" for i, rel in enumerate(sorted(list(all_relations)))}
    
    # 分离DBP和FB的实体和关系
    dbp_entities = set()
    fb_entities = set()
    dbp_relations = set()
    fb_relations = set()
    
    # 通过训练数据确定哪些实体属于哪个图谱
    with open(dbp_train_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                head, relation, tail = parts[0], parts[1], parts[2]
                dbp_entities.add(head)
                dbp_entities.add(tail)
                dbp_relations.add(relation)
    
    with open(fb_train_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                head, relation, tail = parts[0], parts[1], parts[2]
                fb_entities.add(head)
                fb_entities.add(tail)
                fb_relations.add(relation)
    
    # 写入ent_ids_1 (DBP实体)
    with open(ent_ids_1_path, 'w', encoding='utf-8') as f:
        for entity in sorted(list(dbp_entities)):
            eid = entity_to_id[entity]
            f.write(f"{eid}\t{entity}\n")
    
    # 写入ent_ids_2 (FB实体)
    with open(ent_ids_2_path, 'w', encoding='utf-8') as f:
        for entity in sorted(list(fb_entities)):
            eid = entity_to_id[entity]
            f.write(f"{eid}\t{entity}\n")
    
    # 写入rel_ids_1 (DBP关系)
    with open(rel_ids_1_path, 'w', encoding='utf-8') as f:
        for rel in sorted(list(dbp_relations)):
            if rel in relation_to_id:  # 确保关系存在
                rid = relation_to_id[rel]
                f.write(f"{rid}\t{rel}\n")
    
    # 写入rel_ids_2 (FB关系)
    with open(rel_ids_2_path, 'w', encoding='utf-8') as f:
        for rel in sorted(list(fb_relations)):
            if rel in relation_to_id:  # 确保关系存在
                rid = relation_to_id[rel]
                f.write(f"{rid}\t{rel}\n")
    
    # 处理DBP三元组
    with open(os.path.join(dbp_dir, "train.txt"), 'r', encoding='utf-8') as input_f, \
         open(rel_triples_1_path, 'w', encoding='utf-8') as output_f:
        for line in input_f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                head, relation, tail = parts[0], parts[1], parts[2]
                head_id = entity_to_id[head]
                rel_id = relation_to_id[relation]
                tail_id = entity_to_id[tail]
                output_f.write(f"{head_id}\t{rel_id}\t{tail_id}\n")
    
    # 处理FB三元组
    with open(os.path.join(fb_dir, "train.txt"), 'r', encoding='utf-8') as input_f, \
         open(rel_triples_2_path, 'w', encoding='utf-8') as output_f:
        for line in input_f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                head, relation, tail = parts[0], parts[1], parts[2]
                head_id = entity_to_id[head]
                rel_id = relation_to_id[relation]
                tail_id = entity_to_id[tail]
                output_f.write(f"{head_id}\t{rel_id}\t{tail_id}\n")
    
    # 处理已知对齐 (known_shared_entities.txt)
    print("正在处理已知对齐...")
    seed_alignments = set()
    with open(os.path.join(cross_dir, "known_shared_entities.txt"), 'r', encoding='utf-8') as input_f:
        for line in input_f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                dbp_entity, fb_entity = parts[0], parts[1]
                if dbp_entity in entity_to_id and fb_entity in entity_to_id:
                    dbp_id = entity_to_id[dbp_entity]
                    fb_id = entity_to_id[fb_entity]
                    seed_alignments.add((dbp_id, fb_id))
    
    with open(seed_aligns_path, 'w', encoding='utf-8') as output_f:
        for dbp_id, fb_id in seed_alignments:
            output_f.write(f"{dbp_id}\t{fb_id}\n")
    
    # 处理测试对齐 (all_shared_entities.txt中不在known_shared_entities.txt中的pairs)
    print("正在处理测试对齐...")
    all_shared_pairs = set()
    with open(os.path.join(cross_dir, "all_shared_entities.txt"), 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                dbp_entity, fb_entity = parts[0], parts[1]
                if dbp_entity in entity_to_id and fb_entity in entity_to_id:
                    dbp_id = entity_to_id[dbp_entity]
                    fb_id = entity_to_id[fb_entity]
                    all_shared_pairs.add((dbp_id, fb_id))
    
    # 获取测试对齐：all_shared_entities中不在known_shared_entities中的实体对
    test_alignments = all_shared_pairs - seed_alignments
    
    with open(test_aligns_path, 'w', encoding='utf-8') as f:
        for dbp_id, fb_id in test_alignments:
            f.write(f"{dbp_id}\t{fb_id}\n")

    # 处理cross_test.txt，将其转换为ID形式
    print("正在处理cross_test.txt...")
    with open(os.path.join(cross_dir, "cross_test.txt"), 'r', encoding='utf-8') as input_f, \
         open(cross_test_path, 'w', encoding='utf-8') as output_f:
        for line in input_f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                head, relation, tail = parts[0], parts[1], parts[2]
                
                # 检查实体和关系是否存在于映射中
                if head in entity_to_id and tail in entity_to_id and relation in relation_to_id:
                    head_id = entity_to_id[head]
                    rel_id = relation_to_id[relation]
                    tail_id = entity_to_id[tail]
                    output_f.write(f"{head_id}\t{rel_id}\t{tail_id}\n")
                else:
                    # 如果实体或关系不存在于映射中，则跳过
                    continue

    print(f"转换完成！")
    print(f"总实体数量: {len(all_entities)}")
    print(f"DBP实体数量: {len(dbp_entities)}")
    print(f"FB实体数量: {len(fb_entities)}")
    print(f"总关系数量: {len(all_relations)}")
    print(f"DBP三元组数量: {sum(1 for _ in open(os.path.join(dbp_dir, 'train.txt'), 'r', encoding='utf-8'))}")
    print(f"FB三元组数量: {sum(1 for _ in open(os.path.join(fb_dir, 'train.txt'), 'r', encoding='utf-8'))}")
    print(f"种子对齐数量: {sum(1 for _ in open(seed_aligns_path, 'r', encoding='utf-8'))}")
    print(f"测试对齐数量: {sum(1 for _ in open(test_aligns_path, 'r', encoding='utf-8'))}")
    print(f"Cross测试三元组数量: {sum(1 for _ in open(cross_test_path, 'r', encoding='utf-8'))}")
    
    print(f"\n输出文件已保存到: {output_base_dir}/")
    print("- ent_ids_1: DBP实体ID映射")
    print("- ent_ids_2: FB实体ID映射")
    print("- rel_ids_1: DBP关系ID映射")
    print("- rel_ids_2: FB关系ID映射")
    print("- rel_triples_1: DBP关系三元组")
    print("- rel_triples_2: FB关系三元组")
    print("- seed_aligns: 已知对齐实体")
    print("- test_aligns: 测试对齐实体")
    print("- cross_test: 跨知识图谱测试三元组（ID形式）")
    
if __name__ == "__main__":
    convert_dbp_fb_to_ea_format()