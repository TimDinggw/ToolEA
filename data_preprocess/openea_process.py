import os
import shutil

# 设置源目录和目标目录
source_dir = "OpenEA_dataset_v1.1"
source_dir = "OpenEA_dataset_v2.0"
dest_dir = source_dir + '_processed'  # 修复拼写错误

# 确保目标目录存在
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# 定义处理单个数据集的函数
def process_dataset(dataset_name, source_path, dest_path):
    print(f"Processing dataset: {dataset_name}")
    
    # 创建目标目录
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    
    # 复制原始文件到临时位置
    temp_rel_triples_1 = os.path.join(dest_path, "temp_rel_triples_1")
    temp_rel_triples_2 = os.path.join(dest_path, "temp_rel_triples_2")
    temp_attr_triples_1 = os.path.join(dest_path, "temp_attr_triples_1")
    temp_attr_triples_2 = os.path.join(dest_path, "temp_attr_triples_2")
    temp_seed_aligns = os.path.join(dest_path, "temp_seed_aligns")
    temp_test_aligns = os.path.join(dest_path, "temp_test_aligns")
    
    # 复制关系三元组文件
    rel_triples_1_src = os.path.join(source_path, "rel_triples_1")
    rel_triples_2_src = os.path.join(source_path, "rel_triples_2")
    if os.path.exists(rel_triples_1_src):
        shutil.copy2(rel_triples_1_src, temp_rel_triples_1)
    if os.path.exists(rel_triples_2_src):
        shutil.copy2(rel_triples_2_src, temp_rel_triples_2)
    
    # 复制属性三元组文件
    attr_triples_1_src = os.path.join(source_path, "attr_triples_1")
    attr_triples_2_src = os.path.join(source_path, "attr_triples_2")
    if os.path.exists(attr_triples_1_src):
        shutil.copy2(attr_triples_1_src, temp_attr_triples_1)
    if os.path.exists(attr_triples_2_src):
        shutil.copy2(attr_triples_2_src, temp_attr_triples_2)
    
    # 复制对齐文件
    train_links_src = os.path.join(source_path, "721_5fold", "3", "train_links")
    test_links_src = os.path.join(source_path, "721_5fold", "3", "test_links")
    if os.path.exists(train_links_src):
        shutil.copy2(train_links_src, temp_seed_aligns)
    if os.path.exists(test_links_src):
        shutil.copy2(test_links_src, temp_test_aligns)
    
    # 生成实体ID映射文件，并获取最大ID
    ent_ids_1_file = os.path.join(dest_path, "ent_ids_1")
    ent_ids_2_file = os.path.join(dest_path, "ent_ids_2")
    max_ent_id_1 = generate_entity_ids(temp_rel_triples_1, temp_attr_triples_1, ent_ids_1_file, 0)
    
    # 生成图谱2的实体ID，确保不与图谱1重复
    generate_entity_ids(temp_rel_triples_2, temp_attr_triples_2, ent_ids_2_file, max_ent_id_1 + 1)
    
    # 生成关系ID映射文件
    rel_ids_1_file = os.path.join(dest_path, "rel_ids_1")
    rel_ids_2_file = os.path.join(dest_path, "rel_ids_2")
    max_rel_id_1 = generate_relation_ids(temp_rel_triples_1, rel_ids_1_file, 0)
    
    # 生成图谱2的关系ID，确保不与图谱1重复
    generate_relation_ids(temp_rel_triples_2, rel_ids_2_file, max_rel_id_1 + 1)
    
    # 加载ID映射
    ent_id_map_1 = load_id_map(ent_ids_1_file)
    ent_id_map_2 = load_id_map(ent_ids_2_file)
    rel_id_map_1 = load_id_map(rel_ids_1_file)
    rel_id_map_2 = load_id_map(rel_ids_2_file)
    
    # 使用ID替换原始文件中的实体和关系
    rel_triples_1_dst = os.path.join(dest_path, "rel_triples_1")
    rel_triples_2_dst = os.path.join(dest_path, "rel_triples_2")
    attr_triples_1_dst = os.path.join(dest_path, "attr_triples_1")
    attr_triples_2_dst = os.path.join(dest_path, "attr_triples_2")
    seed_aligns_dst = os.path.join(dest_path, "seed_aligns")
    test_aligns_dst = os.path.join(dest_path, "test_aligns")
    
    # 处理关系三元组
    if os.path.exists(temp_rel_triples_1):
        replace_with_ids(temp_rel_triples_1, rel_triples_1_dst, ent_id_map_1, rel_id_map_1)
    if os.path.exists(temp_rel_triples_2):
        replace_with_ids(temp_rel_triples_2, rel_triples_2_dst, ent_id_map_2, rel_id_map_2)
    
    # 处理属性三元组
    if os.path.exists(temp_attr_triples_1):
        replace_attr_with_ids(temp_attr_triples_1, attr_triples_1_dst, ent_id_map_1)
    if os.path.exists(temp_attr_triples_2):
        replace_attr_with_ids(temp_attr_triples_2, attr_triples_2_dst, ent_id_map_2)
    
    # 处理对齐文件
    if os.path.exists(temp_seed_aligns):
        replace_align_with_ids(temp_seed_aligns, seed_aligns_dst, ent_id_map_1, ent_id_map_2)
    if os.path.exists(temp_test_aligns):
        replace_align_with_ids(temp_test_aligns, test_aligns_dst, ent_id_map_1, ent_id_map_2)
    
    # 清理临时文件
    temp_files = [temp_rel_triples_1, temp_rel_triples_2, temp_attr_triples_1, temp_attr_triples_2, 
                  temp_seed_aligns, temp_test_aligns]
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)

# 生成实体ID映射文件，并返回最大ID
def generate_entity_ids(rel_triples_file, attr_triples_file, output_file, start_id=0):
    entities = set()
    
    # 从关系三元组中提取实体
    if os.path.exists(rel_triples_file):
        with open(rel_triples_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    entities.add(parts[0])
                    entities.add(parts[2])
    
    # 从属性三元组中提取实体
    if os.path.exists(attr_triples_file):
        with open(attr_triples_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    entities.add(parts[0])
    
    # 写入实体ID映射文件
    sorted_entities = sorted(entities)
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, entity in enumerate(sorted_entities, start=start_id):
            f.write(f"{i}\t{entity}\n")
    
    # 返回最大ID
    if sorted_entities:
        return start_id + len(sorted_entities) - 1
    return start_id - 1

# 生成关系ID映射文件，并返回最大ID
def generate_relation_ids(rel_triples_file, output_file, start_id=0):
    relations = set()
    
    # 从关系三元组中提取关系
    if os.path.exists(rel_triples_file):
        with open(rel_triples_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    relations.add(parts[1])
    
    # 写入关系ID映射文件
    sorted_relations = sorted(relations)
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, relation in enumerate(sorted_relations, start=start_id):
            f.write(f"{i}\t{relation}\n")
    
    # 返回最大ID
    if sorted_relations:
        return start_id + len(sorted_relations) - 1
    return start_id - 1

# 加载ID映射文件
def load_id_map(id_file):
    id_map = {}
    if os.path.exists(id_file):
        with open(id_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    id_map[parts[1]] = parts[0]
    return id_map

# 替换关系三元组中的实体和关系为ID
def replace_with_ids(input_file, output_file, ent_id_map, rel_id_map):
    with open(input_file, 'r', encoding='utf-8') as f_in:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    # 替换实体和关系为ID
                    subj_id = ent_id_map.get(parts[0], parts[0])
                    rel_id = rel_id_map.get(parts[1], parts[1])
                    obj_id = ent_id_map.get(parts[2], parts[2])
                    f_out.write(f"{subj_id}\t{rel_id}\t{obj_id}\n")
                else:
                    f_out.write(line)

# 替换属性三元组中的实体为ID
def replace_attr_with_ids(input_file, output_file, ent_id_map):
    with open(input_file, 'r', encoding='utf-8') as f_in:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    # 只替换实体为ID，保留属性值
                    ent_id = ent_id_map.get(parts[0], parts[0])
                    attr_value = '\t'.join(parts[1:])
                    f_out.write(f"{ent_id}\t{attr_value}\n")
                else:
                    f_out.write(line)

# 替换对齐文件中的实体为ID
def replace_align_with_ids(input_file, output_file, ent_id_map_1, ent_id_map_2):
    with open(input_file, 'r', encoding='utf-8') as f_in:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    # 替换两个图谱中的实体为ID
                    ent1_id = ent_id_map_1.get(parts[0], parts[0])
                    ent2_id = ent_id_map_2.get(parts[1], parts[1])
                    f_out.write(f"{ent1_id}\t{ent2_id}\n")
                else:
                    f_out.write(line)

# 主函数
def main():
    # 获取源目录中的所有数据集文件夹
    if not os.path.exists(source_dir):
        print(f"源目录 {source_dir} 不存在，请检查路径是否正确。")
        return
    
    datasets = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    for dataset in datasets:
        source_path = os.path.join(source_dir, dataset)
        dest_path = os.path.join(dest_dir, dataset)
        process_dataset(dataset, source_path, dest_path)
    
    print(f"数据集处理完成！结果保存在 {dest_dir} 目录中。")

if __name__ == "__main__":
    main()