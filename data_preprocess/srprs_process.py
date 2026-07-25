import os
import shutil

# 定义数据集目录和输出目录
SRPRS_DIR = 'c:\\Users\\29941\\Desktop\\EADatasets\\SRPRS'
OUTPUT_DIR = 'c:\\Users\\29941\\Desktop\\EADatasets\\SRPRS_processed'

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 获取所有数据集子目录
dataset_dirs = [d for d in os.listdir(SRPRS_DIR) if os.path.isdir(os.path.join(SRPRS_DIR, d))]
dataset_dirs = ['en_de_15k_V1']

# 标准文件列表（与dbp15k_processed保持一致）
standard_files = {
    'attr_triples_1', 'attr_triples_2',
    'ent_ids_1', 'ent_ids_2',
    'rel_ids_1', 'rel_ids_2',
    'rel_triples_1', 'rel_triples_2',
    'seed_aligns', 'test_aligns'
}

def load_id_mapping(file_path):
    """加载ID映射文件"""
    mapping = {}
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    mapping[parts[1]] = parts[0]  # {name: id}
    return mapping

def process_triples(input_file, ent_map, rel_map, output_file):
    """处理三元组文件，将实体和关系名称替换为ID"""
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                head, rel, tail = parts
                # 替换实体和关系为ID
                head_id = ent_map.get(head, head)
                rel_id = rel_map.get(rel, rel)
                tail_id = ent_map.get(tail, tail)
                f_out.write(f"{head_id}\t{rel_id}\t{tail_id}\n")

def process_attr_triples(input_file, ent_map, attr_map, output_file):
    """处理属性三元组文件"""
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                ent, attr, value = parts
                # 只替换实体和属性，值保持不变
                ent_id = ent_map.get(ent, ent)
                attr_id = attr_map.get(attr, attr)
                f_out.write(f"{ent_id}\t{attr_id}\t{value}\n")

def generate_aligns(sup_file, ref_file, seed_file, test_file):
    """生成seed_aligns和test_aligns文件"""
    # 读取sup_ent_ids作为seed_aligns
    seed_pairs = []
    if os.path.exists(sup_file):
        with open(sup_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    seed_pairs.append(f"{parts[0]}\t{parts[1]}\n")
    
    # 读取ref_ent_ids作为test_aligns
    test_pairs = []
    if os.path.exists(ref_file):
        with open(ref_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    test_pairs.append(f"{parts[0]}\t{parts[1]}\n")
    
    # 写入seed_aligns文件
    with open(seed_file, 'w', encoding='utf-8') as f:
        f.writelines(seed_pairs)
    
    # 写入test_aligns文件
    with open(test_file, 'w', encoding='utf-8') as f:
        f.writelines(test_pairs)

# 处理每个数据集
for dataset_dir in dataset_dirs:
    dataset_path = os.path.join(SRPRS_DIR, dataset_dir)
    output_path = os.path.join(OUTPUT_DIR, dataset_dir)
    
    # 确保输出子目录存在
    os.makedirs(output_path, exist_ok=True)
    
    # 优先处理mapping/0_3目录
    mapping_0_3_path = os.path.join(dataset_path, 'mapping', '0_3')
    if os.path.exists(mapping_0_3_path):
        # 加载ID映射
        ent_ids_1 = load_id_mapping(os.path.join(mapping_0_3_path, 'ent_ids_1'))
        ent_ids_2 = load_id_mapping(os.path.join(mapping_0_3_path, 'ent_ids_2'))
        rel_ids_1 = load_id_mapping(os.path.join(mapping_0_3_path, 'rel_ids_1'))
        rel_ids_2 = load_id_mapping(os.path.join(mapping_0_3_path, 'rel_ids_2'))
        
        # 合并实体映射
        ent_map = {**ent_ids_1, **ent_ids_2}
        # 合并关系映射
        rel_map = {**rel_ids_1, **rel_ids_2}
        
        # 优先使用根目录的triples文件（URI格式），配合mapping/0_3的ID映射来生成正确ID范围的三元组
        root_triples_1 = os.path.join(dataset_path, 'triples_1')
        root_triples_2 = os.path.join(dataset_path, 'triples_2')
        
        if os.path.exists(root_triples_1):
            # 根目录的triples是URI格式，需要用mapping/0_3的ID映射来转换
            process_triples(root_triples_1, ent_map, rel_map, 
                          os.path.join(output_path, 'rel_triples_1'))
        
        if os.path.exists(root_triples_2):
            process_triples(root_triples_2, ent_map, rel_map, 
                          os.path.join(output_path, 'rel_triples_2'))
        
        # 复制ID文件
        for id_file in ['ent_ids_1', 'ent_ids_2', 'rel_ids_1', 'rel_ids_2']:
            src = os.path.join(mapping_0_3_path, id_file)
            dst = os.path.join(output_path, id_file)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        
        # 生成对齐文件
        sup_file = os.path.join(mapping_0_3_path, 'sup_ent_ids')
        ref_file = os.path.join(mapping_0_3_path, 'ref_ent_ids')
        seed_file = os.path.join(output_path, 'seed_aligns')
        test_file = os.path.join(output_path, 'test_aligns')
        generate_aligns(sup_file, ref_file, seed_file, test_file)
    
    # 处理根目录的属性文件
    for attr_file in ['attr_triples_1', 'attr_triples_2']:
        src = os.path.join(dataset_path, attr_file)
        dst = os.path.join(output_path, attr_file)
        if os.path.exists(src):
            # 加载属性ID映射（如果存在）
            attr_ids_1 = load_id_mapping(os.path.join(dataset_path, 'mapping', '0_3', 'attr_ids_1')) if os.path.exists(os.path.join(dataset_path, 'mapping', '0_3', 'attr_ids_1')) else {}
            attr_ids_2 = load_id_mapping(os.path.join(dataset_path, 'mapping', '0_3', 'attr_ids_2')) if os.path.exists(os.path.join(dataset_path, 'mapping', '0_3', 'attr_ids_2')) else {}
            attr_map = {**attr_ids_1, **attr_ids_2}
            
            # 获取实体映射
            ent_map = {}
            ent_map_file1 = os.path.join(output_path, 'ent_ids_1')
            ent_map_file2 = os.path.join(output_path, 'ent_ids_2')
            if os.path.exists(ent_map_file1):
                ent_map.update(load_id_mapping(ent_map_file1))
            if os.path.exists(ent_map_file2):
                ent_map.update(load_id_mapping(ent_map_file2))
            
            # 处理属性三元组
            process_attr_triples(src, ent_map, attr_map, dst)
    
    # 清理多余的文件，只保留标准文件
    for file in os.listdir(output_path):
        file_path = os.path.join(output_path, file)
        if os.path.isfile(file_path) and file not in standard_files:
            os.remove(file_path)
    
    print(f"处理完成: {dataset_dir}")

print("所有数据集处理完成！")