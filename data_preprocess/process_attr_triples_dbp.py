import os

# 定义输入和输出文件路径
BASE_DIR = "./data/EA/dbp15k_zh_en"
INPUT_FILES = [
    os.path.join(BASE_DIR, "attr_triples_1"),
    os.path.join(BASE_DIR, "attr_triples_2")
]
OUTPUT_FILES = [
    os.path.join(BASE_DIR, "processed_attr_triples_1"),
    os.path.join(BASE_DIR, "processed_attr_triples_2")
]
ENT_IDS_FILES = [
    os.path.join(BASE_DIR, "ent_ids_1"),
    os.path.join(BASE_DIR, "ent_ids_2")
]

def load_entity_mappings():
    """加载实体ID到名称的映射"""
    entity_mappings = {}
    
    for ent_ids_file in ENT_IDS_FILES:
        print(f"加载实体映射文件: {ent_ids_file}")
        if not os.path.exists(ent_ids_file):
            print(f"警告: 实体映射文件不存在: {ent_ids_file}")
            continue
        
        with open(ent_ids_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) >= 2:
                    entity_id = parts[0].strip()
                    entity_url = parts[1].strip()
                    # 从URL中提取实体名称（URL最后一部分）
                    entity_name = entity_url#.split('/')[-1]
                    entity_mappings[entity_name] = entity_id
    
    print(f"已加载 {len(entity_mappings)} 个实体映射")
    return entity_mappings

def process_attribute_triples(input_file, output_file, entity_mappings):
    print(f"处理文件: {input_file}")
    
    # 统计处理的行数
    line_count = 0
    found_entity_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as in_file, \
         open(output_file, 'w', encoding='utf-8') as out_file:
        
        for line in in_file:
            line = line.strip()
            if not line:
                continue
            
            # 分割行内容
            parts = line.split(' ')
            if len(parts) >= 3:
                # 提取实体ID、属性名和属性值
                entity_id = entity_mappings[parts[0].strip()[1:-1]]
                attr_name = parts[1].strip()[1:-1].split('/')[-1]
                attr_value = ' '.join(parts[2:]).strip()  # 处理属性值中可能包含的\t
                attr_value = attr_value[1:]
                attr_value = attr_value.split('\"@')[0]
                attr_value = attr_value.split('\"^^')[0]

                # 按照指定格式写入输出文件
                formatted_line = f"{entity_id}\t{attr_name}\t{attr_value}"
                out_file.write(formatted_line + '\n')
                line_count += 1
                
                # 统计找到映射的实体数量
                if entity_id in entity_mappings:
                    found_entity_count += 1
            else:
                print(f"警告: 跳过格式不正确的行: {line}")
    
    print(f"已处理 {line_count} 行，其中 {found_entity_count} 个实体有对应的名称映射")
    print(f"结果保存到: {output_file}")
    return entity_mappings

# 处理两个文件
def main():
    print("开始处理属性三元组文件...")
    
    # 首先加载实体ID到名称的映射
    entity_mappings = load_entity_mappings()
    
    # 然后处理属性三元组文件
    for i in range(len(INPUT_FILES)):
        if os.path.exists(INPUT_FILES[i]):
            process_attribute_triples(INPUT_FILES[i], OUTPUT_FILES[i], entity_mappings)
        else:
            print(f"错误: 文件不存在: {INPUT_FILES[i]}")
    
    # 打印实体映射示例
    print("\n实体ID到名称映射示例:")
    sample_mappings = list(entity_mappings.items())[:5]  # 显示前5个映射
    for entity_id, entity_name in sample_mappings:
        print(f"实体ID: {entity_id} -> 实体名称: {entity_name}")
    
    print("处理完成!")

if __name__ == "__main__":
    main()