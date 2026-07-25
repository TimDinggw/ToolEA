import os

# 定义输入和输出文件路径
BASE_DIR = "OpenEA_dataset_v2.0_processed\EN_DE_15K_V1"
BASE_DIR = "SRPRS_processed\en_de_15k_V1"
BASE_DIR = "SRPRS_processed\en_fr_15k_V1"

INPUT_FILES = [
    os.path.join(BASE_DIR, "attr_triples_1"),
    os.path.join(BASE_DIR, "attr_triples_2")
]
OUTPUT_FILES = [
    os.path.join(BASE_DIR, "processed_attr_triples_1"),
    os.path.join(BASE_DIR, "processed_attr_triples_2")
]


def process_attribute_triples(input_file, output_file):
    print(f"处理文件: {input_file}")
    
    # 统计处理的行数
    line_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as in_file, \
         open(output_file, 'w', encoding='utf-8') as out_file:
        
        for line in in_file:
            line = line.strip()
            if not line:
                continue
            
            # 分割行内容
            parts = line.split('\t')
            if len(parts) >= 3:
                # 提取实体ID、属性名和属性值
                entity_id = parts[0].strip()
                attr_name = parts[1].strip().split('/')[-1]
                attr_value = parts[2].strip()  # 处理属性值中可能包含的\t
                if '\"^^' in attr_value:
                    attr_value = attr_value[1:]
                    attr_value = attr_value.split('\"^^')[0]
                if '\"@' in attr_value:
                    attr_value = attr_value[1:]
                    attr_value = attr_value.split('\"@')[0]

                # print(entity_id, attr_name, attr_value)
                # if entity_id == "6333":
                #     assert 0
                # if not entity_id.isdigit():
                #     print(entity_id)
                #     entity_id, attr_value = attr_value, entity_id

                # 按照指定格式写入输出文件
                formatted_line = f"{entity_id}\t{attr_name}\t{attr_value}"
                out_file.write(formatted_line + '\n')
                line_count += 1
                
            else:
                print(f"警告: 跳过格式不正确的行: {line}")
    
    print(f"已处理 {line_count} 行")
    print(f"结果保存到: {output_file}")

# 处理两个文件
def main():
    print("开始处理属性三元组文件...")

    
    # 然后处理属性三元组文件
    for i in range(len(INPUT_FILES)):
        if os.path.exists(INPUT_FILES[i]):
            process_attribute_triples(INPUT_FILES[i], OUTPUT_FILES[i])
        else:
            print(f"错误: 文件不存在: {INPUT_FILES[i]}")
    
    print("处理完成!")

if __name__ == "__main__":
    main()