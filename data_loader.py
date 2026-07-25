import os
import csv
from typing import List, Dict, Any, Optional, Tuple


class UnifiedEntity:
    def __init__(self, entity_id: str, source_name: str):
        self.id = entity_id  # 实体在原始数据源中的ID
        self.source = source_name  # 例如: "KG1", "KG2", "TableA", "TableB"
        self.name: Optional[str] = None  # 实体名称(来自EA 或者 EM 的 name 列)，去掉前缀
        self.origin_name: Optional[str] = None  # 实体原名
        self.attributes: Optional[Dict[str, Any]] = {}  # 属性字典 {attr_name: value}
        self.relations: Optional[Dict[str, List[str]]] = {}  # 关系字典 {rel_id: [target_entity_id, ...]}
        self.info_density_score: Optional[float] = None  # 信息密度，关系个数+属性个数
        self.ontology: Optional[List[str]] = None  # 由 LLM 从ontology_list中选最可能的三个


class AlignmentPair:
    def __init__(self, id_a: str, id_b: str, label: int = 1, confidence: float = 1.0):
        self.id_a = id_a
        self.id_b = id_b
        self.label = label
        self.confidence = confidence


class DataLoader:
    def __init__(self, data_path: str, task_type: str):
        """初始化数据加载器
        
        Args:
            data_path: 数据路径
            task_type: 任务类型，"EA" 或 "EM"
        """
        self.data_path = data_path
        self.task_type = task_type
        self.relation_name_mappings = {}
        self.entity_name_mappings = {}
        
    def load_data(self) -> Tuple[List[UnifiedEntity], List[UnifiedEntity], List[AlignmentPair], List[AlignmentPair]]:
        """加载数据（根据任务类型调用相应的加载方法）
        
        Returns:
            Tuple[source_A_entities, source_B_entities, seed_alignments, test_alignments]
        """
        if self.task_type == "EA":
            return self.load_ea_data()
        elif self.task_type == "EM":
            return self.load_em_data()
        else:
            raise ValueError(f"不支持的任务类型: {self.task_type}")
    
    def load_ea_data(self) -> Tuple[List[UnifiedEntity], List[UnifiedEntity], List[AlignmentPair], List[AlignmentPair], List[AlignmentPair]]:
        """加载EA数据并转换为统一格式
        
        Returns:
            Tuple[source_A_entities, source_B_entities, seed_alignments, test_alignments]
        """
        source_A_entities = []
        source_B_entities = []
        seed_alignments = []
        test_alignments = []
        
        # 加载关系ID到名称的映射
        self._load_relation_name_mappings(os.path.join(self.data_path, 'rel_ids_1'), 'A')
        self._load_relation_name_mappings(os.path.join(self.data_path, 'rel_ids_2'), 'B')
        
        # 加载实体ID到名称的映射
        self._load_entity_name_mappings(os.path.join(self.data_path, 'ent_ids_1'), 'A')
        self._load_entity_name_mappings(os.path.join(self.data_path, 'ent_ids_2'), 'B')
        
        # 加载实体信息
        source_A_entities = self._load_ea_entities(os.path.join(self.data_path, 'ent_ids_1'), 'A')
        source_B_entities = self._load_ea_entities(os.path.join(self.data_path, 'ent_ids_2'), 'B')
        
        # 加载关系信息
        self._load_ea_relations(source_A_entities, os.path.join(self.data_path, 'rel_triples_1'), 'A')
        self._load_ea_relations(source_B_entities, os.path.join(self.data_path, 'rel_triples_2'), 'B')
        
        # 加载属性信息
        self._load_ea_attributes(source_A_entities, os.path.join(self.data_path, 'attr_triples_1'), 'A')
        self._load_ea_attributes(source_B_entities, os.path.join(self.data_path, 'attr_triples_2'), 'B')
        
        # 加载对齐信息
        seed_alignments = self._load_ea_alignments(os.path.join(self.data_path, 'seed_aligns'))
        valid_alignments = self._load_ea_alignments(os.path.join(self.data_path, 'valid_aligns'))
        test_alignments = self._load_ea_alignments(os.path.join(self.data_path, 'test_aligns'))
        
        return source_A_entities, source_B_entities, seed_alignments, valid_alignments, test_alignments
    
    def _load_relation_name_mappings(self, rel_ids_file: str, source_name: str):
        """加载关系ID到名称的映射"""
        with open(rel_ids_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    rel_id, rel_url = parts
                    # 提取URL中的关系名称部分
                    rel_name = rel_url.split('/')[-1]
                    if source_name not in self.relation_name_mappings:
                        self.relation_name_mappings[source_name] = {}
                    self.relation_name_mappings[source_name][rel_id] = rel_name
    
    def _load_entity_name_mappings(self, ent_ids_file: str, source_name: str):
        """加载实体ID到名称的映射"""
        with open(ent_ids_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    ent_id, ent_url = parts
                    # 提取URL中的实体名称部分
                    ent_name = ent_url.split('/')[-1]
                    if source_name not in self.entity_name_mappings:
                        self.entity_name_mappings[source_name] = {}
                    self.entity_name_mappings[source_name][ent_id] = ent_name
    
    def _load_ea_relations(self, entities: List[UnifiedEntity], rel_triples_file: str, source_name: str):
        """加载EA关系信息，使用关系名称而非ID"""
        entity_dict = {entity.id: entity for entity in entities}
        with open(rel_triples_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    head_id, rel_id, tail_id = parts
                    if head_id in entity_dict:
                        # 获取关系名称
                        rel_name = self.relation_name_mappings.get(source_name, {}).get(rel_id, rel_id)
                        # 获取目标实体名称
                        tail_name = self.entity_name_mappings.get(source_name, {}).get(tail_id, tail_id)
                        
                        if rel_name not in entity_dict[head_id].relations:
                            entity_dict[head_id].relations[rel_name] = []
                        entity_dict[head_id].relations[rel_name].append(tail_name)
    
    def _load_ea_attributes(self, entities: List[UnifiedEntity], attr_triples_file: str, source_name: str):
        """加载EA属性信息，处理属性名称"""
        try:
            entity_dict = {entity.id: entity for entity in entities}
            with open(attr_triples_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        head_id, attr_name, attr_value = parts[0], parts[1], '\t'.join(parts[2:])
                        if head_id in entity_dict:
                            # 尝试提取属性名称中的最后一部分
                            if '/' in attr_name:
                                clean_attr_name = attr_name.split('/')[-1]
                            else:
                                clean_attr_name = attr_name
                            entity_dict[head_id].attributes[clean_attr_name] = attr_value
        except Exception as e:
            print("加载属性信息失败或无属性信息: ", e)
    
    def _load_ea_alignments(self, align_file: str) -> List[AlignmentPair]:
        """加载EA对齐信息"""
        alignments = []
        if os.path.exists(align_file):
            with open(align_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        id_a, id_b = parts[0], parts[1]
                        alignments.append(AlignmentPair(id_a, id_b))
        return alignments
    
    def load_em_data(self) -> Tuple[List[UnifiedEntity], List[UnifiedEntity], List[AlignmentPair], List[AlignmentPair], List[AlignmentPair]]:
        """加载EM数据并转换为统一格式
        
        Returns:
            Tuple[source_A_entities, source_B_entities, seed_alignments, test_alignments]
        """
        source_A_entities = self._load_em_entities(os.path.join(self.data_path, 'tableA.csv'), 'A')
        source_B_entities = self._load_em_entities(os.path.join(self.data_path, 'tableB.csv'), 'B')
        
        # 加载训练对齐（作为seed_alignments）
        seed_alignments = self._load_em_alignments(os.path.join(self.data_path, 'train.csv'))
        valid_alignments = self._load_em_alignments(os.path.join(self.data_path, 'valid.csv'))
        test_alignments = self._load_em_alignments(os.path.join(self.data_path, 'test.csv'))
        
        return source_A_entities, source_B_entities, seed_alignments, valid_alignments, test_alignments
    
    def _load_ea_entities(self, ent_ids_file: str, source_name: str) -> List[UnifiedEntity]:
        """加载EA实体信息
        
        Args:
            ent_ids_file: 实体ID文件路径
            source_name: 源名称（如'KG1', 'KG2'）
            
        Returns:
            实体对象列表
        """
        entities = []
        with open(ent_ids_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    entity_id, entity_url = parts
                    entity = UnifiedEntity(entity_id, source_name)
                    # 提取URL中的实体名称部分
                    entity.origin_name = entity_url
                    if '/' in entity_url:
                        entity.name = entity_url.split('/')[-1]
                    else:
                        entity.name = entity_url
                    entities.append(entity)
        return entities
    
    def _load_em_entities(self, table_file: str, source_name: str) -> List[UnifiedEntity]:
        """加载EM实体信息"""
        entities = []
        with open(table_file, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 假设第一列是ID
                entity_id = row[list(row.keys())[0]]
                entity = UnifiedEntity(entity_id, source_name)
                
                # 设置名称（优先使用name列，如果没有则使用ID）
                if 'name' in row:
                    entity.name = row['name']
                elif 'title' in row:
                    entity.name = row['title']
                else:
                    entity.name = entity_id
                
                # 设置所有列为属性
                for key, value in row.items():
                    if key != list(row.keys())[0]:  # 跳过ID列
                        entity.attributes[key] = value
                
                entities.append(entity)
        return entities
    
    def _load_em_alignments(self, align_file: str) -> List[AlignmentPair]:
        """加载EM对齐信息"""
        alignments = []
        if os.path.exists(align_file):
            with open(align_file, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'ltable_id' in row and 'rtable_id' in row:
                        id_a = row['ltable_id']
                        id_b = row['rtable_id']
                        # 如果有label列，则使用它，否则默认为1
                        label = int(row.get('label', 1))
                        alignments.append(AlignmentPair(id_a, id_b, label))
        return alignments


if __name__ == "__main__":
    # 示例用法
    pass