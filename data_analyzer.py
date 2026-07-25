from typing import List, Optional, Dict, Any, Set
import os
import json
import subprocess
import openai
import re
import numpy as np
from openai import OpenAI  # 添加这一行导入OpenAI类
from data_loader import UnifiedEntity
from LLMConfig import LLMConfig  # 导入统一配置
import random

# 自定义JSON序列化器，处理float32类型
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)  # 转换为Python float
        elif isinstance(obj, np.integer):
            return int(obj)    # 转换为Python int
        elif isinstance(obj, np.floating):
            return float(obj)  # 转换为Python float
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # 转换为Python list
        return super().default(obj)

class DataModalReport:
    def __init__(self):
        self.has_names: bool = False
        self.has_relations: bool = False  # 区分是EA任务还是EM任务的关键
        self.has_relation_names: bool = False
        self.has_attributes: bool = False
        self.has_attribute_names: bool = False
        self.has_images: bool = False  # 针对 EA, 检查是否存在文件./data/EA/data_dir/image_pkls
        self.is_ea_task: bool = False  # has_relations为True时为True
    
    def to_dict(self) -> dict:
        """将报告转换为字典格式以便保存"""
        return {
            'has_names': self.has_names,
            'has_relations': self.has_relations,
            'has_relation_names': self.has_relation_names,
            'has_attributes': self.has_attributes,
            'has_attribute_names': self.has_attribute_names,
            'has_images': self.has_images,
            'is_ea_task': self.is_ea_task
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DataModalReport':
        """从字典创建报告对象"""
        report = cls()
        report.has_names = data.get('has_names', False)
        report.has_relations = data.get('has_relations', False)
        report.has_relation_names = data.get('has_relation_names', False)
        report.has_attributes = data.get('has_attributes', False)
        report.has_attribute_names = data.get('has_attribute_names', False)
        report.has_images = data.get('has_images', False)
        report.is_ea_task = data.get('is_ea_task', False)
        return report

class SchemaAlignmentResult:
    def __init__(self):
        self.relation_alignments: Dict[str, str] = {}  # {source_B_relation: source_A_relation}
        self.attribute_alignments: Dict[str, str] = {}  # {source_B_attribute: source_A_attribute}
    
    def to_dict(self) -> dict:
        """将模式对齐结果转换为字典格式以便保存"""
        return {
            'relation_alignments': self.relation_alignments,
            'attribute_alignments': self.attribute_alignments
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SchemaAlignmentResult':
        """从字典创建模式对齐结果对象"""
        result = cls()
        result.relation_alignments = data.get('relation_alignments', {})
        result.attribute_alignments = data.get('attribute_alignments', {})
        return result

class DataAnalyzer:
    def __init__(self, ontology_list_file: str = './ontology_list.txt', schema_alignment_round : int = 5):
        """初始化数据分析器schema_alignment_round
        
        Args:
            ontology_list_file: ontology列表文件路径
        """
        self.ontology_list = self._load_ontology_list(ontology_list_file)
        self.schema_alignment_round = schema_alignment_round
        # 配置OpenAI API以使用本地Qwen3-8B模型
        self._configure_openai()
    
    def _load_ontology_list(self, file_path: str) -> List[str]:
        """加载ontology列表"""
        ontology_list = []
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    ontology = line.strip()
                    if ontology:
                        ontology_list.append(ontology)
        return ontology_list
    
    def _normalize_string(self, s: str) -> str:
        """规范化字符串：移除非字母数字字符，转换为小写
        
        Args:
            s: 输入字符串
            
        Returns:
            规范化后的字符串
        """
        # 移除非字母数字字符（空格、下划线等）
        normalized = re.sub(r'[^a-zA-Z0-9]', '', s)
        # 转换为小写
        return normalized.lower()
    
    def _configure_openai(self):
        """配置OpenAI（使用统一配置）"""
        config = LLMConfig.get_openai_client_config()
        openai.api_key = config["api_key"]
        openai.api_base = config["base_url"]
    
    def _is_meaningful_name(self, name: str) -> bool:
        """使用LLM判断名称是否有意义"""
        if not name or len(name.strip()) == 0:
            return False
        
        try:
            # 使用统一配置
            client_config = LLMConfig.get_openai_client_config("meaningful_name_check")
            meaningful_name_check_config = LLMConfig.get_completion_config("meaningful_name_check")
            
            client = OpenAI(**client_config)
            
            prompt = f"""Determine whether the following string is a meaningful entity, relation, or attribute name rather than a meaningless ID or code (like P1071 or Q7374).

String: "{name}"

Return only "YES" or "NO" without any additional text or explanation."""
            
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an entity name recognition expert skilled at determining whether strings are meaningful."},
                    {"role": "user", "content": prompt}
                ],
                **meaningful_name_check_config
            )
            result = response.choices[0].message.content.strip() 
            print(f"LLM判断名称'{name}'是否有意义的结果: {result}")
            return result.lower() == "yes"
            
        except Exception as e:
            print(f"LLM调用失败: {e}")
            assert 0
            # return self._is_meaningful_name_fallback(name)
    

    def analyze_modal(self, source_A_entities: List[UnifiedEntity], source_B_entities: List[UnifiedEntity], data_path: str) -> DataModalReport:
        """进行模态分析
        
        Args:
            source_A_entities: 源A的实体列表
            source_B_entities: 源B的实体列表
            data_path: 数据路径，用于检查图片文件
            
        Returns:
            DataModalReport对象
        """
        report = DataModalReport()
        
        # 检查是否有名称（使用LLM区分真实名称和ID）
        # 需要两个图谱都有有意义的名称才设置为True，随机抽取50个样本进行判断
        source_A_names = []
        source_B_names = []
        
        # 收集源A和源B的名称
        for entity in source_A_entities:
            if entity.name and entity.name.strip():
                source_A_names.append(entity.name)
        
        for entity in source_B_entities:
            if entity.name and entity.name.strip():
                source_B_names.append(entity.name)
        
        # 随机采样最多20个样本
        sample_size = min(20, len(source_A_names))
        if sample_size > 0:
            sampled_A_names = random.sample(source_A_names, sample_size)
            # 检查源A是否有有意义的名称
            has_A_meaningful_names = any(self._is_meaningful_name(name) for name in sampled_A_names)
        else:
            has_A_meaningful_names = False
        
        sample_size = min(20, len(source_B_names))
        if sample_size > 0:
            sampled_B_names = random.sample(source_B_names, sample_size)
            # 检查源B是否有有意义的名称
            has_B_meaningful_names = any(self._is_meaningful_name(name) for name in sampled_B_names)
        else:
            has_B_meaningful_names = False
        
        # print(f"源A是否有有意义的名称: {has_A_meaningful_names}, 源B是否有有意义的名称: {has_B_meaningful_names}")
        # 只有两个图谱都有有意义的名称时，才设置为True
        report.has_names = has_A_meaningful_names and has_B_meaningful_names
        
        # 检查是否有关系
        has_A_relations = any(entity.relations and len(entity.relations) > 0 for entity in source_A_entities)
        has_B_relations = any(entity.relations and len(entity.relations) > 0 for entity in source_B_entities)
        report.has_relations = has_A_relations or has_B_relations
        
        # 检查关系是否有名称（使用LLM区分真实名称和ID）
        # 需要两个图谱都有有意义的关系名称才设置为True，随机抽取50个样本进行判断
        source_A_rel_names = set()
        source_B_rel_names = set()
        
        # 收集源A和源B的关系名称
        for entity in source_A_entities:
            if entity.relations and isinstance(entity.relations, dict):
                for rel_name in entity.relations.keys():
                    if rel_name and rel_name.strip():
                        source_A_rel_names.add(rel_name.strip())
        
        for entity in source_B_entities:
            if entity.relations and isinstance(entity.relations, dict):
                for rel_name in entity.relations.keys():
                    if rel_name and rel_name.strip():
                        source_B_rel_names.add(rel_name.strip())
        
        # 随机采样最多20个样本
        sample_A_rels = list(source_A_rel_names)
        sample_size = min(20, len(sample_A_rels))
        if sample_size > 0:
            sampled_A_rels = random.sample(sample_A_rels, sample_size)
            # 检查源A是否有有意义的关系名称
            has_A_meaningful_rel_names = any(self._is_meaningful_name(rel_name) for rel_name in sampled_A_rels)
        else:
            has_A_meaningful_rel_names = False
        
        sample_B_rels = list(source_B_rel_names)
        sample_size = min(20, len(sample_B_rels))
        if sample_size > 0:
            sampled_B_rels = random.sample(sample_B_rels, sample_size)
            # 检查源B是否有有意义的关系名称
            has_B_meaningful_rel_names = any(self._is_meaningful_name(rel_name) for rel_name in sampled_B_rels)
        else:
            has_B_meaningful_rel_names = False
        
        # 只有两个图谱都有有意义的关系名称时，才设置为True
        report.has_relation_names = has_A_meaningful_rel_names and has_B_meaningful_rel_names
        
        # 检查是否有属性
        has_A_attrs = any(entity.attributes and len(entity.attributes) > 0 for entity in source_A_entities)
        has_B_attrs = any(entity.attributes and len(entity.attributes) > 0 for entity in source_B_entities)
        report.has_attributes = has_A_attrs or has_B_attrs
        
        # 检查属性是否有名称（使用LLM区分真实名称和ID）
        # 需要两个图谱都有有意义的属性名称才设置为True，随机抽取50个样本进行判断
        source_A_attr_names = set()
        source_B_attr_names = set()
        
        # 收集源A和源B的属性名称
        for entity in source_A_entities:
            if entity.attributes and isinstance(entity.attributes, dict):
                for attr_name in entity.attributes.keys():
                    if attr_name and attr_name.strip():
                        source_A_attr_names.add(attr_name.strip())
        
        for entity in source_B_entities:
            if entity.attributes and isinstance(entity.attributes, dict):
                for attr_name in entity.attributes.keys():
                    if attr_name and attr_name.strip():
                        source_B_attr_names.add(attr_name.strip())
        
        # 随机采样最多20个样本
        sample_A_attrs = list(source_A_attr_names)
        sample_size = min(20, len(sample_A_attrs))
        if sample_size > 0:
            sampled_A_attrs = random.sample(sample_A_attrs, sample_size)
            # 检查源A是否有有意义的属性名称
            has_A_meaningful_attr_names = any(self._is_meaningful_name(attr_name) for attr_name in sampled_A_attrs)
        else:
            has_A_meaningful_attr_names = False
        
        sample_B_attrs = list(source_B_attr_names)
        sample_size = min(20, len(sample_B_attrs))
        if sample_size > 0:
            sampled_B_attrs = random.sample(sample_B_attrs, sample_size)
            # 检查源B是否有有意义的属性名称
            has_B_meaningful_attr_names = any(self._is_meaningful_name(attr_name) for attr_name in sampled_B_attrs)
        else:
            has_B_meaningful_attr_names = False
        
        # 只有两个图谱都有有意义的属性名称时，才设置为True
        report.has_attribute_names = has_A_meaningful_attr_names and has_B_meaningful_attr_names
        
        # 检查是否有图片（针对EA任务）
        if report.has_relations:  # 只有EA任务才需要检查图片
            image_pkls_dir = os.path.join(data_path, 'image_pkls')
            report.has_images = os.path.exists(image_pkls_dir) and len(os.listdir(image_pkls_dir)) > 0
        
        # 判断是否是EA任务
        report.is_ea_task = report.has_relations
        print("report", report.to_dict())
        return report
   
    
    def evaluate_info_density(self, source_A_entities: List[UnifiedEntity], source_B_entities: List[UnifiedEntity]):
        """评估实体的信息密度
        
        Args:
            source_A_entities: 源A的实体列表
            source_B_entities: 源B的实体列表
        """
        for entity in source_A_entities + source_B_entities:
            # 计算关系数量
            relation_count = len(entity.relations) if entity.relations else 0
            # 计算属性数量
            attribute_count = len(entity.attributes) if entity.attributes else 0
            # 计算信息密度分数
            entity.info_density_score = relation_count + attribute_count
    
    # def determine_ontology(self, source_A_entities: List[UnifiedEntity], source_B_entities: List[UnifiedEntity], schema_alignments: Optional[SchemaAlignmentResult] = None):
    #     """为实体确定本体类别
        
    #     Args:
    #         source_A_entities: 源A的实体列表
    #         source_B_entities: 源B的实体列表
    #         schema_alignments: 模式对齐结果，用于在判断前应用对齐
    #     """
    #     # 为每个实体进行本体判断
    #     for entity in source_A_entities + source_B_entities:
    #         if entity.name:
    #             # 准备实体的关系和属性，根据对齐结果进行调整
    #             aligned_relations = entity.relations.copy() if entity.relations else {}
    #             aligned_attributes = entity.attributes.copy() if entity.attributes else {}
                
    #             # 如果提供了对齐结果，对源B的实体应用对齐
    #             if schema_alignments and entity.source in ['KG2', 'B']:
    #                 # 应用关系对齐
    #                 if aligned_relations and isinstance(aligned_relations, dict):
    #                     temp_relations = {}
    #                     for rel_name, targets in aligned_relations.items():
    #                         if rel_name in schema_alignments.relation_alignments:
    #                             temp_relations[schema_alignments.relation_alignments[rel_name]] = targets
    #                         else:
    #                             temp_relations[rel_name] = targets
    #                     aligned_relations = temp_relations
                    
    #                 # 应用属性对齐
    #                 if aligned_attributes and isinstance(aligned_attributes, dict):
    #                     temp_attributes = {}
    #                     for attr_name, value in aligned_attributes.items():
    #                         if attr_name in schema_alignments.attribute_alignments:
    #                             temp_attributes[schema_alignments.attribute_alignments[attr_name]] = value
    #                         else:
    #                             temp_attributes[attr_name] = value
    #                     aligned_attributes = temp_attributes
                
    #             # 使用LLM进行本体判断，传入完整的实体信息
    #             top_ontologies = self._call_llm_for_ontology(entity.name, aligned_attributes, aligned_relations, self.ontology_list)
    #             entity.ontology = top_ontologies[:3]  # 取最可能的三个
    #         else:
    #             entity.ontology = ['Unknown']
    
    def determine_ontology(self, source_A_entities: List[UnifiedEntity], source_B_entities: List[UnifiedEntity], schema_alignments: Optional[SchemaAlignmentResult] = None, batch_size: int = 20):
        """为实体确定本体类别（批量版本）
        
        Args:
            source_A_entities: 源A的实体列表
            source_B_entities: 源B的实体列表
            schema_alignments: 模式对齐结果，用于在判断前应用对齐
            batch_size: 每批处理的实体数量
        """
        all_entities = source_A_entities + source_B_entities
        total_entities = len(all_entities)
        
        for i in range(0, total_entities, batch_size):
            batch_entities = all_entities[i:i + batch_size]
            batch_ontologies = self._call_llm_for_ontology_batch(batch_entities, schema_alignments)
            
            for entity, ontologies in zip(batch_entities, batch_ontologies):
                entity.ontology = ontologies
                print(entity.name, ontologies)
    
    def _call_llm_for_ontology_batch(self, entities: List[UnifiedEntity], schema_alignments: Optional[SchemaAlignmentResult] = None) -> List[List[str]]:
        """批量调用LLM进行本体判断
        
        Args:
            entities: 实体列表（批量）
            schema_alignments: 模式对齐结果
            
        Returns:
            每个实体对应的本体类别列表
        """
        prompt = self._build_batch_ontology_prompt(entities, schema_alignments)
        
        try:
            client_config = LLMConfig.get_openai_client_config("ontology_generation")
            ontology_generation_config = LLMConfig.get_completion_config("ontology_generation")
            client = OpenAI(**client_config)
            
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a knowledge graph entity classification expert."},
                    {"role": "user", "content": prompt}
                ],
                **ontology_generation_config
            )
            #print(prompt)
            result = response.choices[0].message.content.strip()
            #print(result)
            return self._parse_batch_ontology_result(result, len(entities))
            
        except Exception as e:
            print(f"调用LLM时出错: {e}")
            return [['Unknown', 'Other', 'Entity'][:3] for _ in entities]
    
    def _build_batch_ontology_prompt(self, entities: List[UnifiedEntity], schema_alignments: Optional[SchemaAlignmentResult] = None) -> str:
        """构建批量本体判断的prompt
        
        Args:
            entities: 实体列表
            schema_alignments: 模式对齐结果
            
        Returns:
            批量prompt字符串
        """
        entity_info_list = []
        
        for idx, entity in enumerate(entities):
            if not entity.name:
                entity_info_list.append(f"{idx + 1}. Entity Name: Unknown")
                continue
            
            aligned_relations = entity.relations.copy() if entity.relations else {}
            aligned_attributes = entity.attributes.copy() if entity.attributes else {}
            
            if schema_alignments and entity.source in ['KG2', 'B']:
                if aligned_relations and isinstance(aligned_relations, dict):
                    temp_relations = {}
                    for rel_name, targets in aligned_relations.items():
                        if rel_name in schema_alignments.relation_alignments:
                            temp_relations[schema_alignments.relation_alignments[rel_name]] = targets
                        else:
                            temp_relations[rel_name] = targets
                    aligned_relations = temp_relations
                
                if aligned_attributes and isinstance(aligned_attributes, dict):
                    temp_attributes = {}
                    for attr_name, value in aligned_attributes.items():
                        if attr_name in schema_alignments.attribute_alignments:
                            temp_attributes[schema_alignments.attribute_alignments[attr_name]] = value
                        else:
                            temp_attributes[attr_name] = value
                    aligned_attributes = temp_attributes
            
            attributes_text = ""
            if aligned_attributes and isinstance(aligned_attributes, dict) and len(aligned_attributes) > 0:
                attr_list = []
                for k, v in aligned_attributes.items():
                    if k and k.strip() and v is not None:
                        attr_key = str(k).strip()
                        attr_value = str(v).strip() if isinstance(v, (str, int, float, bool)) else str(v)
                        attr_list.append(f"{attr_key}: {attr_value}")
                if attr_list:
                    attributes_text = "Attributes: " + "; ".join(attr_list)
            
            relations_text = ""
            if aligned_relations and isinstance(aligned_relations, dict) and len(aligned_relations) > 0:
                rel_list = []
                for rel_name, target_names in aligned_relations.items():
                    if isinstance(target_names, list):
                        for target_name in target_names:
                            rel_list.append(f"{rel_name} -> {target_name}")
                    else:
                        rel_list.append(f"{rel_name} -> {target_names}")
                if rel_list:
                    relations_text = "Relations: " + "; ".join(rel_list)
            
            entity_info = f"{idx + 1}. Entity Name: '{entity.name}'"
            if attributes_text:
                entity_info += f"\n   {attributes_text}"
            if relations_text:
                entity_info += f"\n   {relations_text}"
            entity_info_list.append(entity_info)
        
        prompt = f"""Please analyze the following entities and select the three most likely categories for each entity from the given ontology list, sorted from highest to lowest probability. YOU MUST RETURN EXACTLY THREE CATEGORIES FOR EACH ENTITY:

{chr(10).join(entity_info_list)}

Ontology list: {', '.join(self.ontology_list)}

Example:
1. Entity Name: 'Apple Inc.'
   Attributes: company -> true; industry -> technology; founded -> 1976
   Relations: headquarters_in -> Cupertino; produces -> iPhone

2. Entity Name: 'Python'
   Attributes: language -> true; paradigm -> object-oriented

Response:
1. Organisation, Device, Media
2. ProgrammingLanguage, Software, Technology

IMPORTANT: YOU MUST RETURN EXACTLY THREE CATEGORIES FOR EACH ENTITY, SEPARATED BY COMMAS. Use "Unknown" as the third category if you cannot determine three distinct categories.
Please ONLY return the response for each entity in the format: "number. ontology1, ontology2, ontology3" (one per line)."""
        
        return prompt

    
    def _parse_batch_ontology_result(self, result: str, expected_count: int) -> List[List[str]]:
        """解析批量本体判断结果
        
        Args:
            result: LLM返回的结果
            expected_count: 期望的实体数量
            
        Returns:
            每个实体对应的本体类别列表
        """
        results = []
        lines = result.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if '.' in line:
                parts = line.split('.', 1)
                if len(parts) > 1:
                    ontologies = parts[1].strip().split(',')
                    ontologies = [o.strip() for o in ontologies if o.strip()]
                    results.append(ontologies[:3] if ontologies else ['Unknown', 'Other', 'Entity'])
                else:
                    results.append(['Unknown', 'Other', 'Entity'])
            else:
                ontologies = line.split(',')
                ontologies = [o.strip() for o in ontologies if o.strip()]
                results.append(ontologies[:3] if ontologies else ['Unknown', 'Other', 'Entity'])
        
        while len(results) < expected_count:
            results.append(['Unknown', 'Other', 'Entity'])
        
        return results[:expected_count]

    def _call_llm_for_ontology(self, entity_name: str, entity_attributes: dict, entity_relations: dict, ontology_list: List[str]) -> List[str]:
        """调用LLM进行本体判断
        
        Args:
            entity_name: 实体名称
            entity_attributes: 实体属性信息（字典格式{attr_name: value}）
            entity_relations: 实体关系信息（字典格式{rel_name: [target_entity_name, ...]}）
            ontology_list: 可选的本体列表
            
        Returns:
            最可能的三个本体类别列表
        """
        # 准备属性信息文本 - 增强处理
        attributes_text = ""
        if entity_attributes and isinstance(entity_attributes, dict) and len(entity_attributes) > 0:
            attr_list = []
            count = 0
            for k, v in entity_attributes.items():
                # if count >= 5:  # 限制属性数量，避免提示词过长
                #     break
                # 确保键不为空，值可以转换为字符串
                if k and k.strip() and v is not None:
                    # 确保使用名称而非ID
                    attr_key = str(k).strip()
                    attr_value = str(v).strip() if isinstance(v, (str, int, float, bool)) else str(v)
                    attr_list.append(f"{attr_key}: {attr_value}")
                    count += 1
            
            if attr_list:
                attributes_text = "Attributes: " + "; ".join(attr_list)
        
        # 准备关系信息文本
        relations_text = ""
        if entity_relations and isinstance(entity_relations, dict) and len(entity_relations) > 0:
            rel_list = []
            count = 0
            for rel_name, target_names in entity_relations.items():
                if isinstance(target_names, list):
                    for target_name in target_names:
                        rel_list.append(f"{rel_name} -> {target_name}")
                        count += 1
                        # if count >= 5:
                        #     break
                else:
                    rel_list.append(f"{rel_name} -> {target_names}")
                    count += 1
            if rel_list:
                relations_text = "Relations: " + "; ".join(rel_list)
        
        # 创建提示词
        prompt = f"""Please analyze the following entity information and select the three most likely categories from the given ontology list, sorted from highest to lowest probability:
        
Entity Name: '{entity_name}'
{attributes_text}
{relations_text}
        
Ontology list: {', '.join(ontology_list)}
        
Example:
Entity Name: 'Apple Inc.'
Attributes: company: true; industry: technology; founded: 1976
Relations: headquarters_in -> Cupertino; produces -> iPhone
Response: Organisation, Device, Media
        
Please ONLY and MUST return the names of the three ontology categories, separated by commas."""

        # 打印prompt以便观察
        #print("\n=== LLM Prompt ===")
        #print(prompt)
        
        try:
            
            client_config = LLMConfig.get_openai_client_config("ontology_generation")
            ontology_generation_config = LLMConfig.get_completion_config("ontology_generation")
            client = OpenAI(**client_config)

            # 调用本地的Qwen3-8B模型
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "你是一个知识图谱实体分类专家。"},
                    {"role": "user", "content": prompt}
                ],
                **ontology_generation_config
            )
            
            # 解析返回结果
            result = response.choices[0].message.content.strip()
            
            # 打印result以便观察
            #print("=== LLM Result ===")
            #print(result)
            #print("================")
            
            # 将结果分割为列表
            ontologies = result.split('\n')[-1].replace(' ', '').split(',')
            print(entity_name, ontologies)
            
            # 验证结果是否在ontology_list中
            validated_ontologies = []
            for onto in ontologies:
                validated_ontologies.append(onto)
            
            # 如果没有有效的结果，使用默认值
            if not validated_ontologies:
                return ['Unknown', 'Other', 'Entity'][:3]
            
            return validated_ontologies
        except Exception as e:
            print(f"调用LLM时出错: {e}")
            # 出错时使用简单的回退机制
            return ['Unknown', 'Other', 'Entity'][:3]
    


    def _extract_schema_elements(self, entities: List[UnifiedEntity]) -> tuple[Set[str], Set[str]]:
        """从实体列表中提取关系名和属性名
        
        Args:
            entities: 实体列表
            
        Returns:
            (关系名集合, 属性名集合)
        """
        relation_names = set()
        attribute_names = set()
        
        for entity in entities:
            # 提取关系名
            if entity.relations and isinstance(entity.relations, dict):
                relation_names.update(entity.relations.keys())
            
            # 提取属性名
            if entity.attributes and isinstance(entity.attributes, dict):
                attribute_names.update(entity.attributes.keys())
        return relation_names, attribute_names
    
    def _call_llm_for_schema_alignment(self, source_A_elements: List[str], source_B_elements: List[str], element_type: str) -> Dict[str, str]:
        """调用LLM进行模式对齐
        
        Args:
            source_A_elements: 源A的元素列表
            source_B_elements: 源B的元素列表
            element_type: 元素类型，'relation'或'attribute'
            
        Returns:
            对齐结果字典
        """
        element_type_name = 'Relation' if element_type == 'relation' else 'Attribute'
        
        # 准备提示词
        prompt = f"""
You are a knowledge graph and database schema alignment expert.

**Source A** {element_type_name} list: {', '.join(source_A_elements)}
**Source B** {element_type_name} list: {', '.join(source_B_elements)}

Please find {element_type_name}s in **Source B** that have the exact SAME semantic meaning as those in **Source A**, and provide the alignment results.
The alignment result format should be: 
Source B {element_type_name} -> Corresponding Source A {element_type_name}

Example:
**Source A** attributes: 出生地點, country, DateOfBirth
**Source B** attributes: 國家, birthPlace, birthDate, name

Response:
國家 -> country
birthPlace -> 出生地點
birthDate -> DateOfBirth

Please strictly follow the above format for the output."""

        print(prompt)
        try:

            client_config = LLMConfig.get_openai_client_config("schema_alignment")
            schema_alignment_config = LLMConfig.get_completion_config("schema_alignment")
            
            client = OpenAI(**client_config)

            # 调用本地的Qwen3-8B模型 - 使用英文系统提示词
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a knowledge graph and database schema alignment expert."},
                    {"role": "user", "content": prompt}
                ],
                **schema_alignment_config
            )
            
            # 解析返回结果
            result = response.choices[0].message.content.strip()
            print(f"Raw LLM response for {element_type_name} alignment:\n{result}")
            
            alignments = {}
            
            # 增强的结果解析逻辑
            for line in result.split('\n'):
                line = line.strip()
                # 跳过空行和非对齐行
                if not line or line.startswith(('Source', 'Example:', 'Response:')):
                    continue
                
                # 改进的解析逻辑，处理不同的箭头格式
                if '->' in line:
                    parts = line.split('->')
                elif '→' in line:
                    parts = line.split('→')
                else:
                    continue
                
                if len(parts) == 2:
                    b_element = parts[0].strip()
                    a_element = parts[1].strip()
                    # 确保两个元素都不为空
                    if b_element and a_element:
                        alignments[b_element] = a_element
                        #print(f"Found alignment: {b_element} -> {a_element}")
            
            print(f"Final {element_type_name} alignments: {alignments}")
            return alignments
        except Exception as e:
            print(f"Error calling LLM for schema alignment: {e}")
            # 移除对_simple_heuristic_alignment的调用
            return {}
    
    def perform_schema_alignment(self, report: DataModalReport, source_A_entities: List[UnifiedEntity], source_B_entities: List[UnifiedEntity]) -> SchemaAlignmentResult:
        """执行模式级别对齐
        
        Args:
            source_A_entities: 源A的实体列表
            source_B_entities: 源B的实体列表
            
        Returns:
            SchemaAlignmentResult对象
        """
        
        # 提取关系名和属性名
        print("\n==== Extracting schema elements ====")
        a_relation_names, a_attribute_names = self._extract_schema_elements(source_A_entities)
        b_relation_names, b_attribute_names = self._extract_schema_elements(source_B_entities)
        
        # 创建对齐结果对象
        result = SchemaAlignmentResult()
        
        # 创建规范化字符串到原始字符串的映射，用于关系名
        a_rel_normalized_to_original = {self._normalize_string(rel): rel for rel in a_relation_names}
        
        # 1. 找出匹配的关系名（使用规范化字符串）
        exact_match_relations = {}
        for b_rel in b_relation_names:
            b_rel_normalized = self._normalize_string(b_rel)
            if b_rel_normalized in a_rel_normalized_to_original:
                a_rel_original = a_rel_normalized_to_original[b_rel_normalized]
                exact_match_relations[b_rel] = a_rel_original
        
        # 创建规范化字符串到原始字符串的映射，用于属性名
        a_attr_normalized_to_original = {self._normalize_string(attr): attr for attr in a_attribute_names}
        
        # 2. 找出匹配的属性名（使用规范化字符串）
        exact_match_attributes = {}
        for b_attr in b_attribute_names:
            b_attr_normalized = self._normalize_string(b_attr)
            if b_attr_normalized in a_attr_normalized_to_original:
                a_attr_original = a_attr_normalized_to_original[b_attr_normalized]
                exact_match_attributes[b_attr] = a_attr_original
        

        if report.has_relation_names or report.has_attribute_names:
            # 3. 准备传递给LLM的不一致元素
            # 过滤掉已经匹配的关系
            remaining_a_relations = sorted([rel for rel in a_relation_names if rel not in exact_match_relations.values()])
            remaining_b_relations = sorted([rel for rel in b_relation_names if rel not in exact_match_relations.keys()])
            
            # 过滤掉已经匹配的属性
            remaining_a_attributes = sorted([attr for attr in a_attribute_names if attr not in exact_match_attributes.values()])
            remaining_b_attributes = sorted([attr for attr in b_attribute_names if attr not in exact_match_attributes.keys()])
            
            print(f"\nExact match relations: {exact_match_relations}")
            print(f"Exact match attributes: {exact_match_attributes}")
            print(f"Remaining relations for LLM alignment - A: {remaining_a_relations}, B: {remaining_b_relations}")
            print(f"Remaining attributes for LLM alignment - A: {remaining_a_attributes}, B: {remaining_b_attributes}")
            
            # 初始化LLM对齐结果
            llm_relation_alignments = {}
            llm_attribute_alignments = {}
            
            if report.has_relation_names:
                # 多轮关系对齐
                for round in range(self.schema_alignment_round):
                    if not remaining_a_relations or not remaining_b_relations:
                        break
                        
                    print(f"\n=== Schema Alignment Round {round+1} for Relations ===")
                    print(f"Remaining relations - A: {remaining_a_relations}, B: {remaining_b_relations}")
                    
                    # 调用LLM进行关系对齐
                    current_round_alignments = self._call_llm_for_schema_alignment(
                        remaining_a_relations, 
                        remaining_b_relations, 
                        'relation'
                    )
                    
                    # 验证并保留有效的对齐结果
                    valid_alignments = {}
                    for b_rel, a_rel in current_round_alignments.items():
                        # 检查两个关系是否都存在于剩余列表中
                        if a_rel in remaining_a_relations and b_rel in remaining_b_relations:
                            valid_alignments[b_rel] = a_rel
                            print(f"Valid alignment found: {b_rel} -> {a_rel}")
                        else:
                            print(f"Invalid alignment skipped: {b_rel} -> {a_rel} (not in remaining lists)")
                    
                    # 更新对齐结果
                    llm_relation_alignments.update(valid_alignments)
                    
                    # 更新剩余关系列表，移除已对齐的元素
                    for b_rel, a_rel in valid_alignments.items():
                        if a_rel in remaining_a_relations:
                            remaining_a_relations.remove(a_rel)
                        if b_rel in remaining_b_relations:
                            remaining_b_relations.remove(b_rel)
                print(f"\nLLM aligned relations {len(llm_relation_alignments)}: \n{llm_relation_alignments}")
                result.relation_alignments = {**exact_match_relations, **llm_relation_alignments}

            if report.has_attribute_names:
                # 多轮属性对齐
                for round in range(self.schema_alignment_round):
                    if not remaining_a_attributes or not remaining_b_attributes:
                        break
                        
                    print(f"\n=== Schema Alignment Round {round+1} for Attributes ===")
                    print(f"Remaining attributes - A: {remaining_a_attributes}, B: {remaining_b_attributes}")
                    
                    # 调用LLM进行属性对齐
                    current_round_alignments = self._call_llm_for_schema_alignment(
                        remaining_a_attributes, 
                        remaining_b_attributes, 
                        'attribute'
                    )
                    
                    # 验证并保留有效的对齐结果
                    valid_alignments = {}
                    for b_attr, a_attr in current_round_alignments.items():
                        # 检查两个属性是否都存在于剩余列表中
                        if a_attr in remaining_a_attributes and b_attr in remaining_b_attributes:
                            valid_alignments[b_attr] = a_attr
                            print(f"Valid alignment found: {b_attr} -> {a_attr}")
                        else:
                            print(f"Invalid alignment skipped: {b_attr} -> {a_attr} (not in remaining lists)")
                    
                    # 更新对齐结果
                    llm_attribute_alignments.update(valid_alignments)
                    
                    # 更新剩余属性列表，移除已对齐的元素
                    for b_attr, a_attr in valid_alignments.items():
                        if a_attr in remaining_a_attributes:
                            remaining_a_attributes.remove(a_attr)
                        if b_attr in remaining_b_attributes:
                            remaining_b_attributes.remove(b_attr)

                print(f"LLM aligned attributes {len(llm_attribute_alignments)}: \n{llm_attribute_alignments}")
                result.attribute_alignments = {**exact_match_attributes, **llm_attribute_alignments}

        
        # 输出对齐结果
        print("\n=== Schema Alignment Results ===")
        print("\nRelation Alignments:")
        if result.relation_alignments:
            for b_rel, a_rel in result.relation_alignments.items():
                print(f"  {b_rel} -> {a_rel}")
        else:
            print("  No relation alignments found")
        
        print("\nAttribute Alignments:")
        if result.attribute_alignments:
            for b_attr, a_attr in result.attribute_alignments.items():
                print(f"  {b_attr} -> {a_attr}")
        else:
            print("  No attribute alignments found")
            print("  Debug info - Source A attributes:", a_attribute_names)
            print("  Debug info - Source B attributes:", b_attribute_names)
        print("====================\n")
        
        return result
    
    # 修改apply_schema_alignment方法，返回新的实体列表而不是直接修改原列表
    def apply_schema_alignment(self, entities: List[UnifiedEntity], alignments: SchemaAlignmentResult) -> List[UnifiedEntity]:
        """应用模式对齐到实体数据，返回新的实体列表
        
        Args:
            entities: 实体列表
            alignments: 模式对齐结果
            
        Returns:
            应用对齐后的新实体列表
        """
        # 创建实体副本列表，避免修改原实体
        aligned_entities = []
        
        for entity in entities:
            # 创建实体副本
            from copy import deepcopy
            new_entity = deepcopy(entity)
            
            # 只修改源B实体（KG2或B）
            if new_entity.source in ['KG2', 'B']:
                # 应用关系对齐
                if new_entity.relations and isinstance(new_entity.relations, dict):
                    aligned_relations = {}
                    for rel_name, targets in new_entity.relations.items():
                        # 如果关系名在对齐结果中，使用对齐后的名称
                        if rel_name in alignments.relation_alignments:
                            aligned_relations[alignments.relation_alignments[rel_name]] = targets
                        else:
                            aligned_relations[rel_name] = targets
                    new_entity.relations = aligned_relations
                
                # 应用属性对齐
                if new_entity.attributes and isinstance(new_entity.attributes, dict):
                    aligned_attributes = {}
                    for attr_name, value in new_entity.attributes.items():
                        # 如果属性名在对齐结果中，使用对齐后的名称
                        if attr_name in alignments.attribute_alignments:
                            aligned_attributes[alignments.attribute_alignments[attr_name]] = value
                        else:
                            aligned_attributes[attr_name] = value
                    new_entity.attributes = aligned_attributes
            
            # 添加到结果列表
            aligned_entities.append(new_entity)
        
        return aligned_entities
    
    # 修改analyze方法以符合用户要求的流程
    def analyze(self, source_A_entities: List[UnifiedEntity], source_B_entities: List[UnifiedEntity], data_path: str) -> DataModalReport:
        """执行数据分析，包括模态分析、模式对齐、信息密度评估和本体判断
        
        Args:
            source_A_entities: 源A的实体列表
            source_B_entities: 源B的实体列表
            data_path: 数据路径
            
        Returns:
            DataModalReport对象
        """
        # 获取分析结果保存路径
        analysis_path = self.get_analysis_save_path(data_path)
        entities_path = self.get_entities_save_path(data_path)
        
        # 尝试从缓存加载分析结果
        report, schema_alignments, loaded_source_A_entities, loaded_source_B_entities = self._load_analysis_data(analysis_path)
        # 如果成功加载到报告，直接返回
        if report:
            return report, schema_alignments, loaded_source_A_entities, loaded_source_B_entities
        
        # 执行全新分析
        print("\n=== Performing fresh analysis ===")
        
        # 执行模态分析
        print("Performing modal analysis...")
        report = self.analyze_modal(source_A_entities, source_B_entities, data_path)
        
        # 执行模式对齐
        print("Performing schema level alignment...")
        schema_alignments = self.perform_schema_alignment(report, source_A_entities, source_B_entities)
        
        # 应用模式对齐到源B实体数据，保存为新变量
        print("Applying schema alignments to source B entities...")
        source_B_entities_schema_aligned = self.apply_schema_alignment(source_B_entities, schema_alignments)
        
        # 评估信息密度
        self.evaluate_info_density(source_A_entities, source_B_entities_schema_aligned)
        
        # # 如果有名称，基于对齐结果进行本体判断
        if report.has_names:
            print("Performing ontology determination based on alignment results...")
            self.determine_ontology(source_A_entities, source_B_entities_schema_aligned)
        
        # 保存分析结果
        print(f"Saving analysis results to: {analysis_path}")
        self._save_analysis_data(report, schema_alignments, analysis_path)
        
        # 保存实体数据
        print(f"Saving entity data to: {entities_path}")
        self._save_entity_data(source_A_entities, source_B_entities_schema_aligned, entities_path)
        
        return report, schema_alignments, source_A_entities, source_B_entities_schema_aligned

    def _save_entity_data(self, source_A_entities: List[UnifiedEntity], source_B_entities: List[UnifiedEntity], save_path: str):
        """保存实体分析数据到文件
        
        Args:
            source_A_entities: 源A的实体列表
            source_B_entities: 源B的实体列表
            save_path: 保存路径
        """
        # 创建保存目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 准备实体数据
        entities_data = {
            'source_A': [
                {
                    'id': entity.id,
                    'source': entity.source,
                    'info_density_score': float(entity.info_density_score) if entity.info_density_score is not None else None,  # 转换为Python float
                    'ontology': entity.ontology,
                    'name': entity.name,
                    'origin_name': entity.origin_name,
                    'attributes': entity.attributes,
                    'relations': entity.relations
                }
                for entity in source_A_entities
            ],
            'source_B': [
                {
                    'id': entity.id,
                    'source': entity.source,
                    'info_density_score': float(entity.info_density_score) if entity.info_density_score is not None else None,  # 转换为Python float
                    'ontology': entity.ontology,
                    'name': entity.name,
                    'origin_name': entity.origin_name,
                    'attributes': entity.attributes,
                    'relations': entity.relations
                }
                for entity in source_B_entities
            ]
        }
        
        # 保存到文件
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(entities_data, f, ensure_ascii=False, indent=2, cls=CustomJSONEncoder)

    def _load_entity_data(self, source_A_entities: List[UnifiedEntity], source_B_entities: List[UnifiedEntity], load_path: str) -> bool:
        """从文件加载实体分析数据
        
        Args:
            source_A_entities: 源A的实体列表
            source_B_entities: 源B的实体列表
            load_path: 加载路径
            
        Returns:
            是否成功加载
        """
        if not os.path.exists(load_path):
            return False
        
        try:
            with open(load_path, 'r', encoding='utf-8') as f:
                entities_data = json.load(f)
            
            # 创建实体ID到实体对象的映射
            source_A_dict = {entity.id: entity for entity in source_A_entities}
            source_B_dict = {entity.id: entity for entity in source_B_entities}
            
            # 恢复源A实体的数据
            for entity_data in entities_data.get('source_A', []):
                entity_id = entity_data.get('id')
                if entity_id in source_A_dict:
                    source_A_dict[entity_id].info_density_score = entity_data.get('info_density_score')
                    source_A_dict[entity_id].ontology = entity_data.get('ontology')
                    source_A_dict[entity_id].name = entity_data.get('name')  # 恢复名称字段
                    source_A_dict[entity_id].origin_name = entity_data.get('origin_name')
                    source_A_dict[entity_id].attributes = entity_data.get('attributes', {})  # 恢复属性字段
                    source_A_dict[entity_id].relations = entity_data.get('relations', {})  # 恢复关系字段
            
            # 恢复源B实体的数据
            for entity_data in entities_data.get('source_B', []):
                entity_id = entity_data.get('id')
                if entity_id in source_B_dict:
                    source_B_dict[entity_id].info_density_score = entity_data.get('info_density_score')
                    source_B_dict[entity_id].ontology = entity_data.get('ontology')
                    source_B_dict[entity_id].name = entity_data.get('name')  # 恢复名称字段
                    source_B_dict[entity_id].origin_name = entity_data.get('origin_name')
                    source_B_dict[entity_id].attributes = entity_data.get('attributes', {})  # 恢复属性字段
                    source_B_dict[entity_id].relations = entity_data.get('relations', {})  # 恢复关系字段
            
            return True
        except Exception as e:
            print(f"Error loading entity data: {e}")
            return False
    
    def _save_analysis_data(self, report: DataModalReport, schema_alignments: SchemaAlignmentResult, save_path: str):
        """保存分析结果数据到文件
        
        Args:
            report: 数据模态报告
            schema_alignments: 模式对齐结果
            save_path: 保存路径
        """
        # 创建保存目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 准备分析数据
        analysis_data = {
            'report': report.to_dict(),
            'schema_alignments': schema_alignments.to_dict()
        }
        
        # 保存到文件
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, ensure_ascii=False, indent=2)
    
    def _load_analysis_data(self, load_path: str) -> tuple[Optional[DataModalReport], Optional[SchemaAlignmentResult], Optional[List[UnifiedEntity]], Optional[List[UnifiedEntity]]]:
        """从文件加载分析结果数据和实体数据，并执行缓存结果的后续处理
        
        Args:
            load_path: 分析结果加载路径
            
        Returns:
            (数据模态报告, 模式对齐结果, 源A实体列表, 源B实体列表)，如果加载失败则返回(None, None, None, None)
        """
        if not os.path.exists(load_path):
            return None, None, None, None
        
        try:
            # 加载分析结果数据
            with open(load_path, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
            
            # 恢复报告数据
            report = DataModalReport.from_dict(analysis_data.get('report', {}))
            
            # 恢复模式对齐结果
            schema_alignments = SchemaAlignmentResult.from_dict(analysis_data.get('schema_alignments', {}))
            
            # 同时加载实体数据
            # 从load_path中提取data_path相关信息来构建entities.json路径
            # 假设load_path格式为 ./analysis_results/{data_dir_name}_analysis.json
            analysis_file_name = os.path.basename(load_path)
            data_dir_name = analysis_file_name.replace('_analysis.json', '')
            entities_path = os.path.join('./analysis_results', f'{data_dir_name}_entities.json')
            
            source_A_entities = []
            source_B_entities = []
            
            if os.path.exists(entities_path):
                try:
                    with open(entities_path, 'r', encoding='utf-8') as f:
                        entities_data = json.load(f)
                    
                    # 创建源A实体对象
                    for entity_data in entities_data.get('source_A', []):
                        entity = UnifiedEntity(entity_data.get('id'), entity_data.get('source'))
                        entity.info_density_score = entity_data.get('info_density_score')
                        entity.ontology = entity_data.get('ontology')
                        entity.name = entity_data.get('name')
                        entity.attributes = entity_data.get('attributes', {})
                        entity.relations = entity_data.get('relations', {})
                        source_A_entities.append(entity)
                    
                    # 创建源B实体对象
                    for entity_data in entities_data.get('source_B', []):
                        entity = UnifiedEntity(entity_data.get('id'), entity_data.get('source'))
                        entity.info_density_score = entity_data.get('info_density_score')
                        entity.ontology = entity_data.get('ontology')
                        entity.name = entity_data.get('name')
                        entity.attributes = entity_data.get('attributes', {})
                        entity.relations = entity_data.get('relations', {})
                        source_B_entities.append(entity)
                    print(f"Successfully loaded entity data from: {entities_path}")
                except Exception as e:
                    print(f"Error loading entity data: {e}")
                    # 如果加载实体数据失败，返回空的实体列表
                    source_A_entities = []
                    source_B_entities = []
            else:
                print(f"Entity data file not found: {entities_path}")
                # 如果实体数据文件不存在，返回空的实体列表
                source_A_entities = []
                source_B_entities = []
            
            # 打印缓存的对齐结果
            print(f"Found cached analysis results: {load_path}")
            print("\n=== Cached Schema Alignment Results ===")
            print("\nRelation Alignments:")
            if schema_alignments.relation_alignments:
                for b_rel, a_rel in schema_alignments.relation_alignments.items():
                    print(f"  {b_rel} -> {a_rel}")
            else:
                print("  No relation alignments")
            
            print("\nAttribute Alignments:")
            if schema_alignments.attribute_alignments:
                for b_attr, a_attr in schema_alignments.attribute_alignments.items():
                    print(f"  {b_attr} -> {a_attr}")
            else:
                print("  No attribute alignments")
            print("=========================\n")
            
            # # 应用模式对齐到源B实体数据
            # print("Applying cached schema alignments to source B entities...")
            # self.apply_schema_alignment(source_B_entities, schema_alignments)
            
            # # 评估信息密度
            # self.evaluate_info_density(source_A_entities, source_B_entities)
            
            # # 如果有名称，基于对齐结果进行本体判断
            # if report.has_names:
            #     print("Performing ontology determination based on cached alignment results...")
            #     self.determine_ontology(source_A_entities, source_B_entities, schema_alignments)
            
            return report, schema_alignments, source_A_entities, source_B_entities
        except Exception as e:
            print(f"Error loading analysis data: {e}")
            return None, None, None, None
    
    def get_entities_save_path(self, data_path: str) -> str:
        """获取实体数据保存路径
        
        Args:
            data_path: 数据路径
            
        Returns:
            实体数据保存路径
        """
        # 创建一个基于数据路径的唯一保存目录
        data_dir_name = os.path.basename(data_path)
        return os.path.join('./analysis_results', f'{data_dir_name}_entities.json')
    def get_analysis_save_path(self, data_path: str) -> str:
        """获取分析结果保存路径
        
        Args:
            data_path: 数据路径
            
        Returns:
            分析结果保存路径
        """
        # 创建一个基于数据路径的唯一保存目录
        data_dir_name = os.path.basename(data_path)
        return os.path.join('./analysis_results', f'{data_dir_name}_analysis.json')

if __name__ == "__main__":
    # 示例用法
    pass