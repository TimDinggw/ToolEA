import os
import json
import pickle
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
import concurrent.futures
import time

from data_loader import UnifiedEntity, AlignmentPair
from tools import BaseTool
from data_analyzer import DataModalReport
from evaluation import evaluate_alignments  # 导入评估函数
from openai import OpenAI  # 添加OpenAI客户端导入
from LLMConfig import LLMConfig  # 导入LLM配置
import re
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


class AlignmentAgent:
    """对齐执行Agent"""
    def __init__(self, data_name: str,data_modal_report: DataModalReport, confidence_threshold: float = 0.7, max_iterations: int = 10, convergence_threshold: float = 0, top_k: int = 10, percentage_per_round: float = 0.1, data_path: str = None, seed_alignments: List[AlignmentPair] = None, test_alignments: List[AlignmentPair] = None, supervision: bool = True, unsupervised_seed_count: float = 0):
        self.data_name = data_name
        self.data_modal_report = data_modal_report
        self.confidence_threshold = confidence_threshold
        self.dynamic_threshold = confidence_threshold  # 动态阈值

        self.max_iterations = max_iterations if max_iterations > 0 else 10
        self.convergence_threshold = convergence_threshold  # 收敛阈值
        self.memory = {}  # {id_a: (id_b, confidence)}
        self.top_k = top_k  # top-k搜索的k值
        self.reverse_memory = {}  # {id_b: id_a}
        self.tool_outputs = []  # [(tool_name, id_a, id_b, confidence)]
        self.source_A_entities = None
        self.source_B_entities = None
        self.data_path = data_path  # 保存数据路径
        self.test_alignments = test_alignments  # 保存测试对齐用于每轮评估
        self.tool_weights = {}  # 存储工具权重
        self.percentage_per_round = percentage_per_round
        self.supervision = supervision  # 有监督/无监督模式
        self.unsupervised_seed_count = unsupervised_seed_count
        
        # 新增：专门存储种子对齐的变量
        self.seed_alignments = seed_alignments
        self.seed_memory = set()  # 存储种子对齐中的实体ID
        self.seed_reverse_memory = set()  # 存储种子对齐中的反向实体ID
        

        # 新增：存储历史轮次信息
        self.previous_round_tools = {}  # {round_num: [tool_names]}
        self.previous_round_weights = {}  # {round_num: {tool_name: weight}}
        self.previous_round_alignments = {}  # {round_num: alignment_count}

        self.warmup = True
        self.pre_count = 0  # 上一轮对齐数量，用于计算占比  

        # 创建输出目录
        os.makedirs("./output", exist_ok=True)
        # 创建批量查询目录
        os.makedirs("./batch_queries", exist_ok=True)

    def _sort_entities_by_density(self, entities: List[UnifiedEntity]) -> List[UnifiedEntity]:
        """按信息密度排序实体        
        使用info_density_score属性进行降序排序
        """
        return sorted(entities, key=lambda x: x.info_density_score, reverse=True)
    

    def _save_round_alignments_files(self, round_num: int, new_alignments: List):
        """保存当前轮次的对齐结果到两个文件
        round_{round_num}_alignments.txt: 当前轮次找到的对齐，包含置信度
        round_{round_num}_alignments_detailed.txt: 包含详细信息的当前轮次对齐
        """
        # 构建文件名前缀，包含数据集名称和percentage_per_round
        file_prefix = f"./output/{self.data_name}_p{int(self.percentage_per_round*100)}_round_{round_num}"
        
        # 保存本轮新对齐（包含置信度）
        alignments_file_path = f"{file_prefix}_alignments.txt"
        with open(alignments_file_path, 'w', encoding='utf-8') as f:
            for id_a, id_b, conf in new_alignments:
                f.write(f"{id_a}\t{id_b}\t{conf:.4f}\n")
        

        # 保存本轮所有对齐
        all_alignments = [(id_a, id_b) for id_a, (id_b, conf) in self.memory.items()]
        alignments_all_file_path = f"{file_prefix}_alignments_all.txt"
        with open(alignments_all_file_path, 'w', encoding='utf-8') as f:
          for id_a, id_b in all_alignments:
                f.write(f"{id_a}\t{id_b}\n")

        # 保存本轮详细结果（包含实体名称）
        detailed_file_path = f"{file_prefix}_alignments_detailed.txt"
        alignments_all_name_file_path = f"{file_prefix}_alignments_all_name.txt"

        
        a_entity_map = {e.id: e.name for e in self.source_A_entities}
        b_entity_map = {e.id: e.name for e in self.source_B_entities}
        
        with open(detailed_file_path, 'w', encoding='utf-8') as f:
            f.write("id_a\tname_a\tid_b\tname_b\tconfidence\n")
            for id_a, id_b, conf in new_alignments:
                name_a = a_entity_map.get(id_a, "None")
                name_b = b_entity_map.get(id_b, "None")
                f.write(f"{id_a}\t{name_a}\t{id_b}\t{name_b}\t{conf:.4f}\n")
                
        with open(alignments_all_name_file_path, 'w', encoding='utf-8') as f:
            for id_a, id_b in all_alignments:
                name_a = a_entity_map.get(id_a, "None")
                name_b = b_entity_map.get(id_b, "None")
                f.write(f"{name_a}\t{name_b}\n")

        
        print(f"第 {round_num} 轮生成了 {len(new_alignments)} 个新对齐")
        print(f"第 {round_num} 轮对齐结果已保存到 {alignments_file_path}")
        print(f"第 {round_num} 轮详细对齐结果已保存到 {detailed_file_path}")
        print(f"第 {round_num} 轮详细对齐结果已保存到 {alignments_all_file_path}")
    
    def _evaluate_current_round(self, round_num: int):
        """评估当前轮次的结果 - 修改为使用round_{round_num}_alignments_all.txt文件，并排除种子对齐"""
        if not self.test_alignments:
            print("未提供测试对齐，跳过本轮评估")
            return None, None, None
        
        # 读取当前轮次的所有已知对齐文件（包括种子对齐和之前轮次的结果）
        all_alignments_file = f"./output/{self.data_name}_p{int(self.percentage_per_round*100)}_round_{round_num}_alignments_all.txt"
        if not os.path.exists(all_alignments_file):
            print(f"警告: 当前轮次的所有对齐文件不存在: {all_alignments_file}")
            return None, None, None
        
        # 读取文件中的对齐
        current_alignments = []
        try:
            with open(all_alignments_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            id_a = parts[0]
                            id_b = parts[1]
                            # 使用默认值1.0，因为alignments_all.txt不包含置信度
                            conf = 1.0
                            current_alignments.append((id_a, id_b, conf))
        except Exception as e:
            print(f"读取对齐文件失败: {e}")
            return None, None, None
        
        # 读取种子对齐文件
        seed_alignments = []
        seed_alignments_file = f"./output/{self.data_name}_p{int(self.percentage_per_round*100)}_round_0_alignments_all.txt"
        if os.path.exists(seed_alignments_file):
            try:
                with open(seed_alignments_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split('\t')
                            if len(parts) >= 2:
                                id_a = parts[0]
                                id_b = parts[1]
                                conf = 1.0  # 种子对齐置信度设为1.0
                                seed_alignments.append((id_a, id_b, conf))
            except Exception as e:
                print(f"读取种子对齐文件失败: {e}")
        
        # 进行评估（排除种子对齐）
        try:
            precision, recall, f1_score = evaluate_alignments(current_alignments, seed_alignments, self.test_alignments)
            
            # 打印评估结果
            print(f"第 {round_num} 轮评估结果 - 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1分数: {f1_score:.4f}")
            current_count = len(current_alignments)
            percentage = (current_count - self.pre_count) / current_count * 100 if current_count > 0 else 0
            print(f"已知对齐数量: {current_count}, 当前轮次对齐数量: {current_count - self.pre_count}, 占比: {percentage:.2f}%")
            self.pre_count = current_count

            # 保存评估结果到文件
            eval_file_path = f"./output/{self.data_name}_p{int(self.percentage_per_round*100)}_round_{round_num}_evaluation.txt"
            with open(eval_file_path, 'w', encoding='utf-8') as f:
                f.write(f"精确率: {precision:.4f}\n")
                f.write(f"召回率: {recall:.4f}\n")
                f.write(f"F1分数: {f1_score:.4f}\n")
            
            print(f"第 {round_num} 轮评估结果已保存到 {eval_file_path}")
            return precision, recall, f1_score
            
        except Exception as e:
            print(f"第 {round_num} 轮评估失败: {e}")
            return None, None, None
    
    def _save_final_alignments(self) -> List[Tuple[str, str, float]]:
        """保存最终对齐结果 - 修改为直接使用最后一轮的round_{round_num}_alignments_all.txt"""
        final_alignments = [(id_a, id_b, conf) for id_a, (id_b, conf) in self.memory.items()]
        
        # 按置信度排序
        final_alignments.sort(key=lambda x: x[2], reverse=True)
        
        # 不再保存final_alignments.txt和final_alignments_detailed.txt
        # 直接使用最后一轮的round_{round_num}_alignments_all.txt进行最终测试
        print("最终对齐结果已包含在最后一轮的round_{round_num}_alignments_all.txt文件中")
        print(f"总对齐数量: {len(final_alignments)}")
        
        return final_alignments

    def _save_seed_alignments_to_file(self, seed_alignments: List[AlignmentPair]) -> str:
        """保存种子对齐到文件，返回文件路径"""
        # 收集种子对齐
        seed_aligns = []
        for alignment in seed_alignments:
            if alignment.label == 1:  # 只添加正样本对齐
                seed_aligns.append((alignment.id_a, alignment.id_b))
        
        # 保存到文件
        file_path = f"./output/{self.data_name}_p{int(self.percentage_per_round*100)}_round_0_alignments_all.txt"
        with open(file_path, 'w', encoding='utf-8') as f:
            for alignment in seed_alignments:
                f.write(f"{alignment.id_a}\t{alignment.id_b}\n")

        # 保存到详细文件（包含置信度）
        detailed_file_path = f"./output/{self.data_name}_p{int(self.percentage_per_round*100)}_round_0_alignments_all_detailed.txt"
        with open(detailed_file_path, 'w', encoding='utf-8') as f:
            f.write("id_a\tid_b\tconfidence\n")
            for alignment in seed_alignments:
                f.write(f"{alignment.id_a}\t{alignment.id_b}\t{alignment.confidence:.4f}\n")
        
        print(f"种子对齐已保存到 {file_path}，共 {len(seed_aligns)} 个对齐")
        print(f"种子对齐详细结果已保存到 {detailed_file_path}")
        return file_path
    
    def _run_tools_preprocessing(self, available_tools: List[BaseTool], seed_aligns_path: str):
        """对所有工具执行预处理"""
        if not available_tools:
            return
        
        print(f"开始对所有 {len(available_tools)} 个工具执行预处理...")
        
        # 使用线程池并行执行预处理
        def run_tool_preprocessing(tool):
            print(f"开始运行{tool.name}的预处理...")
            tool.run_preprocessing(self.data_path, seed_aligns_path)
            return tool.name
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(available_tools))) as executor:
            # 提交所有任务
            future_to_tool = {executor.submit(run_tool_preprocessing, tool): tool for tool in available_tools}
            
            # 等待所有任务完成并获取结果
            for future in concurrent.futures.as_completed(future_to_tool):
                tool_name = future_to_tool[future]
                try:
                    result = future.result()
                    print(f"{result} 的预处理执行完成")
                except Exception as e:
                    print(f"{tool_name} 的预处理执行失败: {e}")
        
        print("所有工具预处理完成")
    
    def _load_cached_results(self):
        """加载缓存的结果
        
        检查是否存在缓存文件，如果存在则加载
        """
        # 检查种子对齐文件是否存在
        seed_alignments_file = f"./output/{self.data_name}_p{int(self.percentage_per_round*100)}_round_0_alignments_all.txt"
        if os.path.exists(seed_alignments_file):
            print(f"发现缓存文件，开始加载...")
            
            # 加载种子对齐
            try:
                with open(seed_alignments_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split('\t')
                            if len(parts) >= 2:
                                id_a = parts[0]
                                id_b = parts[1]
                                # 假设置信度为1.0
                                self.memory[id_a] = (id_b, 1.0)
                                self.reverse_memory[id_b] = id_a
                                self.seed_memory.add(id_a)
                                self.seed_reverse_memory.add(id_b)
                print(f"成功加载种子对齐，共 {len(self.memory)} 个对齐")
            except Exception as e:
                print(f"加载种子对齐失败: {e}")
            
            # 加载每轮的结果
            for round_num in range(1, self.max_iterations + 1):
                # 检查对齐文件是否存在
                alignments_file = f"./output/{self.data_name}_p{int(self.percentage_per_round*100)}_round_{round_num}_alignments.txt"
                if os.path.exists(alignments_file):
                    try:
                        with open(alignments_file, 'r', encoding='utf-8') as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    parts = line.split('\t')
                                    if len(parts) >= 3:
                                        id_a = parts[0]
                                        id_b = parts[1]
                                        conf = float(parts[2])
                                        self.memory[id_a] = (id_b, conf)
                                        self.reverse_memory[id_b] = id_a
                        print(f"成功加载第 {round_num} 轮对齐结果")
                    except Exception as e:
                        print(f"加载第 {round_num} 轮对齐结果失败: {e}")
                
                # 检查权重文件是否存在
                weight_file = f"./output/{self.data_name}_p{int(self.percentage_per_round*100)}_round_{round_num}_weights.txt"
                if os.path.exists(weight_file):
                    try:
                        tool_weights = {}
                        with open(weight_file, 'r', encoding='utf-8') as f:
                            for line in f:
                                line = line.strip()
                                if ':' in line and not line.startswith('Round') and not line.startswith('Dynamic') and not line.startswith('Tool') and not line.startswith('Weight Sum'):
                                    parts = line.split(':', 1)
                                    if len(parts) == 2:
                                        tool_name = parts[0].strip()
                                        weight = float(parts[1].strip())
                                        tool_weights[tool_name] = weight
                        if tool_weights:
                            self.previous_round_weights[round_num] = tool_weights
                            print(f"成功加载第 {round_num} 轮工具权重")
                    except Exception as e:
                        print(f"加载第 {round_num} 轮工具权重失败: {e}")
            
            return True
        return False

    def run(self, source_A_entities: List[UnifiedEntity], source_B_entities: List[UnifiedEntity],
            available_tools: List[BaseTool]) -> List[Tuple[str, str, float]]:
        """运行对齐过程
        
        按照要求逐轮处理未对齐的高信息密度实体
        """
        # 保存实体信息
        self.source_A_entities = source_A_entities
        self.source_B_entities = source_B_entities
        
        # 按信息密度排序实体
        self.sorted_entities_A = self._sort_entities_by_density(source_A_entities)
        self.sorted_entities_B = self._sort_entities_by_density(source_B_entities)
        print(f"按信息密度排序实体完成: A集合{len(self.sorted_entities_A)}个, B集合{len(self.sorted_entities_B)}个")
        
        # 检查是否存在缓存文件
        if self._load_cached_results():
            print("缓存加载完成，继续执行")
        else:
            # 1. 初始化记忆池
            if self.supervision:
                print("有监督模式：使用提供的种子对齐")
                seed_alignments = self.seed_alignments
            else:
                print("无监督模式：生成伪对齐种子")
                seed_alignments = self._generate_unsupervised_seeds(source_A_entities, source_B_entities, available_tools)
            self._init_memory(seed_alignments)
            # 保存对齐文件
            self._save_seed_alignments_to_file(seed_alignments)
        
        # 2. 工具预执行 - 为每个工具设置实体信息和对齐记忆
        # print("开始工具预执行和初始化...")
        # for tool in available_tools:
        #     tool.set_entities(source_A_entities, source_B_entities)
        #     # 为RelationSimilarityTool设置对齐记忆
        #     if hasattr(tool, 'set_alignment_memory'):
        #         tool.set_alignment_memory(self.memory)
        # print("所有工具初始化完成，开始对齐迭代。")
        
        # 3. 迭代对齐主循环
        for round_idx in range(self.max_iterations):
            current_round_num = round_idx + 1
            
            # 检查当前轮次是否已经完成
            alignments_file = f"./output/{self.data_name}_p{int(self.percentage_per_round*100)}_round_{current_round_num}_alignments.txt"
            if os.path.exists(alignments_file):
                print(f"第 {current_round_num} 轮已经完成，跳过")
                continue
                
            print(f"\n===== 第 {current_round_num} 轮迭代开始 =====")
            start_round_time = time.time()
            new_alignments_this_round = []
            
            # 3.0: 确定本轮使用的工具
            if self.warmup and current_round_num == 1:
                # 第一轮使用所有工具
                tools_for_this_round = available_tools
                print("第1轮：使用所有可用工具")
            else:
                # 后续轮次由LLM决定使用哪些工具
                tools_for_this_round = self._determine_tools_for_round(current_round_num, available_tools)
                #tools_for_this_round = available_tools
                print(f"第{current_round_num}轮：LLM选择了{len(tools_for_this_round)}个工具")
            
            

            # 保存本轮使用的工具信息
            self.previous_round_tools[current_round_num] = [tool.name for tool in tools_for_this_round]
            
            # 3.1: 在每轮开始前，利用所有已知对齐对当前轮次使用的工具执行run_preprocessing初始化
            print(f"第 {current_round_num} 轮开始前，执行工具预处理...")
            
            seed_aligns_path = f"./output/{self.data_name}_p{int(self.percentage_per_round*100)}_round_{round_idx}_alignments_all.txt"

            for tool in tools_for_this_round:
                tool.set_current_round(round_idx)


            # 只对当前轮次使用的工具执行预处理
            #if current_round_num % 2 == 1:
            self._run_tools_preprocessing(tools_for_this_round, seed_aligns_path)
            
            # 确定本轮要处理的实体百分比
            if self.warmup and current_round_num == 1:
                percent_to_process = self.percentage_per_round
            else:
                percent_to_process = min((current_round_num - 1) * self.percentage_per_round, 1.0)  # 第一轮10%，第二轮20%，最多100%
            print(f"本轮处理前 {percent_to_process * 100:.0f}% 的未对齐实体")
            
            # 获取本轮要处理的实体（未对齐的高信息密度实体）
            total_a_to_process = int(len(source_A_entities) * percent_to_process)
            total_b_to_process = int(len(source_B_entities) * percent_to_process)
            
            # 选取未对齐的高信息密度实体
            current_round_entities_A = []
            for entity in self.sorted_entities_A[:total_a_to_process]:
                if entity.id not in self.seed_memory:  # 只排除种子对齐中的实体
                    current_round_entities_A.append(entity)
    
            current_round_entities_B = []
            for entity in self.sorted_entities_B[:total_b_to_process]:
                if entity.id not in self.seed_reverse_memory:  # 只排除种子对齐中的实体
                    current_round_entities_B.append(entity)
            
            print(f"本轮处理A集合{len(current_round_entities_A)}个实体, B集合{len(current_round_entities_B)}个实体")
            
            # 3.2: 候选生成 - 修改为只使用当前轮次选择的工具
            candidate_pairs_with_scores = self._generate_candidates_with_scores(
                tools_for_this_round, current_round_entities_A, current_round_entities_B, current_round_num)
            print(f"生成了 {len(candidate_pairs_with_scores)} 个带置信度的候选对")
            
            # 3.3: 多工具融合打分 - 修改为对top-k候选进行其他工具的打分
            scored_candidates = self._score_candidates_enhanced(candidate_pairs_with_scores, tools_for_this_round, current_round_num)
            print(f"为 {len(scored_candidates)} 个候选对完成多工具打分")
            
            # 3.5: 排序 & 选择高置信对齐
            if self.warmup and current_round_num == 1:
                print(f"热身轮：设置阈值为{self.dynamic_threshold}，不产生对齐") 
            
            scored_candidates.sort(key=lambda x: x[2], reverse=True)
            high_conf_pairs = [(a, b, c) for (a, b, c) in scored_candidates if c >= self.dynamic_threshold]
            print(f"找到 {len(high_conf_pairs)} 个高置信度对齐 (>= {self.dynamic_threshold:.4f}, 动态阈值)")

            # 3.6: 冲突检测与消解
            for (id_a, id_b, conf) in high_conf_pairs:
                self._resolve_conflict(id_a, id_b, conf, new_alignments_this_round)
                        
            # 3.7: 保存本轮结果
            self._save_round_alignments_files(current_round_num, new_alignments_this_round)
            
            # 保存本轮对齐数量到历史记录
            self.previous_round_alignments[current_round_num] = len(new_alignments_this_round)
            
            # 3.8: 每轮结束时进行测试评估
            print(f"第 {current_round_num} 轮结束，开始评估...")
            self._evaluate_current_round(current_round_num)
            
            # 3.9: 收敛判断
            if self._check_convergence(new_alignments_this_round):
                print(f"在第 {current_round_num} 轮收敛。停止迭代。")
                break
            
            round_time = time.time() - start_round_time
            print(f"第 {current_round_num} 轮耗时: {round_time:.2f} 秒")
        
        # 保存最终对齐结果
        final_alignments = self._save_final_alignments()
        
        return final_alignments

    def _init_memory(self, seed_alignments: List[AlignmentPair]):
        """初始化记忆池"""
        for alignment in seed_alignments:
            self.memory[alignment.id_a] = (alignment.id_b, alignment.confidence)
            self.reverse_memory[alignment.id_b] = alignment.id_a
            # 新增：同时存储到种子对齐集合中
            self.seed_memory.add(alignment.id_a)
            self.seed_reverse_memory.add(alignment.id_b)
        print(f"初始化记忆池: {len(self.memory)} 个种子对齐")
    
    def _generate_candidates_with_scores(self, available_tools: List[BaseTool],
                           source_A_entities: List[UnifiedEntity],
                           source_B_entities: List[UnifiedEntity],
                           round_num: int) -> Dict[Tuple[str, str], Dict[str, float]]:
        """生成带置信度的候选对 - 从top-k工具和one-time工具获取"""
        # 使用字典存储候选对及其来自不同工具的初始分数
        # 格式: {(id_a, id_b): {tool_name: score}}
        candidate_scores_map = {}
        
        # 1. 使用top-k工具生成候选并保留置信度
        for tool in available_tools:
            if tool.tool_type == "top-k":
                print(f"使用 {tool.name} 生成带置信度的候选对...")
                
                # 批量处理：收集所有需要查询的实体
                all_queries = []
                query_entities_map = {}  # 记录查询实体对应的源实体
                
                # 收集A实体的查询
                for entity in source_A_entities:
                    if entity.id not in self.memory:
                        all_queries.append(entity.id)
                        query_entities_map[entity.id] = ('A', entity)
                
                # 收集B实体的查询
                for entity in source_B_entities:
                    if entity.id not in self.reverse_memory:
                        all_queries.append(entity.id)
                        query_entities_map[entity.id] = ('B', entity)
                
                if all_queries:
                    # 创建批量查询文件 - 使用传入的round_num参数
                    query_file_path = f"./batch_queries/{self.data_name}_p{int(self.percentage_per_round*100)}_batch_query_round_{round_num}.txt"
                    with open(query_file_path, 'w', encoding='utf-8') as f:
                        for query_id in all_queries:
                            f.write(f"{query_id}\n")
                    
                    # 修复：使用正确的predict_topk方法而不是batch_query
                    batch_results = tool.predict_topk(query_file_path, self.top_k)
                    
                    # 处理批量结果
                    for query_id, candidates in batch_results.items():
                        source_type, source_entity = query_entities_map[query_id]
                        
                        for candidate_id, confidence in candidates:
                            if source_type == 'A':
                                pair = (query_id, candidate_id)
                            else:
                                pair = (candidate_id, query_id)
                            
                            if pair not in candidate_scores_map:
                                candidate_scores_map[pair] = {}
                            candidate_scores_map[pair][tool.name] = confidence
                
                print(f"{tool.name} 生成了 {len(candidate_scores_map)} 个候选对")

        # 2. 使用one-time工具生成候选 - 修复：使用get_one_time_results方法，并筛选本轮相关实体
        # 创建实体ID集合，用于快速查找本轮相关的实体
        source_A_ids = {entity.id for entity in source_A_entities}
        source_B_ids = {entity.id for entity in source_B_entities}
        for tool in available_tools:
            if tool.tool_type == "one-time":
                print(f"使用 {tool.name} 生成候选对...")
                # 修复：使用正确的get_one_time_results方法而不是generate_candidates
                tool_candidates = tool.get_one_time_results()
                
                # 筛选只包含本轮相关实体的候选对
                filtered_candidates_cnt = 0
                for id_a, id_b, confidence in tool_candidates:
                    # 检查候选对的两个实体是否都在本轮处理的实体集合中
                    if id_a in source_A_ids or id_b in source_B_ids:    
                        pair = (id_a, id_b)
                        filtered_candidates_cnt += 1
                        if pair not in candidate_scores_map:
                            candidate_scores_map[pair] = {}
                        candidate_scores_map[pair][tool.name] = confidence
                    
                print(f"{tool.name} 生成了 {len(tool_candidates)} 个候选对，筛选后保留 {filtered_candidates_cnt} 个本轮相关候选对")
        
        return candidate_scores_map

    def _score_candidates_enhanced(self, candidate_scores_map: Dict[Tuple[str, str], Dict[str, float]],
                                  available_tools: List[BaseTool], round_num: int) -> List[Tuple[str, str, float]]:
        """增强版候选打分 - 对候选对使用所有可用工具进行批量打分，并行执行工具打分以提高效率"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        scored_candidates = []
        
        # 创建批量查询目录
        os.makedirs("./batch_scoring", exist_ok=True)
        
        # 收集需要批量打分的候选对
        batch_candidates = []
        for (id_a, id_b), tool_scores in candidate_scores_map.items():
            batch_candidates.append((id_a, id_b))
        
        if not batch_candidates:
            return scored_candidates
        
        # 为每个工具准备批量查询文件，并行执行打分
        tool_tasks = []
        for tool in available_tools:
            # 筛选出当前工具还没有打分的候选对
            candidates_need_scoring = []
            for id_a, id_b in batch_candidates:
                tool_scores = candidate_scores_map.get((id_a, id_b), {})
                if tool.name not in tool_scores:
                    candidates_need_scoring.append((id_a, id_b))
            
            if not candidates_need_scoring:
                print(f"{tool.name} 所有候选对都已打过分，跳过")
                continue
            
            print(f"{tool.name} 需要对 {len(candidates_need_scoring)} 个候选对进行打分")
            
            # 创建批量查询文件
            batch_file_path = f"./batch_scoring/{self.data_name}_p{int(self.percentage_per_round*100)}_{tool.name}_batch_queries_round_{round_num}.txt"
            with open(batch_file_path, 'w', encoding='utf-8') as f:
                for id_a, id_b in candidates_need_scoring:
                    str_id_a = str(id_a) if not isinstance(id_a, str) else id_a
                    str_id_b = str(id_b) if not isinstance(id_b, str) else id_b
                    f.write(f"{str_id_a}\t{str_id_b}\n")
            
            tool_tasks.append((tool, batch_file_path, candidates_need_scoring))
        
        # 并行执行所有工具的批量打分
        def score_tool(tool, batch_file_path, candidates_need_scoring):
            try:
                batch_results = tool.score_batch(batch_file_path)
                print(f"{tool.name} 批量打分完成，处理了 {len(batch_results)} 个候选对")
                return tool.name, batch_results
            except Exception as e:
                print(f"工具 {tool.name} 批量打分失败: {e}")
                return tool.name, {}
        
        # 使用 ThreadPoolExecutor 并行执行所有工具打分
        if tool_tasks:
            max_workers = len(tool_tasks) # 最多8个并行任务
            print(f"并行执行 {len(tool_tasks)} 个工具的打分，最多 {max_workers} 个线程...")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(score_tool, tool, batch_file_path, candidates): (tool, candidates)
                    for tool, batch_file_path, candidates in tool_tasks
                }
                
                for future in as_completed(futures):
                    tool_name, batch_results = future.result()
                    tool, candidates = futures[future]
                    
                    # 处理批量结果
                    for (id_a, id_b), score in batch_results.items():
                        if (id_a, id_b) in candidate_scores_map:
                            candidate_scores_map[(id_a, id_b)][tool_name] = score
                        else:
                            candidate_scores_map[(id_a, id_b)] = {tool_name: score}
        
        # 保存每轮每对实体对所有工具的打分结果
        self._save_round_tool_scores(candidate_scores_map, round_num)
        
        # 使用LLM决定工具权重和阈值（传递available_tools参数）
        print(f"第 {round_num} 轮: 使用LLM分配工具权重...")
        tool_weights, dynamic_threshold = self._get_llm_weight_assignment(candidate_scores_map, round_num, available_tools)
        self.tool_weights = tool_weights
        self.previous_round_weights[round_num] = tool_weights
        self.dynamic_threshold = dynamic_threshold
        
        # 计算每个候选对的加权分数
        for (id_a, id_b), tool_scores in candidate_scores_map.items():
            # 跳过已对齐的实体对
            if id_a in self.memory or id_b in self.reverse_memory:
                continue
            
            # 计算加权分数
            final_score = self._calculate_weighted_score(tool_scores, tool_weights)
            scored_candidates.append((id_a, id_b, final_score))
        
        # 保存权重信息
        self._save_weight_info(tool_weights, dynamic_threshold, round_num)
        
        return scored_candidates


    def _save_weight_info(self, tool_weights: Dict[str, float], threshold: float, round_num: int):
        """保存权重和阈值信息"""
        weight_file = f"./output/{self.data_name}_p{int(self.percentage_per_round*100)}_round_{round_num}_weights.txt"
        
        with open(weight_file, 'w', encoding='utf-8') as f:
            f.write(f"Round {round_num} Weight Assignment\n")
            f.write(f"Dynamic Threshold: {threshold:.4f}\n\n")
            f.write("Tool Weights:\n")
            
            for tool_name, weight in tool_weights.items():
                f.write(f"{tool_name}: {weight:.4f}\n")
            
            f.write(f"\nWeight Sum: {sum(tool_weights.values()):.4f}\n")
        
        print(f"第 {round_num} 轮权重信息已保存到 {weight_file}")

    def _save_round_tool_scores(self, candidate_scores_map: Dict[Tuple[str, str], Dict[str, float]], round_num: int):
        """保存每轮每对实体对所有工具的打分结果"""
        # 确保输出目录存在
        os.makedirs("./output", exist_ok=True)
        
        # 构建实体ID到名称的映射
        a_entity_map = {e.id: e for e in self.source_A_entities} if self.source_A_entities else {}
        b_entity_map = {e.id: e for e in self.source_B_entities} if self.source_B_entities else {}
        
        # 保存详细工具打分结果
        tool_scores_file = f"./output/{self.data_name}_p{int(self.percentage_per_round*100)}_round_{round_num}_tool_scores_detailed.txt"
        
        with open(tool_scores_file, 'w', encoding='utf-8') as f:
            # 写入表头
            f.write("id_a\tname_a\tid_b\tname_b\t")
            
            # 收集所有工具名称
            all_tool_names = set()
            for tool_scores in candidate_scores_map.values():
                all_tool_names.update(tool_scores.keys())
            
            # 写入工具名称作为列头
            tool_names_sorted = sorted(list(all_tool_names))
            f.write("\t".join(tool_names_sorted))
            f.write("\tfinal_score\n")
            
            # 写入每对实体的工具打分结果
            for (id_a, id_b), tool_scores in candidate_scores_map.items():
                # 获取实体名称
                name_a = "N/A"
                if id_a in a_entity_map:
                    name_a = a_entity_map[id_a].name or "N/A"
                elif id_a in b_entity_map:
                    name_a = b_entity_map[id_a].name or "N/A"
                
                name_b = "N/A"
                if id_b in b_entity_map:
                    name_b = b_entity_map[id_b].name or "N/A"
                elif id_b in a_entity_map:
                    name_b = a_entity_map[id_b].name or "N/A"
                
                # 写入实体信息
                f.write(f"{id_a}\t{name_a}\t{id_b}\t{name_b}\t")
                
                # 写入每个工具的打分
                for tool_name in tool_names_sorted:
                    score = tool_scores.get(tool_name, "N/A")
                    if score != "N/A":
                        f.write(f"{score:.4f}\t")
                    else:
                        f.write("N/A\t")
                
                # 计算并写入最终分数
                final_score = self._calculate_weighted_score(tool_scores, self.tool_weights)
                f.write(f"{final_score:.4f}")
                
                f.write("\n")
        
        print(f"第 {round_num} 轮工具打分详细结果已保存到 {tool_scores_file}")
        
        # 保存简化的工具打分结果（仅包含实体ID和最终分数）
        simplified_file = f"./output/{self.data_name}_p{int(self.percentage_per_round*100)}_round_{round_num}_tool_scores_simplified.txt"
        
        with open(simplified_file, 'w', encoding='utf-8') as f:
            f.write("id_a\tid_b\tfinal_score\ttools_used\tweighted_method\n")
            
            for (id_a, id_b), tool_scores in candidate_scores_map.items():
                # 计算最终分数（使用加权分数）
                final_score = self._calculate_weighted_score(tool_scores, self.tool_weights)
                tools_used = ",".join(tool_scores.keys())
                f.write(f"{id_a}\t{id_b}\t{final_score:.4f}\t{tools_used}\tweighted\n")

        
        print(f"第 {round_num} 轮工具打分简化结果已保存到 {simplified_file}")

    def _resolve_conflict(self, id_a: str, id_b: str, confidence: float, new_alignments: List):
        """冲突检测与消解"""
        if confidence < self.dynamic_threshold:
            return
        
        # 检查冲突
        conflict_a = id_a in self.memory
        conflict_b = id_b in self.reverse_memory
        
        if not conflict_a and not conflict_b:
            # 无冲突，直接添加
            self.memory[id_a] = (id_b, confidence)
            self.reverse_memory[id_b] = id_a
            new_alignments.append((id_a, id_b, confidence))
            #print(f"添加新对齐: {id_a} <-> {id_b} (置信度: {confidence:.4f})")
        else:
            # 有冲突，选择置信度更高的对齐
            if conflict_a:
                existing_b, existing_conf = self.memory[id_a]
                if confidence > existing_conf:
                    # 移除旧对齐
                    del self.reverse_memory[existing_b]
                    # 添加新对齐
                    self.memory[id_a] = (id_b, confidence)
                    self.reverse_memory[id_b] = id_a
                    new_alignments.append((id_a, id_b, confidence))
                    print(f"冲突消解: 替换 {id_a} <-> {existing_b} (置信度: {existing_conf:.4f}) 为 {id_a} <-> {id_b} (置信度: {confidence:.4f})")
            
            if conflict_b:
                existing_a = self.reverse_memory[id_b]
                existing_conf = self.memory[existing_a][1]
                if confidence > existing_conf:
                    # 移除旧对齐
                    del self.memory[existing_a]
                    # 添加新对齐
                    self.memory[id_a] = (id_b, confidence)
                    self.reverse_memory[id_b] = id_a
                    new_alignments.append((id_a, id_b, confidence))
                    print(f"冲突消解: 替换 {existing_a} <-> {id_b} (置信度: {existing_conf:.4f}) 为 {id_a} <-> {id_b} (置信度: {confidence:.4f})")

    def _check_convergence(self, new_alignments: List) -> bool:
        """检查是否收敛"""
        if len(new_alignments) <= self.convergence_threshold:
            return True
        return False



    def _parse_weight_output(self, llm_output: str, all_tools: Set[str]) -> Dict[str, float]:
        """解析LLM权重输出"""
        try:
            # 提取JSON部分
            import re
            json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                
                weights = result.get('weights', {})
                
                # 修复：确保所有权重值都是Python原生float类型
                for tool_name, weight in weights.items():
                    if isinstance(weight, (np.float32, np.float64, np.float16)):
                        weights[tool_name] = float(weight)
                    elif isinstance(weight, (np.int32, np.int64, np.int16)):
                        weights[tool_name] = float(weight)
                
                # 验证权重总和是否为1.0
                total_weight = sum(weights.values())
                if abs(total_weight - 1.0) > 0.01:  # 允许微小误差
                    # 重新归一化权重
                    for tool in weights:
                        weights[tool] = float(weights[tool] / total_weight)
                
                # 确保所有工具都有权重
                for tool in all_tools:
                    if tool not in weights:
                        weights[tool] = 0.0
                
                print(f"LLM确定的工具权重: {weights}")
                return weights
            else:
                raise ValueError("未找到有效的JSON输出")
        except Exception as e:
            print(f"解析权重输出失败: {e}, 使用默认权重")
            # 失败时使用默认权重
            return {tool: float(1.0/len(all_tools)) for tool in all_tools}

    def _parse_threshold_output(self, llm_output: str) -> float:
        """解析LLM阈值输出"""
        try:
            # 提取JSON部分
            import re
            json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                
                threshold = result.get('threshold', 0.7)
                
                # 修复：确保阈值是Python原生float类型
                if isinstance(threshold, (np.float32, np.float64, np.float16)):
                    threshold = float(threshold)
                elif isinstance(threshold, (np.int32, np.int64, np.int16)):
                    threshold = float(threshold)
                
                # 确保阈值在合理范围内
                threshold = max(0.0, min(1.0, threshold))
                
                print(f"LLM确定的阈值: {threshold}")
                return threshold
            else:
                raise ValueError("未找到有效的JSON输出")
        except Exception as e:
            print(f"解析阈值输出失败: {e}, 使用默认阈值")
            return float(self.confidence_threshold)

    def _calculate_weighted_score(self, tool_scores: Dict[str, float], tool_weights: Dict[str, float]) -> float:
        """计算加权分数"""
        weighted_score = 0.0
        for tool_name, score in tool_scores.items():
            if score is None:
                continue  # 跳过None值的工具
            weight = tool_weights.get(tool_name, 0.0)
            # 修复：确保score和weight都是Python原生float类型
            if isinstance(score, (np.float32, np.float64, np.float16)):
                score = float(score)
            if isinstance(weight, (np.float32, np.float64, np.float16)):
                weight = float(weight)
            weighted_score += score * weight
        return float(weighted_score)


    def _get_llm_weight_assignment(self, candidate_scores_map: Dict[Tuple[str, str], Dict[str, float]], 
                                 round_num: int, available_tools: List[BaseTool]) -> Tuple[Dict[str, float], float]:
        """使用LLM分步骤决定工具权重和阈值
        
        步骤1: 使用LLM确定工具权重（基于批量打分的已知对齐和候选对齐）
        步骤2: 使用LLM确定最终对齐阈值
        """
        
        # 收集所有工具名称
        all_tools = set()
        for tool_scores in candidate_scores_map.values():
            all_tools.update(tool_scores.keys())
        
        if not all_tools:
            # 如果没有工具，返回默认权重和阈值
            default_weights = {tool: 1.0/len(all_tools) for tool in all_tools} if all_tools else {}
            return default_weights, self.confidence_threshold
        
        # 步骤1: 批量获取已知对齐和候选对齐的打分数据
        print("known_alignments_with_scores")
        known_alignments_with_scores = self._get_batch_scoring_for_known_alignments(round_num, available_tools, 20)
        print("candidate_alignments_with_scores")
        candidate_alignments_with_scores = self._get_batch_scoring_for_candidates(candidate_scores_map, available_tools, 20)
        
        if not known_alignments_with_scores or not candidate_alignments_with_scores:
            # 如果无法获取足够的数据，返回默认权重
            default_weights = {tool: 1.0/len(all_tools) for tool in all_tools}
            return default_weights, self.confidence_threshold
        
        try:
            # 步骤1: 使用LLM确定工具权重
            tool_weights = self._determine_tool_weights_with_batch_scoring(
               known_alignments_with_scores, candidate_alignments_with_scores, all_tools, available_tools)
            
            ##################################
            #tool_weights = {tool: 1.0/len(all_tools) for tool in all_tools}
            if self.warmup and round_num == 1:
                return tool_weights, 100.0

            # 步骤2: 使用LLM确定最终对齐阈值
            threshold = self._determine_alignment_threshold_with_batch_scoring(
                known_alignments_with_scores, candidate_alignments_with_scores, all_tools, tool_weights, available_tools)
            #threshold = 0.6
            
            return tool_weights, threshold
            
        except Exception as e:
            print(f"LLM权重分配失败: {e}, 使用默认权重")
            # 失败时使用默认权重（所有工具权重相等）
            default_weights = {tool: 1.0/len(all_tools) for tool in all_tools}
            return default_weights, self.confidence_threshold



    def _get_batch_scoring_for_known_alignments(self, round_num: int, available_tools: List[BaseTool], count: int = 20) -> List[Tuple[str, str, Dict[str, float]]]:
        """批量获取已知对齐的打分数据"""
        # 获取已知对齐数据        
        known_alignments = []
        seed_alignments_file = f"./output/{self.data_name}_p{int(self.percentage_per_round*100)}_round_0_alignments_all.txt"

        if os.path.exists(seed_alignments_file):
            try:
                with open(seed_alignments_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split('\t')
                            if len(parts) >= 2:
                                id_a = parts[0]
                                id_b = parts[1]
                                known_alignments.append((id_a, id_b))
            except Exception as e:
                print(f"读取种子对齐文件失败: {e}")
        
        # 随机选择指定数量的已知对齐
        if len(known_alignments) > count:
            selected_alignments = random.sample(known_alignments, count)
        else:
            selected_alignments = known_alignments
        
        print(f"从{len(known_alignments)}个已知对齐中随机选择{len(selected_alignments)}个进行打分")
        
        # 创建临时文件，只包含选中的已知对齐
        temp_alignments_file = f"./batch_scoring/{self.data_name}_p{int(self.percentage_per_round*100)}_temp_known_alignments_round_{round_num}.txt"
        with open(temp_alignments_file, 'w', encoding='utf-8') as f:
            for id_a, id_b in selected_alignments:
                f.write(f"{id_a}\t{id_b}\n")
        
        # 使用批量打分获取选中对齐的工具打分
        known_alignments_with_scores = []
        
        # 为每个工具批量获取选中对齐的打分
        tool_batch_results = {}
        for tool in available_tools:
            print(tool.name, tool.tool_type)
            if tool.tool_type == "one-time":
                continue
            try:
                # 使用批量打分方式获取该工具对选中对齐的分数
                batch_results = tool.score_batch(temp_alignments_file)
                tool_batch_results[tool.name] = batch_results
                print(f"工具 {tool.name} 批量打分完成，共获取 {len(batch_results)} 个分数")
            except Exception as e:
                print(f"工具 {tool.name} 批量打分失败: {e}")
                tool_batch_results[tool.name] = {}
        
        # 为每个对齐对收集所有工具的打分
        for id_a, id_b in selected_alignments:
            tool_scores = {}
            valid_tools_count = 0
            all_tools_have_scores = True
            
            for tool in available_tools:
                if tool.tool_type == "one-time":
                    continue
                try:
                    # 从该工具的批量结果中获取当前对齐对的分数
                    score = tool_batch_results[tool.name].get((id_a, id_b))
                    if score is not None:
                        tool_scores[tool.name] = float(score)
                        valid_tools_count += 1
                    else:
                        tool_scores[tool.name] = None
                        all_tools_have_scores = False
                except Exception as e:
                    print(f"工具 {tool.name} 对已知对齐 ({id_a}, {id_b}) 获取分数失败: {e}")
                    tool_scores[tool.name] = None
                    all_tools_have_scores = False
            
            # 收集至少有一半工具有打分的对齐对
            if valid_tools_count >= len(available_tools) // 2:
                known_alignments_with_scores.append((id_a, id_b, tool_scores))
        
        print(f"已知对齐批量打分：筛选出{len(known_alignments_with_scores)}个有效对齐对")
        
        # 删除临时文件
        if os.path.exists(temp_alignments_file):
            os.remove(temp_alignments_file)
        
        return known_alignments_with_scores
 
    
    def _get_batch_scoring_for_candidates(self, candidate_scores_map: Dict[Tuple[str, str], Dict[str, float]], 
                                        available_tools: List[BaseTool], count: int = 20) -> List[Tuple[str, str, Dict[str, float]]]:
        """批量获取候选对齐的打分数据"""
        # 筛选有完整工具打分的候选对（所有工具都有打分）
        complete_candidates = []
        for (id_a, id_b), tool_scores in candidate_scores_map.items():
            # 检查是否所有工具都有打分
            all_tools_have_scores = True
            for tool in available_tools:
                if tool.name not in tool_scores:
                    all_tools_have_scores = False
                    break
            if all_tools_have_scores:
                complete_candidates.append((id_a, id_b, tool_scores))
        
        # 如果完整候选对不足，放宽条件：至少有一半工具有打分
        if len(complete_candidates) < count:
            partial_candidates = []
            for (id_a, id_b), tool_scores in candidate_scores_map.items():
                # 计算有打分的工具数量
                valid_tools_count = sum(1 for tool in available_tools if tool.name in tool_scores)
                # 如果至少有一半工具有打分，则保留
                if valid_tools_count >= len(available_tools) // 2:
                    partial_candidates.append((id_a, id_b, tool_scores))
            
            # 合并完整和部分候选对
            all_candidates = complete_candidates + partial_candidates
            # 去重
            seen = set()
            unique_candidates = []
            for candidate in all_candidates:
                key = (candidate[0], candidate[1])
                if key not in seen:
                    seen.add(key)
                    unique_candidates.append(candidate)
            
            candidate_alignments_with_scores = unique_candidates
        else:
            candidate_alignments_with_scores = complete_candidates
        
        # 随机选择20条候选对齐
        if len(candidate_alignments_with_scores) > count:
            candidate_alignments_with_scores = random.sample(candidate_alignments_with_scores, count)
        
        print(f"批量打分：从{len(candidate_scores_map)}个候选对中筛选出{len(candidate_alignments_with_scores)}个有效候选对")
        return candidate_alignments_with_scores

    def _determine_tool_weights_with_batch_scoring(self, known_alignments: List[Tuple[str, str, Dict[str, float]]], 
                                                 candidate_alignments: List[Tuple[str, str, Dict[str, float]]], 
                                                 all_tools: Set[str], available_tools: List[BaseTool]) -> Dict[str, float]:
        """步骤1: 使用LLM确定工具权重（基于批量打分的已知对齐和候选对齐）"""
        # 构建LLM输入数据
        llm_input = self._build_weight_input_with_batch_scoring(known_alignments, candidate_alignments, all_tools, available_tools)
        
        # 获取LLM配置
        client_config = LLMConfig.get_openai_client_config("weight_assignment")
        llm_config = LLMConfig.get_completion_config("weight_assignment")

        # 创建OpenAI客户端
        client = OpenAI(**client_config)
        
        # 构建权重确定prompt
        tools_list = sorted(list(all_tools))
        prompt = f"""You are an expert in entity alignment. Your task is to determine optimal weights for each alignment tool based on the provided data.

Available tools: {', '.join(tools_list)}

Input data:
{llm_input}

The input contains two types of alignment pairs:
1. KNOWN ALIGNMENTS (20 pairs): Entity pairs that are confirmed to be correctly aligned (from seed alignments or previous rounds)
2. CANDIDATE ALIGNMENTS (20 pairs): Entity pairs that may or may not be aligned (need evaluation)

Instructions:
1. Analyze the score patterns for each tool across both known and candidate alignments
2. Consider tool reliability, consistency, and discriminative power
3. Assign weights to each tool (sum must be exactly 1.0)
4. Higher weights should be given to tools that consistently give high scores to known alignments and low scores to non-alignments
5. Pay special attention to how tools perform on the 20 known alignments vs the 20 candidate alignments

Output format (JSON only):
{{
    "weights": {{
        "tool1": 0.25,
        "tool2": 0,
        ...
    }}
}}

Requirements:
- Weights must sum to exactly 1.0
- All available tools must be included in the weights
- Provide only the JSON output, no explanations"""

        # 调用LLM获取权重
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert in entity alignment and tool weight assignment."},
                {"role": "user", "content": prompt}
            ],
            **llm_config
        )
        print(prompt)
        print(response)
        # 解析权重响应
        llm_output = response.choices[0].message.content.strip()
        return self._parse_weight_output(llm_output, all_tools)

    def _determine_alignment_threshold_with_batch_scoring(self, known_alignments: List[Tuple[str, str, Dict[str, float]]], 
                                                        candidate_alignments: List[Tuple[str, str, Dict[str, float]]], 
                                                        all_tools: Set[str], tool_weights: Dict[str, float],
                                                        available_tools: List[BaseTool]) -> float:
        """步骤2: 使用LLM确定最终对齐阈值（基于批量打分的已知对齐和候选对齐）"""
        # 计算加权分数

        known_scores = []
        candidate_scores = []

        weighted_known_scores = []
        for id_a, id_b, tool_scores in known_alignments:
            weighted_score = self._calculate_weighted_score(tool_scores, tool_weights)
            weighted_known_scores.append((id_a, id_b, tool_scores, weighted_score))
            known_scores.append(weighted_score)
        
        weighted_candidate_scores = []
        for id_a, id_b, tool_scores in candidate_alignments:
            weighted_score = self._calculate_weighted_score(tool_scores, tool_weights)
            weighted_candidate_scores.append((id_a, id_b, tool_scores, weighted_score))
            candidate_scores.append(weighted_score)
        
        stats = f"""
KNOWN ALIGNMENTS (positive samples):
- Count: {len(known_scores)}
- Average score: {sum(known_scores)/len(known_scores) if known_scores else 0:.4f}
- Min score: {min(known_scores) if known_scores else 0:.4f}
- Max score: {max(known_scores) if known_scores else 0:.4f}

CANDIDATE ALIGNMENTS (mixed samples):
- Count: {len(candidate_scores)}
- Average score: {sum(candidate_scores)/len(candidate_scores) if candidate_scores else 0:.4f}
- Min score: {min(candidate_scores) if candidate_scores else 0:.4f}
- Max score: {max(candidate_scores) if candidate_scores else 0:.4f}
"""


        # 构建阈值确定输入
        threshold_input = self._build_threshold_input_with_batch_scoring(
            weighted_known_scores, weighted_candidate_scores, all_tools, tool_weights, available_tools)
        
        # 获取LLM配置
        client_config = LLMConfig.get_openai_client_config("weight_assignment")
        llm_config = LLMConfig.get_completion_config("weight_assignment")

        # 创建OpenAI客户端
        client = OpenAI(**client_config)
        
        # 构建阈值确定prompt
        tools_list = sorted(list(all_tools))
        prompt = f"""You are an expert in entity alignment. Your task is to determine the optimal threshold for alignment decisions based on the provided weighted scores.

Available tools: {', '.join(tools_list)}
Tool weights: {tool_weights}

Input data:
{threshold_input}

Data statistics:
{stats}

The input contains:
1. KNOWN ALIGNMENTS (20 pairs): Entity pairs that are confirmed to be correctly aligned (from seed alignments or previous rounds)
2. CANDIDATE ALIGNMENTS (20 pairs): Entity pairs that may or may not be aligned (need evaluation)
3. Weighted scores calculated using the provided tool weights

Instructions:
1. Analyze the distribution of weighted scores for known alignments vs candidate alignments
2. Determine a threshold (0.0-1.0) that maximizes true positives while minimizes false positives
3. Consider the trade-off between precision and recall
4. The threshold should effectively separate true alignments from non-alignments
5. Pay special attention to the score distribution of the 20 known alignments

Output format (JSON only):
{{
    "threshold": 0.1
}}

Requirements:
- Threshold should be between 0.0 and 1.0
- Provide only the JSON output, no explanations"""

        # 调用LLM获取阈值
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert in entity alignment and threshold determination."},
                {"role": "user", "content": prompt}
            ],
            **llm_config
        )
        print(prompt)
        print(response)
        # 解析阈值响应
        llm_output = response.choices[0].message.content.strip()
        return self._parse_threshold_output(llm_output)












    def _build_weight_input_with_batch_scoring(self, known_alignments: List[Tuple[str, str, Dict[str, float]]], 
                                     candidate_alignments: List[Tuple[str, str, Dict[str, float]]], 
                                     all_tools: Set[str], available_tools: List[BaseTool]) -> str:
        """构建权重确定输入数据（基于批量打分的已知对齐和候选对齐）"""
        alignments_data = []
        
        # 首先添加已知对齐数据
        for id_a, id_b, tool_scores in known_alignments:
            # 获取实体信息
            name_a = "N/A"
            name_b = "N/A"
            relations_a = {}
            relations_b = {}
            attributes_a = {}
            attributes_b = {}
            
            if self.source_A_entities:
                a_entity = next((e for e in self.source_A_entities if e.id == id_a), None)
                if a_entity:
                    if self.data_modal_report.has_names:
                        name_a = a_entity.name or "N/A"
                    if self.data_modal_report.has_relation_names and a_entity.relations:
                        relations_a = a_entity.relations
                    if self.data_modal_report.has_attribute_names and a_entity.attributes:
                        attributes_a = a_entity.attributes
            
            if self.source_B_entities:
                b_entity = next((e for e in self.source_B_entities if e.id == id_b), None)
                if b_entity:
                    if self.data_modal_report.has_names:
                        name_b = b_entity.name or "N/A"
                    if self.data_modal_report.has_relation_names and b_entity.relations:
                        relations_b = b_entity.relations
                    if self.data_modal_report.has_attribute_names and b_entity.attributes:
                        attributes_b = b_entity.attributes
            
            # 构建带工具名称的分数字典
            scores_dict = {}
            for tool in sorted(all_tools):
                score = tool_scores.get(tool)
                if score is None:
                    scores_dict[tool] = None
                else:
                    if isinstance(score, (np.float32, np.float64, np.float16)):
                        score = float(score)
                    scores_dict[tool] = score
            
            # 构建实体对信息（按要求顺序：id, name, relation, attribute, scores）
            entity_info = []
            entity_info.append("{")
            entity_info.append(f"alignment_type: known_alignment")
            entity_info.append(f"id_a: {id_a}")
            entity_info.append(f"id_b: {id_b}")
            
            if self.data_modal_report.has_names:
                entity_info.append(f"name_a: {name_a}")
                entity_info.append(f"name_b: {name_b}")
            
            if self.data_modal_report.has_relation_names:
                if relations_a:
                    entity_info.append(f"relations_a: {relations_a}")
                if relations_b:
                    entity_info.append(f"relations_b: {relations_b}")
            
            # if self.data_modal_report.has_attribute_names:
            #     if attributes_a:
            #         entity_info.append(f"attributes_a: {attributes_a}")
            #     if attributes_b:
            #         entity_info.append(f"attributes_b: {attributes_b}")
            
            entity_info.append(f"scores: {scores_dict}")
            entity_info.append("}")
            alignments_data.append("\n".join(entity_info))
        
        # 然后添加候选对齐数据
        for id_a, id_b, tool_scores in candidate_alignments:
            # 获取实体信息
            name_a = "N/A"
            name_b = "N/A"
            relations_a = {}
            relations_b = {}
            attributes_a = {}
            attributes_b = {}
            
            if self.source_A_entities:
                a_entity = next((e for e in self.source_A_entities if e.id == id_a), None)
                if a_entity:
                    if self.data_modal_report.has_names:
                        name_a = a_entity.name or "N/A"
                    if self.data_modal_report.has_relation_names and a_entity.relations:
                        relations_a = a_entity.relations
                    if self.data_modal_report.has_attribute_names and a_entity.attributes:
                        attributes_a = a_entity.attributes
            
            if self.source_B_entities:
                b_entity = next((e for e in self.source_B_entities if e.id == id_b), None)
                if b_entity:
                    if self.data_modal_report.has_names:
                        name_b = b_entity.name or "N/A"
                    if self.data_modal_report.has_relation_names and b_entity.relations:
                        relations_b = b_entity.relations
                    if self.data_modal_report.has_attribute_names and b_entity.attributes:
                        attributes_b = b_entity.attributes
            
            # 构建带工具名称的分数字典
            scores_dict = {}
            for tool in sorted(all_tools):
                score = tool_scores.get(tool)
                if score is None:
                    scores_dict[tool] = None
                else:
                    if isinstance(score, (np.float32, np.float64, np.float16)):
                        score = float(score)
                    scores_dict[tool] = score
            
            # 构建实体对信息（按要求顺序：id, name, relation, attribute, scores）
            entity_info = []
            entity_info.append("{")
            entity_info.append(f"alignment_type: candidate_alignment")
            entity_info.append(f"id_a:{id_a}")
            entity_info.append(f"id_b:{id_b}")
            
            if self.data_modal_report.has_names:
                entity_info.append(f"name_a:{name_a}")
                entity_info.append(f"name_b:{name_b}")
            
            if self.data_modal_report.has_relation_names:
                if relations_a:
                    entity_info.append(f"relations_a:{relations_a}")
                if relations_b:
                    entity_info.append(f"relations_b:{relations_b}")
            
            # if self.data_modal_report.has_attribute_names:
            #     if attributes_a:
            #         entity_info.append(f"attributes_a:{attributes_a}")
            #     if attributes_b:
            #         entity_info.append(f"attributes_b:{attributes_b}")
            
            entity_info.append(f"scores:{scores_dict}")
            entity_info.append("}")

            alignments_data.append("\n".join(entity_info) + "\n")
        
        # 构建输入格式
        input_data = "\n".join(alignments_data)

        # input_data = input_data.replace("'", "")
        # print(input_data)
        return input_data
        

    def _build_threshold_input_with_batch_scoring(self, weighted_known_scores: List[Tuple[str, str, Dict[str, float], float]], 
                                        weighted_candidate_scores: List[Tuple[str, str, Dict[str, float], float]], 
                                        all_tools: Set[str], tool_weights: Dict[str, float],
                                        available_tools: List[BaseTool]) -> str:
        """构建阈值确定输入数据（基于批量打分的已知对齐和候选对齐）"""
        alignments_data = []
        
        # 首先添加已知对齐数据
        for id_a, id_b, tool_scores, weighted_score in weighted_known_scores:
            # 获取实体信息
            name_a = "N/A"
            name_b = "N/A"
            relations_a = {}
            relations_b = {}
            attributes_a = {}
            attributes_b = {}
            
            if self.source_A_entities:
                a_entity = next((e for e in self.source_A_entities if e.id == id_a), None)
                if a_entity:
                    if self.data_modal_report.has_names:
                        name_a = a_entity.name or "N/A"
                    if self.data_modal_report.has_relation_names and a_entity.relations:
                        relations_a = a_entity.relations
                    if self.data_modal_report.has_attribute_names and a_entity.attributes:
                        attributes_a = a_entity.attributes
            
            if self.source_B_entities:
                b_entity = next((e for e in self.source_B_entities if e.id == id_b), None)
                if b_entity:
                    if self.data_modal_report.has_names:
                        name_b = b_entity.name or "N/A"
                    if self.data_modal_report.has_relation_names and b_entity.relations:
                        relations_b = b_entity.relations
                    if self.data_modal_report.has_attribute_names and b_entity.attributes:
                        attributes_b = b_entity.attributes
            
            # 构建工具分数字典
            scores_dict = {}
            for tool in sorted(all_tools):
                score = tool_scores.get(tool)
                if score is None:
                    scores_dict[tool] = None
                else:
                    if isinstance(score, (np.float32, np.float64, np.float16)):
                        score = float(score)
                    scores_dict[tool] = score
            
            # 确保weighted_score是Python原生float类型
            if isinstance(weighted_score, (np.float32, np.float64, np.float16)):
                weighted_score = float(weighted_score)
            
            # 构建实体对信息（按要求顺序：id, name, relation, attribute, scores, weighted_score）
            entity_info = []
            entity_info.append("{")
            entity_info.append(f"alignment_type: known_alignment")
            entity_info.append(f"id_a: {id_a}")
            entity_info.append(f"id_b: {id_b}")
            
            if self.data_modal_report.has_names:
                entity_info.append(f"name_a: {name_a}")
                entity_info.append(f"name_b: {name_b}")
            
            if self.data_modal_report.has_relation_names:
                if relations_a:
                    entity_info.append(f"relations_a: {relations_a}")
                if relations_b:
                    entity_info.append(f"relations_b: {relations_b}")
            
            # if self.data_modal_report.has_attribute_names:
            #     if attributes_a:
            #         entity_info.append(f"attributes_a: {attributes_a}")
            #     if attributes_b:
            #         entity_info.append(f"attributes_b: {attributes_b}")
            
            entity_info.append(f"scores: {scores_dict}")
            entity_info.append(f"weighted_score: {weighted_score}")
            entity_info.append("}")
            alignments_data.append("\n".join(entity_info) + "\n")
        
        # 然后添加候选对齐数据
        for id_a, id_b, tool_scores, weighted_score in weighted_candidate_scores:
            # 获取实体信息
            name_a = "N/A"
            name_b = "N/A"
            relations_a = {}
            relations_b = {}
            attributes_a = {}
            attributes_b = {}
            
            if self.source_A_entities:
                a_entity = next((e for e in self.source_A_entities if e.id == id_a), None)
                if a_entity:
                    if self.data_modal_report.has_names:
                        name_a = a_entity.name or "N/A"
                    if self.data_modal_report.has_relation_names and a_entity.relations:
                        relations_a = a_entity.relations
                    if self.data_modal_report.has_attribute_names and a_entity.attributes:
                        attributes_a = a_entity.attributes
            
            if self.source_B_entities:
                b_entity = next((e for e in self.source_B_entities if e.id == id_b), None)
                if b_entity:
                    if self.data_modal_report.has_names:
                        name_b = b_entity.name or "N/A"
                    if self.data_modal_report.has_relation_names and b_entity.relations:
                        relations_b = b_entity.relations
                    if self.data_modal_report.has_attribute_names and b_entity.attributes:
                        attributes_b = b_entity.attributes
            
            # 构建工具分数字典
            scores_dict = {}
            for tool in sorted(all_tools):
                score = tool_scores.get(tool)
                if score is None:
                    scores_dict[tool] = None
                else:
                    if isinstance(score, (np.float32, np.float64, np.float16)):
                        score = float(score)
                    scores_dict[tool] = score
            
            # 确保weighted_score是Python原生float类型
            if isinstance(weighted_score, (np.float32, np.float64, np.float16)):
                weighted_score = float(weighted_score)
            
            # 构建实体对信息（按要求顺序：id, name, relation, attribute, scores, weighted_score）
            entity_info = []
            entity_info.append("{")
            entity_info.append(f"alignment_type: candidate_alignment")
            entity_info.append(f"id_a: {id_a}")
            entity_info.append(f"id_b: {id_b}")
            
            if self.data_modal_report.has_names:
                entity_info.append(f"name_a: {name_a}")
                entity_info.append(f"name_b: {name_b}")
            
            if self.data_modal_report.has_relation_names:
                if relations_a:
                    entity_info.append(f"relations_a: {relations_a}")
                if relations_b:
                    entity_info.append(f"relations_b: {relations_b}")
            
            # if self.data_modal_report.has_attribute_names:
            #     if attributes_a:
            #         entity_info.append(f"attributes_a: {attributes_a}")
            #     if attributes_b:
            #         entity_info.append(f"attributes_b: {attributes_b}")
            
            entity_info.append(f"scores: {scores_dict}")
            entity_info.append(f"weighted_score: {weighted_score}")
            entity_info.append("}")
            alignments_data.append("\n".join(entity_info) + "\n")
        
        # 构建输入格式
        input_data = "\n".join(alignments_data)
        
        return input_data














    def _determine_tools_for_round(self, round_num: int, available_tools: List[BaseTool]) -> List[BaseTool]:
        """使用LLM决定当前轮次要使用的工具"""
        
        # 构建工具介绍信息
        tool_descriptions = []
        for tool in available_tools:
            tool_descriptions.append({
                'name': tool.name,
                'type': tool.tool_type,
                'description': tool.description if hasattr(tool, 'description') else 'No description available'
            })
        
        # 构建历史轮次信息
        round_history = []
        for r in range(1, round_num):  # 从第1轮到当前轮次的前一轮
            if r in self.previous_round_tools and r in self.previous_round_weights and r in self.previous_round_alignments:
                round_info = {
                    'round': r,
                    'tools_used': self.previous_round_tools[r],
                    'tool_weights': self.previous_round_weights[r],
                    'alignments_found': "warn_up round" if r == 1 else self.previous_round_alignments[r]
                }
                round_history.append(round_info)
        
        # 构建LLM输入
        llm_input = {
            'current_round': round_num,
            'available_tools': tool_descriptions,
            'previous_rounds': round_history,
            'total_available_tools': len(available_tools)
        }
        
        # 获取LLM配置
        client_config = LLMConfig.get_openai_client_config("tool_selection")
        llm_config = LLMConfig.get_completion_config("tool_selection")

        # 创建OpenAI客户端
        client = OpenAI(**client_config)
        
        # 构建工具选择prompt
        prompt = f"""You are an expert in entity alignment tool selection. Your task is to determine which tools to use in the current round of alignment.

Current Round: {round_num}
Total Available Tools: {len(available_tools)}

Available Tools Information:
{json.dumps(tool_descriptions, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)}

Previous Rounds History:
{json.dumps(round_history, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)}

Instructions:
1. Analyze the performance of tools in previous rounds based on their weights and alignment results
2. Consider tool types (top-k vs one-time) and their characteristics
3. Select the most appropriate tools for the current round
4. For early rounds (1-3), consider using more tools to gather comprehensive data
5. For later rounds (4+), focus on the most effective tools
6. Balance between exploration (trying new tools) and exploitation (using proven tools)

Output format (JSON only):
{{
    "selected_tools": ["tool_name1", "tool_name2", ...],
    "reasoning": "Brief explanation of the selection strategy"
}}

Requirements:
- Selected tools must be from the available tools list
- Provide only the JSON output, no explanations"""

        prompt = f"""You are an expert in entity alignment tool selection. Your task is to determine which tools to use in the current round of alignment.

Current Round: {round_num}
Total Available Tools: {len(available_tools)}

Available Tools Information:
{json.dumps(tool_descriptions, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)}

Previous Rounds History:
{json.dumps(round_history, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)}

Instructions:
1. Analyze the weights of all tools and identify those with exceptionally low weights compared to others
2. Delete tools whose weights are exceptionally lower than the majority of other tools 

Output format (JSON only):
{{
    "selected_tools": ["tool_name1", "tool_name2", ...],
    "reasoning": "Brief explanation of the selection strategy"
}}

Requirements:
- Selected tools must be from the available tools list
- Provide only the JSON output, no explanations"""

        # 调用LLM获取工具选择
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert in entity alignment tool selection."},
                {"role": "user", "content": prompt}
            ],
            **llm_config
        )
        print(prompt)
        print(response)
        # 解析LLM响应
        llm_output = response.choices[0].message.content.strip()
        return self._parse_tool_selection_output(llm_output, available_tools)

    def _parse_tool_selection_output(self, llm_output: str, available_tools: List[BaseTool]) -> List[BaseTool]:
        """解析LLM的工具选择输出"""
        try:
            # 提取JSON部分
            json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                
                selected_tool_names = data.get('selected_tools', [])
                reasoning = data.get('reasoning', 'No reasoning provided')
                
                print(f"LLM工具选择结果: {selected_tool_names}")
                print(f"选择理由: {reasoning}")
                
                # 根据工具名称获取对应的工具对象
                tool_name_map = {tool.name: tool for tool in available_tools}
                selected_tools = []
                
                for tool_name in selected_tool_names:
                    if tool_name in tool_name_map:
                        selected_tools.append(tool_name_map[tool_name])
                    else:
                        print(f"警告: 工具 '{tool_name}' 不存在，跳过")
                
                if not selected_tools:
                    print("警告: LLM没有选择任何有效工具，使用所有可用工具")
                    return available_tools
                
                return selected_tools
                
            else:
                print("警告: 无法从LLM响应中提取JSON，使用所有可用工具")
                return available_tools
                
        except Exception as e:
            print(f"解析LLM工具选择输出失败: {e}，使用所有可用工具")
            return available_tools


    def _generate_unsupervised_seeds(self, source_A_entities: List[UnifiedEntity], 
                                   source_B_entities: List[UnifiedEntity],
                                   available_tools: List[BaseTool]) -> List[AlignmentPair]:
        """无监督模式下使用tool.supervision=False的工具及LLM辅助生成高置信度伪对齐作为种子"""