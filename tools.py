import os
import subprocess
import json
import time
import signal
import psutil
from typing import List, Dict, Any, Optional, Tuple, Set
from abc import ABC
import numpy as np
import requests
from transformers import AutoModel, AutoTokenizer
import torch
from openai import OpenAI
from data_loader import UnifiedEntity
from torch import cuda
from LLMConfig import LLMConfig

class BaseTool(ABC):
    """工具基类"""
    def __init__(self, name: str, tool_type: str):
        self.name = name
        self.tool_type = tool_type  # "one-time", "top-k", "single", "llm", "direct"
        self.source_A_entities = None
        self.source_B_entities = None
        self._entity_id_to_info = {}  # 用于快速查找实体信息
        self.has_init_sh = False  # 是否有init.sh脚本
        self.has_predict_sh = False  # 是否有predict.sh脚本
        self.has_score_sh = False  # 是否有score.sh脚本
        self.tool_path = os.path.join("./tools", self.name)  # 工具目录路径
        self.init_process_group = None  # 存储init.sh启动的进程组ID
        self.supervision = False # 是否需要监督学习
        self._cache = {}  # 用于缓存无监督工具预测结果
        self.seed_aligns_path = ""
        self._current_round_alignments = {}
        # 检测是否有init.sh和predict.sh
        self._detect_scripts()
        self.current_round = 0  # 当前轮次
        
    
    def set_current_round(self, round_num: int):
        """设置当前轮次并加载对应的对齐文件"""
        self.current_round = round_num

    def _load_current_round_alignments(self):
        """加载当前轮次的对齐文件"""
        self._current_round_alignments = {}
        if not os.path.exists(self.seed_aligns_path):
            print(f"警告: 当前轮次对齐文件不存在: {self.seed_aligns_path}")
            return
        
        try:
            with open(self.seed_aligns_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            id_a = parts[0]
                            id_b = parts[1]
                            self._current_round_alignments[id_a] = id_b
                            # 同时存储反向映射，方便查找
                            self._current_round_alignments[id_b] = id_a
            
            print(f"成功加载第{self.current_round}轮对齐信息，共{len(self._current_round_alignments)//2}个对齐")
        except Exception as e:
            print(f"加载对齐文件失败: {e}")

    def _detect_scripts(self):
        """检测工具目录下是否有init.sh, predict.sh 和 score.sh 脚本"""
        if os.path.exists(self.tool_path):
            self.has_init_sh = os.path.exists(os.path.join(self.tool_path, "init.sh"))
            self.has_predict_sh = os.path.exists(os.path.join(self.tool_path, "predict.sh"))
            self.has_score_sh = os.path.exists(os.path.join(self.tool_path, "score.sh"))
    
    def set_entities(self, source_A_entities: List[UnifiedEntity], source_B_entities: List[UnifiedEntity]):
        """设置实体信息"""
        self.source_A_entities = source_A_entities
        self.source_B_entities = source_B_entities
        
        # 构建实体ID到实体对象的映射
        for entity in source_A_entities + source_B_entities:
            self._entity_id_to_info[entity.id] = entity
    
    def get_entity(self, entity_id: str) -> Optional[UnifiedEntity]:
        """根据实体ID获取实体对象"""
        return self._entity_id_to_info.get(entity_id)
    
    # def score_batch(self, batch_file_path: str) -> Dict[Tuple[str, str], float]:
    #     """批量预测多个实体对的置信度
        
    #     参数:
    #         batch_file_path: 批量查询文件路径，每行格式为"id_a\tid_b"
            
    #     返回:
    #         字典: {(id_a, id_b): confidence_score}
    #     """
    #     batch_results = {}
    #     # 读取批量查询文件
    #     try:
    #         if self.has_score_sh:
    #             batch_results = self.score(batch_file_path)
    #         else:

    #             with open(batch_file_path, 'r', encoding='utf-8') as f:
    #                 for line in f:
    #                     line = line.strip()
    #                     if not line:
    #                         continue
                        
    #                     parts = line.split('\t')
    #                     if len(parts) == 2:
    #                         id_a = parts[0]
    #                         id_b = parts[1]
    #                         #print(id_a, id_b)
    #                         # 使用现有的predict方法进行预测
    #                         score = self.score(id_a, id_b)
    #                         if score is not None:
    #                             batch_results[(id_a, id_b)] = score
    #     except Exception as e:
    #         print(f"读取批量查询文件失败: {e}")
        
    #     return batch_results
    

    def get_one_time_results(self) -> List[Tuple[str, str, str, float]]:
        """
        读取one-time工具的结果文件并返回格式化的结果
        
        返回:
            包含(id_a, id_b, score)元组的列表
        """

        self._detect_scripts()


        # 首先检查是否是one-time工具
        if self.tool_type != "one-time":
            print(f"警告: {self.name}不是one-time工具，无法获取一次性结果")
            return []
        
        results_file = os.path.join(self.tool_path, "results.txt")
        if not os.path.exists(results_file):
            print(f"警告: {self.name}的结果文件不存在: {results_file}")
            return []
        
        results = []
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 解析格式为"{id_a}\t{id_b}\t{score}"的行
                    try:
                        parts = line.strip().split('\t')
                        if len(parts) == 3:
                            id_a = parts[0]
                            id_b = parts[1]
                            score = float(parts[2])
                            # 添加到结果列表
                            results.append((id_a, id_b, score))
                        else:
                            print(f"警告: 忽略格式不正确的行: {line}")
                    except Exception as e:
                        print(f"解析行失败 '{line}': {e}")
        except Exception as e:
            print(f"读取结果文件失败: {e}")
        
        return results

    
    def _get_child_processes(self, parent_pid: int) -> Set[int]:
        """获取父进程的所有子进程ID"""
        try:
            parent = psutil.Process(parent_pid)
            children = parent.children(recursive=True)
            return {child.pid for child in children}
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return set()
    
    def _wait_for_processes(self, processes: Set[int], timeout: int = 3600) -> bool:
        """等待所有进程完成，返回是否全部完成"""
        start_time = time.time()
        remaining_processes = processes.copy()
        
        while remaining_processes and (time.time() - start_time < timeout):
            current_remaining = set()
            for pid in remaining_processes:
                try:
                    process = psutil.Process(pid)
                    if process.is_running():
                        current_remaining.add(pid)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass  # 进程已终止
            
            remaining_processes = current_remaining
            if remaining_processes:
                print(f"{self.name}: 等待{len(remaining_processes)}个子进程完成...")
                time.sleep(5)  # 每5秒检查一次
        
        if remaining_processes:
            print(f"{self.name}: 超时，仍有{len(remaining_processes)}个子进程未完成")
            return False
        
        print(f"{self.name}: 所有子进程已完成")
        return True
    
    def run_preprocessing(self, data_path: str, seed_aligns_path: Optional[str] = None):
        """预处理（检测并运行init.sh，并等待其启动的所有子进程完成）
        
        Args:
            data_path: 数据路径
            seed_aligns_path: 当前轮次的对齐信息路径
        """
        
        # 重新检测脚本
        self._detect_scripts()
        self.seed_aligns_path = seed_aligns_path
        self._load_current_round_alignments()
        if self.supervision:
            self._cache = {} 

        # 检查是否存在init.sh
        if self.has_init_sh:
            init_script = os.path.join(self.tool_path, "init.sh")
            print(f"{self.name}: 检测到init.sh，开始预处理...")
            print(f"{self.name}: 数据路径: {data_path}")
            if seed_aligns_path:
                print(f"{self.name}: 种子对齐路径: {seed_aligns_path}")
            
            try:
                st = os.stat(init_script)
                if not st.st_mode & 0o111:  # 检查是否有任何执行权限
                    print(f"{self.name}: 添加执行权限...")
                    os.chmod(init_script, st.st_mode | 0o111)
                # 准备命令参数
                cmd = [init_script]
                
                # 添加DATA_DIR参数
                cmd.extend(["--data_dir", data_path])
                
                # 如果有seed_aligns_path，添加SEED_ALIGNS参数
                if seed_aligns_path and os.path.exists(seed_aligns_path):
                    cmd.extend(["--seed_aligns", seed_aligns_path])
                
                # 在新的进程组中运行init.sh，以便能够跟踪其所有子进程
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid  # 创建新的进程组
                )
                
                # 获取父进程ID
                parent_pid = process.pid
                print(f"{self.name}: 启动init.sh，进程ID: {parent_pid}")
                
                # 等待init.sh主进程完成
                stdout, stderr = process.communicate()
                return_code = process.returncode
                
                # 记录输出
                if stdout:
                    print(f"{self.name}: init.sh标准输出: {stdout.decode('utf-8', errors='ignore').strip()}")
                if stderr:
                    print(f"{self.name}: init.sh标准错误: {stderr.decode('utf-8', errors='ignore').strip()}")
                
                # 检查是否成功
                if return_code != 0:
                    print(f"{self.name}: init.sh返回错误码: {return_code}")
                
                # 获取并等待所有子进程完成
                child_processes = self._get_child_processes(parent_pid)
                if child_processes:
                    print(f"{self.name}: 检测到{len(child_processes)}个子进程，等待它们完成...")
                    self._wait_for_processes(child_processes)
                
                print(f"{self.name}: 预处理完成")
                
            except subprocess.CalledProcessError as e:
                print(f"{self.name}: 预处理失败，返回码: {e.returncode}")
                if e.stderr:
                    print(f"{self.name}: 错误输出: {e.stderr.decode('utf-8', errors='ignore')}")
            except Exception as e:
                print(f"{self.name}: 预处理异常: {str(e)}")
        else:
            print(f"{self.name}: 未检测到init.sh，进行python预处理或者跳过预处理")

    def score_batch(self, batch_file_path: str) -> Dict[Tuple[str, str], float]:
        """为多个实体对打分
        
        参数:
            batch_file_path: 批量查询文件路径，每行格式为"id_a\tid_b"
            
        返回:
            字典: {(id_a, id_b): score}
        """

        self._detect_scripts()

        # 如果有score.sh脚本，则调用它
        batch_results = {}

        if self.has_score_sh:
            score_script = os.path.join(self.tool_path, "score.sh")
            output_file = os.path.join(self.tool_path, "results.txt")
            
            try:
                # 确保score.sh有执行权限
                st = os.stat(score_script)
                if not st.st_mode & 0o111:
                    os.chmod(score_script, st.st_mode | 0o111)
                
                # 构建命令
                subprocess.run([
                    score_script,
                    "--query_file", batch_file_path,
                    "--output_file", output_file
                ], check=True)

                # 读取输出文件
                
                with open(output_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        content = line.strip()
                        parts = content.split('\t')
                        if len(parts) == 3:
                            id_a = parts[0]
                            id_b = parts[1]
                            try:
                                score = float(parts[2])
                                if self.name == "DualAMN_EA":
                                    score = (max(0.0, min(1.0, (score + 1) / 2)))
                                batch_results[(id_a, id_b)] = score
                            except ValueError:
                                print(f"警告: 无法将分数转换为浮点数: {parts[2]}")
                        else:
                            print(f"警告: 忽略格式不正确的行: {line}")
            except Exception as e:
                print(f"执行score.sh失败: {e}")
                assert 0
        else:
            with open(batch_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('\t')
                    if len(parts) == 2:
                        id_a = parts[0]
                        id_b = parts[1]
                        score = self.score(id_a, id_b)
                        if score is not None:
                            batch_results[(id_a, id_b)] = score
        return batch_results

    def predict_topk(self, query_file_path: str, top_k: int) -> Dict[str, List[Tuple[str, float]]]:
        """批量预测多个实体的最相似实体"""
        output_file = os.path.join(self.tool_path, "results.txt")

        # 检查工具目录是否存在
        if not os.path.exists(self.tool_path):
            print(f"工具路径不存在: {self.tool_path}")
            return {}
        
        # 优先使用predict.sh脚本进行批量查询
        if self.has_predict_sh:
            predict_script = os.path.join(self.tool_path, "predict.sh")

            st = os.stat(predict_script)
            if not st.st_mode & 0o111:  # 检查是否有任何执行权限
                print(f"{self.name}: 添加执行权限...")
                os.chmod(predict_script, st.st_mode | 0o111)

            print(f"{self.name}: 调用predict.sh脚本进行批量查询...")
            try:
                subprocess.run([
                    predict_script,
                    "--query_file", query_file_path,
                    "--output_file", output_file,
                    "--top_k", str(top_k)
                ], check=True)
                
                # 读取并解析批量结果
                return self._parse_top_k_batch_results_file(output_file)
            except Exception as e:
                print(f"调用predict.sh批量查询失败: {e}")
                return {}
        else:
            print(f"predict.sh脚本不存在: {os.path.join(self.tool_path, 'predict.sh')}")
            return {}


    def _parse_top_k_batch_results_file(self, results_file: str) -> Dict[str, List[Tuple[str, float]]]:
        results = {}
        
        if not os.path.exists(results_file):
            print(f"结果文件不存在: {results_file}")
            return results
        
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        parts = line.strip().split('\t')
                        if len(parts) >= 3:
                            entity_id = parts[0]
                            target_id = parts[1]
                            score = float(parts[2])
 
                            if entity_id not in results:
                                results[entity_id] = []
                            results[entity_id].append((target_id, score))
                    except Exception as e:
                        print(f"解析行失败 '{line}': {e}")
        except Exception as e:
            print(f"读取结果文件失败: {e}")
        return results


class ToolsManager:
    """工具管理器"""
    def __init__(self):
        self.tools = []
    
    def get_available_tools(self, data_modal_report, data_path: str,
                          source_A_entities: List[UnifiedEntity] = None,
                          source_B_entities: List[UnifiedEntity] = None,
                          seed_aligns_path: Optional[str] = None) -> List[BaseTool]:
        """根据数据模态报告获取可用工具
        
        Args:
            data_modal_report: 数据模态报告
            data_path: 数据路径
            source_A_entities: 源A的实体列表
            source_B_entities: 源B的实体列表
            seed_aligns_path: 当前轮次的对齐信息路径
            
        Returns:
            可用工具列表
        """
        # 创建工具实例
        tools_to_create = []
        
        # 通用工具
        # if data_modal_report.has_relations:
        #     tools_to_create.append(NeighborhoodSimilarity_EA_Tool())
        # if data_modal_report.has_relations or data_modal_report.has_attributes:
        #     tools_to_create.append(InfoDensitySimilarityTool())

        if data_modal_report.has_names:
            tools_to_create.append(NameSimilarityTool())
            #tools_to_create.append(NameSimilarity2Tool())
            tools_to_create.append(OntologySimilarityTool())
        
        # if data_modal_report.has_attribute_names:
        #     tools_to_create.append(AttributeSimilarityTool())
        
        # if data_modal_report.has_relation_names:
        #     tools_to_create.append(RelationSimilarityTool())
        
        
        # # EA专用工具
        if data_modal_report.is_ea_task:

            tools_to_create.append(MTransE_EA_Tool())
            #tools_to_create.append(BootEA_EA_Tool())
            #tools_to_create.append(DistMult_EA_Tool())
            tools_to_create.append(TransEdge_EA_Tool())
            tools_to_create.append(GCN_Align_EA_Tool())


            tools_to_create.append(DualAMN_EA_Tool())
            #if data_modal_report.has_names:
            tools_to_create.append(ParisPlus_EA_Tool())
            if data_modal_report.has_names and data_modal_report.has_relation_names and data_modal_report.has_attribute_names:
                tools_to_create.append(BERTINT_EA_Tool()) 
 

        #tools_to_create.append(MTransE_EA_Tool())
        # 设置实体信息
        for tool in tools_to_create:
            if source_A_entities and source_B_entities:
                tool.set_entities(source_A_entities, source_B_entities)
    
        return tools_to_create


class NameSimilarityTool(BaseTool):
    """名称相似度工具1（Top-k查询工具）"""
    def __init__(self):
        super().__init__("NameSimilarityTool", "top-k")
        self.description = "Calculates semantic similarity of entity names using paraphrase-multilingual-MiniLM-L12-v2 model. Converts names to vector representations through pre-trained language models and computes cosine similarity to evaluate semantic similarity between names."
        self._model = None
        self._tokenizer = None
        self._embeddings = {}  # {entity_id: embedding}
        self._embeddings_file = None  # 延迟初始化，将在run_preprocessing中设置
        self._device = 'cuda' if cuda.is_available() else 'cpu'
    
    def run_preprocessing(self, data_path: str, seed_aligns_path: Optional[str] = None):
        """加载模型和预计算嵌入，并保存到文件（文件名包含数据集名称）"""
        # 调用基类的预处理方法（检测并运行init.sh）
        super().run_preprocessing(data_path, seed_aligns_path)
        
        # 从data_path中提取数据集名称
        dataset_name = os.path.basename(data_path.rstrip('/'))
        
        # 设置包含数据集名称的embedding文件路径
        self._embeddings_file = os.path.join(self.tool_path, f"embeddings_{dataset_name}.npz")
        
        # 先检查是否存在对应的embedding文件
        if os.path.exists(self._embeddings_file):
            print(f"NameSimilarityTool: 检测到数据集 {dataset_name} 的embedding文件，直接加载...")
            try:
                # 加载保存的embedding
                data = np.load(self._embeddings_file)
                self._embeddings = {key: data[key] for key in data.files}
                print(f"NameSimilarityTool: 成功加载{len(self._embeddings)}个实体的embedding")
                return  # 已加载成功，直接返回
            except Exception as e:
                print(f"NameSimilarityTool: 加载embedding文件失败: {e}")
                print("NameSimilarityTool: 将重新生成embedding...")
        
        try:
            # 尝试加载模型
            try:
                self._tokenizer = AutoTokenizer.from_pretrained("./tools/paraphrase-multilingual-MiniLM-L12-v2")
                self._model = AutoModel.from_pretrained("./tools/paraphrase-multilingual-MiniLM-L12-v2")
                self._model = self._model.to(self._device)  # 将模型移动到GPU
            except Exception as e:
                print(f"NameSimilarityTool: 无法加载预训练模型，将使用简单的文本相似度计算: {e}")
                return
            
            # 如果有实体信息，预计算嵌入
            if self.source_A_entities and self.source_B_entities:
                print(f"NameSimilarityTool: 为数据集 {dataset_name} 生成embedding...")
                all_entities = self.source_A_entities + self.source_B_entities
                for entity in all_entities:
                    if entity.name:
                        # 计算嵌入
                        inputs = self._tokenizer(entity.name, return_tensors='pt', truncation=True, max_length=128)
                        inputs = {k: v.to(self._device) for k, v in inputs.items()}  # 将输入数据移动到GPU
                        with torch.no_grad():
                            outputs = self._model(**inputs)
                            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()  # 将结果移回CPU
                            self._embeddings[entity.id] = embedding
                
                # 确保工具目录存在
                os.makedirs(self.tool_path, exist_ok=True)
                
                # 保存embedding到文件（包含数据集名称）
                print(f"NameSimilarityTool: 保存embedding到文件 {self._embeddings_file}")
                np.savez(self._embeddings_file, **self._embeddings)
                print(f"NameSimilarityTool: 成功保存{len(self._embeddings)}个实体的embedding")
        except Exception as e:
            print(f"NameSimilarityTool预处理失败: {e}")

    def predict_topk(self, query_file_path: str, top_k: int) -> Dict[str, List[Tuple[str, float]]]:
        """批量预测多个实体的top-k相似实体
        
        参数:
            query_file_path: 查询文件路径，每行一个实体ID
            top_k: 返回的top-k结果数量
            
        返回:
            字典: {entity_id: [(target_id, score), ...]}
        """
        # 尝试加载embedding文件
        if not self._embeddings and os.path.exists(self._embeddings_file):
            print(f"NameSimilarityTool: 从文件加载embedding {self._embeddings_file}")
            try:
                data = np.load(self._embeddings_file)
                self._embeddings = {key: data[key] for key in data.files}
                print(f"NameSimilarityTool: 成功加载{len(self._embeddings)}个实体的embedding")
            except Exception as e:
                print(f"NameSimilarityTool: 加载embedding文件失败: {e}")
        
        # 读取查询文件
        query_entities = []
        try:
            with open(query_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        query_entities.append(line)
        except Exception as e:
            print(f"NameSimilarityTool: 读取查询文件失败: {e}")
            return {}
        if not query_entities:
            print("NameSimilarityTool: 查询文件中没有有效的实体ID")
            return {}
        
        print(f"NameSimilarityTool: 开始批量处理 {len(query_entities)} 个查询实体")
        
        results = {}
        
        # 分离源实体和目标实体
        src_queries = []
        tgt_queries = []
        
        for entity_id in query_entities:
            entity = self.get_entity(entity_id)
            if not entity or not entity.name:
                continue
                
            # 判断实体来源
            if entity in self.source_A_entities:
                src_queries.append(entity_id)
            elif entity in self.source_B_entities:
                tgt_queries.append(entity_id)
        
        # 批量处理源语言到目标语言的预测
        if src_queries and self.source_B_entities:
            print(f"NameSimilarityTool: 处理 {len(src_queries)} 个源实体查询")
            
            # 准备目标实体embedding
            target_entities = self.source_B_entities
            target_ids = []
            target_embeddings = []
            for target_entity in target_entities:
                if target_entity.id in self._embeddings:
                    target_ids.append(target_entity.id)
                    target_embeddings.append(self._embeddings[target_entity.id])
            
            if target_embeddings:
                # 将目标embedding转换为tensor并移到GPU
                target_embeddings_tensor = torch.from_numpy(np.stack(target_embeddings)).to(self._device)
                
                # 批量处理源实体查询
                for entity_id in src_queries:
                    if entity_id not in self._embeddings:
                        continue
                        
                    try:
                        # 获取查询实体的embedding
                        embedding_a = torch.from_numpy(self._embeddings[entity_id]).to(self._device)
                        embedding_a = embedding_a.unsqueeze(0)  # 增加batch维度
                        
                        # 批量计算余弦相似度
                        cos = torch.nn.CosineSimilarity(dim=1)
                        similarities = cos(embedding_a, target_embeddings_tensor)
                        
                        # 转换到[0,1]范围并转换为numpy
                        similarities = ((similarities + 1) / 2).cpu().numpy()
                        
                        # 构建结果列表
                        entity_results = list(zip(target_ids, similarities))
                        entity_results.sort(key=lambda x: x[1], reverse=True)
                        results[entity_id] = entity_results[:top_k]
                        
                    except Exception as e:
                        print(f"NameSimilarityTool: 处理实体 {entity_id} 失败: {e}")
                        results[entity_id] = []
        
        # 批量处理目标语言到源语言的预测
        if tgt_queries and self.source_A_entities:
            print(f"NameSimilarityTool: 处理 {len(tgt_queries)} 个目标实体查询")
            
            # 准备源实体embedding
            target_entities = self.source_A_entities
            target_ids = []
            target_embeddings = []
            for target_entity in target_entities:
                if target_entity.id in self._embeddings:
                    target_ids.append(target_entity.id)
                    target_embeddings.append(self._embeddings[target_entity.id])
            
            if target_embeddings:
                # 将目标embedding转换为tensor并移到GPU
                target_embeddings_tensor = torch.from_numpy(np.stack(target_embeddings)).to(self._device)
                
                # 批量处理目标实体查询
                for entity_id in tgt_queries:
                    if entity_id not in self._embeddings:
                        continue
                        
                    try:
                        # 获取查询实体的embedding
                        embedding_a = torch.from_numpy(self._embeddings[entity_id]).to(self._device)
                        embedding_a = embedding_a.unsqueeze(0)  # 增加batch维度
                        
                        # 批量计算余弦相似度
                        cos = torch.nn.CosineSimilarity(dim=1)
                        similarities = cos(embedding_a, target_embeddings_tensor)
                        
                        # 转换到[0,1]范围并转换为numpy
                        similarities = ((similarities + 1) / 2).cpu().numpy()
                        
                        # 构建结果列表
                        entity_results = list(zip(target_ids, similarities))
                        entity_results.sort(key=lambda x: x[1], reverse=True)
                        results[entity_id] = entity_results[:top_k]
                        
                    except Exception as e:
                        print(f"NameSimilarityTool: 处理实体 {entity_id} 失败: {e}")
                        results[entity_id] = []
        
        print(f"NameSimilarityTool: 批量预测完成，共处理 {len(results)} 个实体的查询")
        return results
    def score(self, id_a: str, id_b: str) -> Optional[float]:
        """计算两个实体名称的相似度分数
        
        参数:
            id_a: 实体A的ID
            id_b: 实体B的ID
            
        返回:
            相似度分数 (0-1之间)，如果无法计算则返回None
        """
        
        # 获取实体信息
        cache_key = f"{id_a}_{id_b}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        entity_a = self.get_entity(id_a)
        entity_b = self.get_entity(id_b)
        if not entity_a or not entity_b:
            return None
        
        # 如果实体没有名称，返回None
        if not entity_a.name or not entity_b.name:
            return None
        
        # 尝试加载embedding文件
        if not self._embeddings and os.path.exists(self._embeddings_file):
            print(f"NameSimilarityTool: 从文件加载embedding {self._embeddings_file}")
            try:
                data = np.load(self._embeddings_file)
                self._embeddings = {key: data[key] for key in data.files}
                print(f"NameSimilarityTool: 成功加载{len(self._embeddings)}个实体的embedding")
            except Exception as e:
                print(f"NameSimilarityTool: 加载embedding文件失败: {e}")
        
        if self._embeddings and id_a in self._embeddings and id_b in self._embeddings:
            try:
                embedding_a = torch.from_numpy(self._embeddings[id_a]).to(self._device)
                embedding_b = torch.from_numpy(self._embeddings[id_b]).to(self._device)
                
                # 计算余弦相似度并转换到[0,1]范围
                cos = torch.nn.CosineSimilarity(dim=0)
                similarity = cos(embedding_a, embedding_b)
                similarity = ((similarity + 1) / 2).item()  # 转换到[0,1]范围
                self._cache[cache_key] = similarity
                return similarity
            except Exception as e:
                print(f"NameSimilarityTool: 使用embedding计算相似度失败: {e}")
        
        return None

class InfoDensitySimilarityTool(BaseTool):
    """信息密度相似度工具（直接计算工具）"""
    def __init__(self):
        super().__init__("InfoDensitySimilarityTool", "direct")
        self.description = "Computes entity information density similarity by analyzing the richness of entity attributes and information content. Entities with higher information density are considered more similar."
        
    def score(self, id_a: str, id_b: str) -> Optional[float]:
        """计算两个实体的信息密度相似度"""
        # 检查缓存
        
        # 获取实体信息
        cache_key = f"{id_a}_{id_b}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        entity_a = self.get_entity(id_a)
        entity_b = self.get_entity(id_b)
        
        if not entity_a or not entity_b:
            return None
        
        # 获取信息密度分数
        density_a = entity_a.info_density_score if entity_a.info_density_score is not None else 0
        density_b = entity_b.info_density_score if entity_b.info_density_score is not None else 0
        
        # 如果信息密度分数未计算，重新计算
        if density_a == 0:
            density_a = len(entity_a.relations) if entity_a.relations else 0
            density_a += len(entity_a.attributes) if entity_a.attributes else 0
            entity_a.info_density_score = density_a
        
        if density_b == 0:
            density_b = len(entity_b.relations) if entity_b.relations else 0
            density_b += len(entity_b.attributes) if entity_b.attributes else 0
            entity_b.info_density_score = density_b
        
        # 计算信息密度相似度
        # 使用1减去相对差的绝对值
        if max(density_a, density_b) == 0:
            similarity = 1.0  # 两者都是0，相似度为1
        else:
            # 使用1 - |a - b| / max(a, b) 来计算相似度
            similarity = 1 - abs(density_a - density_b) / max(density_a, density_b)
        self._cache[cache_key] = similarity
        return similarity



class DualAMN_EA_Tool(BaseTool):
    """Dual-AMN工具（Top-k查询工具）"""
    def __init__(self):
        super().__init__("DualAMN_EA", "top-k")  # 注意：这里使用工具目录名作为name
        self.description = "Performs entity alignment using Dual-AMN, combining structural and attribute information. Learns entity representations through attention mechanisms for more precise entity matching."
        self.supervision = True  # Requires supervised learning
        
    def run_preprocessing(self, data_path: str, seed_aligns_path: Optional[str] = None):
        """运行Dual-AMN工具的预处理，检测并运行init.sh"""
        # 调用基类的预处理方法（检测并运行init.sh）
        super().run_preprocessing(data_path, seed_aligns_path)
    
    
    def _parse_top_k_batch_results_file(self, results_file: str) -> Dict[str, List[Tuple[str, float]]]:
        """解析批量结果文件"""
        results = {}
        
        if not os.path.exists(results_file):
            print(f"结果文件不存在: {results_file}")
            return results
        
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 解析格式如 (0, 10500, 0.108379) 的行
                    try:
                        # 移除括号并分割
                        content = line.strip()
                        parts = content.split('\t')
                        if len(parts) == 3:
                            entity_id = parts[0]
                            target_id = parts[1]
                            score = float(parts[2])
                            
                            # 本来会有负数，好像范围是[-1,1]，这里调整到[0,1]
                            score = (max(0.0, min(1.0, (score + 1) / 2)))

                            if entity_id not in results:
                                results[entity_id] = []
                            results[entity_id].append((target_id, score))
                    except Exception as e:
                        print(f"解析行失败 '{line}': {e}")
        except Exception as e:
            print(f"读取结果文件失败: {e}")
        return results

    # def score(self, batch_file_path: str) -> Dict[Tuple[str, str], float]:
    #     """为多个实体对打分（使用Dual-AMN工具的特定实现）
        
    #     参数:
    #         batch_file_path: 批量查询文件路径，每行格式为"id_a\tid_b"
            
    #     返回:
    #         字典: {(id_a, id_b): score}
    #     """
    #     # 使用父类的score方法（会自动调用score.sh脚本）
    #     results = super().score_batch(batch_file_path)
        
    #     # 如果需要对结果进行特殊处理，可以在这里添加
    #     # 例如，Dual-AMN的分数范围可能需要调整
    #     processed_results = {}
    #     for (id_a, id_b), score in results.items():
    #         # 确保分数在[0, 1]范围内
    #         processed_score = max(0.0, min(1.0, (score + 1) / 2)) if score < 0 else score
    #         processed_results[(id_a, id_b)] = processed_score
        
    #     return processed_results


class ParisPlus_EA_Tool(BaseTool):
    """PARIS+工具（一次性获取结果工具）"""
    def __init__(self):
        super().__init__("PARIS_plus_EA", "one-time")
        self.description = "Improved version based on Paris algorithm, combining path similarity and attribute similarity for rule-based entity alignment."
        self.supervision = True
       
    def run_preprocessing(self, data_path: str, seed_aligns_path: Optional[str] = None):
        """运行PARIS+工具的预处理，检测并运行init.sh"""
        # 调用基类的预处理方法（检测并运行init.sh）
        super().run_preprocessing(data_path, seed_aligns_path)
        
        # 尝试读取结果文件
        results_file = os.path.join(self.tool_path, "results.txt")
        if os.path.exists(results_file):
            try:
                # 读取结果并缓存
                with open(results_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            content = line.strip()
                            parts = content.split('\t')
                            if len(parts) == 3:
                                id_a, id_b, score = parts[0], parts[1], float(parts[2])
                        except Exception as e:
                            print(f"解析PARIS+结果行失败 '{line}': {e}")
            except Exception as e:
                print(f"ParisPlus_EA_Tool结果读取失败: {e}")
    def score(self, id_a: str, id_b: str) -> Optional[float]:
        """计算两个实体在PARIS+算法中的对齐置信度
        
        参数:
            id_a: 实体A的ID
            id_b: 实体B的ID
            
        返回:
            对齐置信度分数 (0-1之间)，如果无法计算则返回None
        """
        
        # 尝试从结果文件中查找
        results_file = os.path.join(self.tool_path, "results.txt")
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            content = line.strip()
                            parts = content.split('\t')
                            if len(parts) == 3:
                                file_id_a, file_id_b, score = parts[0], parts[1], float(parts[2])
                                # 检查是否匹配当前查询
                                if (file_id_a == id_a and file_id_b == id_b) or (file_id_a == id_b and file_id_b == id_a):
                                    return score
                        except Exception as e:
                            print(f"解析PARIS+结果行失败 '{line}': {e}")
            except Exception as e:
                print(f"ParisPlus_EA_Tool结果文件读取失败: {e}")
                
        return None


class NameSimilarity2Tool(BaseTool):
    """名称相似度工具2（直接计算工具）"""
    def __init__(self):
        super().__init__("NameSimilarity2Tool", "direct")
        self.description = "Computes entity name similarity using different embedding models. Provides alternative semantic representation methods compared to NameSimilarity1, increasing diversity of alignment approaches."
        
    def score(self, id_a: str, id_b: str) -> Optional[float]:
        """使用文本相似度算法预测两个实体的名称相似度"""
        cache_key = f"{id_a}_{id_b}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        # 获取实体信息
        entity_a = self.get_entity(id_a)
        entity_b = self.get_entity(id_b)
        
        if not entity_a or not entity_b:
            return None  # 实体不存在时返回None
        
        # 检查名称是否存在
        name_a = getattr(entity_a, 'name', '')
        name_b = getattr(entity_b, 'name', '')
        
        if not name_a and not name_b:
            return None  # 两个实体都没有名称，无法比较，返回None
        elif not name_a or not name_b:
            return 0.0  # 只有一个实体有名称，视为不相似
        
        # 使用改进的文本相似度算法
        name_a = name_a.lower()
        name_b = name_b.lower()
        
        # 计算Jaccard相似度
        set_a = set(name_a)
        set_b = set(name_b)
        if not set_a and not set_b:
            jaccard_sim = 1.0
        elif not set_a or not set_b:
            jaccard_sim = 0.0
        else:
            jaccard_sim = len(set_a & set_b) / len(set_a | set_b)
            
        # 计算编辑距离相似度
        edit_distance = self._levenshtein_distance(name_a, name_b)
        max_len = max(len(name_a), len(name_b))
        edit_sim = 1 - edit_distance / max_len if max_len > 0 else 0.0
        
        # 计算最长公共子串相似度
        lcs_len = self._longest_common_substring(name_a, name_b)
        lcs_sim = lcs_len / max_len if max_len > 0 else 0.0
        
        # 综合计算相似度，确保在[0,1]范围内
        similarity = (jaccard_sim + edit_sim + lcs_sim) / 3.0
        similarity = max(0.0, min(1.0, similarity))  # 确保在[0,1]范围内
        self._cache[cache_key] = similarity
        return similarity
    
    def _levenshtein_distance(self, s: str, t: str) -> int:
        """计算两个字符串的编辑距离"""
        if len(s) < len(t):
            return self._levenshtein_distance(t, s)
        
        if len(t) == 0:
            return len(s)
        
        previous_row = list(range(len(t) + 1))
        for i, c1 in enumerate(s):
            current_row = [i + 1]
            for j, c2 in enumerate(t):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _longest_common_substring(self, s: str, t: str) -> int:
        """计算两个字符串的最长公共子串长度"""
        m, n = len(s), len(t)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        max_len = 0
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s[i - 1] == t[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    max_len = max(max_len, dp[i][j])
        
        return max_len

class OntologySimilarityTool(BaseTool):
    """本体相似度工具（直接计算工具）"""
    def __init__(self):
        super().__init__("OntologySimilarityTool", "direct")
        self.description = "Computes entity ontology similarity by analyzing entity types, categories, and hierarchical structure information. Particularly suitable for entities with rich ontological information."
        
    def score(self, id_a: str, id_b: str) -> Optional[float]:
        """使用集合相似度算法预测两个实体的本体相似度"""
        cache_key = f"{id_a}_{id_b}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        # 获取实体信息
        entity_a = self.get_entity(id_a)
        entity_b = self.get_entity(id_b)
        
        if not entity_a or not entity_b:
            return None  # 实体不存在时返回None
        
        # 检查本体信息是否存在
        ontology_a = getattr(entity_a, 'ontology', [])
        ontology_b = getattr(entity_b, 'ontology', [])
        
        # 如果ontology是字符串，尝试解析为列表
        if isinstance(ontology_a, str):
            try:
                import json
                ontology_a = json.loads(ontology_a)
            except:
                ontology_a = []
        if isinstance(ontology_b, str):
            try:
                import json
                ontology_b = json.loads(ontology_b)
            except:
                ontology_b = []
        
        if not ontology_a and not ontology_b:
            return None  # 两个实体都没有本体信息，无法比较，返回None
        elif not ontology_a or not ontology_b:
            return 0.0  # 只有一个实体有本体信息，视为不相似
        
        # 使用集合相似度算法
        set_a = set(ontology_a)
        set_b = set(ontology_b)
        
        # 计算Jaccard相似度
        jaccard_sim = len(set_a & set_b) / len(set_a | set_b) if set_a or set_b else 0.0
        
        # 计算重叠系数
        # min_size = min(len(set_a), len(set_b))
        # overlap_sim = len(set_a & set_b) / min_size if min_size > 0 else 0.0
    
        # 综合计算相似度，确保在[0,1]范围内
        similarity = jaccard_sim
        similarity = max(0.0, min(1.0, similarity))  # 确保在[0,1]范围内
        self._cache[cache_key] = similarity
        return similarity

class AttributeSimilarityTool(BaseTool):
    """属性相似度工具（直接计算工具）"""
    def __init__(self):
        super().__init__("AttributeSimilarityTool", "direct")
        self.description = "Computes entity attribute similarity by comparing attribute values, attribute types, and attribute distributions. Suitable for entity alignment scenarios with rich attribute information."
        
    def score(self, id_a: str, id_b: str) -> Optional[float]:
        """使用属性匹配算法预测两个实体的属性相似度"""
        cache_key = f"{id_a}_{id_b}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        # 获取实体信息
        entity_a = self.get_entity(id_a)
        entity_b = self.get_entity(id_b)
        
        if not entity_a or not entity_b:
            return None  # 实体不存在时返回None
        
        # 检查属性信息是否存在
        attr_a = getattr(entity_a, 'attributes', {})
        attr_b = getattr(entity_b, 'attributes', {})
        
        # 如果attributes是字符串，尝试解析为字典
        if isinstance(attr_a, str):
            try:
                import json
                attr_a = json.loads(attr_a)
            except:
                attr_a = {}
        if isinstance(attr_b, str):
            try:
                import json
                attr_b = json.loads(attr_b)
            except:
                attr_b = {}
        
        # 使用属性匹配算法
        similarity = self._calculate_simple_attribute_similarity(attr_a, attr_b)
        
        # 确保在[0,1]范围内
        similarity = max(0.0, min(1.0, similarity))
        self._cache[cache_key] = similarity
        return similarity
    
    def _calculate_simple_attribute_similarity(self, attr_a: Dict, attr_b: Dict) -> float:
        """改进的属性相似度计算"""
        if not attr_a and not attr_b:
            return 1.0
        if not attr_a or not attr_b:
            return 0.0
        
        # 计算属性名的相似度（不区分大小写）
        attr_a_lower = {k.lower(): v for k, v in attr_a.items()}
        attr_b_lower = {k.lower(): v for k, v in attr_b.items()}
        
        common_attrs = set(attr_a_lower.keys()) & set(attr_b_lower.keys())
        all_attrs = set(attr_a_lower.keys()) | set(attr_b_lower.keys())

        # 属性名匹配率
        name_match_rate = len(common_attrs) / len(all_attrs)
        
        # 属性值匹配率
        value_match_score = 0.0
        if common_attrs:
            for attr_name in common_attrs:
                val_a = str(attr_a_lower[attr_name]).lower()
                val_b = str(attr_b_lower[attr_name]).lower()
                
                # 计算属性值相似度
                if val_a == val_b:
                    value_match_score += 1.0
                else:
                    # 使用字符串相似度
                    set_a = set(val_a)
                    set_b = set(val_b)
                    if set_a and set_b:
                        similarity = len(set_a & set_b) / len(set_a | set_b)
                        value_match_score += similarity
            
            value_match_score /= len(common_attrs)
        
        # 综合计算相似度
        similarity = name_match_rate * 0.6 + value_match_score * 0.4
        return similarity
   
class RelationSimilarityTool(BaseTool):
    """关系相似度工具（直接计算工具）"""
    def __init__(self):
        super().__init__("RelationSimilarityTool", "direct")
        self.description = "Computes entity relation similarity by analyzing entity relationship networks, relation types, and relation density. Particularly suitable for knowledge graphs with dense relationships."
        
    def score(self, id_a: str, id_b: str) -> Optional[float]:
        """使用关系结构匹配算法预测两个实体的关系相似度"""
        cache_key = f"{id_a}_{id_b}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        # 获取实体信息
        entity_a = self.get_entity(id_a)
        entity_b = self.get_entity(id_b)

        if not entity_a or not entity_b:
            return None  # 实体不存在时返回None
        
        # 检查relations属性类型并转换为列表
        rel_a = self._safe_get_relations(entity_a)
        rel_b = self._safe_get_relations(entity_b)
        
        # 使用关系结构匹配算法
        similarity = self._calculate_simple_relation_similarity(rel_a, rel_b)
        
        # 应用邻域对齐提升
        # boost = self._calculate_neighborhood_alignment_boost(entity_a, entity_b)
        # similarity = min(1.0, similarity * 0.6 + boost)
        
        # # 确保在[0,1]范围内
        # similarity = max(0.0, min(1.0, similarity))
        self._cache[cache_key] = similarity
        return similarity
    
    def _safe_get_relations(self, entity) -> list:
        """安全地获取实体关系列表"""
        if not hasattr(entity, 'relations') or not entity.relations:
            return []
        
        # 如果relations是字符串，尝试多种解析方式
        if isinstance(entity.relations, str):
            try:
                # 尝试解析JSON字符串
                import json
                relations = json.loads(entity.relations)
                if isinstance(relations, list):
                    return relations
                elif isinstance(relations, dict):
                    return [relations]  # 单个关系转为列表
                else:
                    return []
            except:
                # 如果不是JSON，尝试按分隔符分割
                try:
                    # 尝试按逗号分割
                    relations = [r.strip() for r in entity.relations.split(',') if r.strip()]
                    return relations
                except:
                    # 如果无法解析，返回空列表
                    return []
        # 如果relations是列表，直接返回
        elif isinstance(entity.relations, list):
            return entity.relations
        # 如果relations是字典，转为列表
        elif isinstance(entity.relations, dict):
            return [entity.relations]
        # 其他类型，返回空列表
        else:
            return []
    
    # def _calculate_simple_relation_similarity(self, rel_a: List, rel_b: List) -> float:
    #     """改进的属性相似度计算"""
    #     if not rel_a and not rel_b:
    #         return 1.0
    #     if not rel_a or not rel_b:
    #         return 0.0
    #     # rel 是 list(dict(a:[b, c]))

    #     print("rel_a, rel_b", rel_a, rel_b)
    #     rel_a_lower = [k.lower() for d in rel_a for k in d.keys()]
    #     rel_b_lower = [k.lower() for d in rel_b for k in d.keys()]
    #     print(1)
    #     print(rel_a_lower)
    #     print(rel_b_lower)
    #     common_rels = set(rel_a_lower) & set(rel_b_lower)
    #     all_rels = set(rel_a_lower) | set(rel_b_lower)
    #     name_match_rate = len(common_rels) / len(all_rels)
        
    #     # 属性值匹配率
    #     value_match_score = 0.0
    #     if common_rels:
    #         for rel_name in common_rels:
                
    #             print(rel_name)
    #             val_a = str(rel_a_lower[0][rel_name]).lower()
    #             val_b = str(rel_b_lower[0][rel_name]).lower()
    #             print(2)

    #             # 计算属性值相似度
    #             if val_a == val_b:
    #                 value_match_score += 1.0
    #             else:
    #                 # 使用字符串相似度
    #                 set_a = set(val_a)
    #                 set_b = set(val_b)
    #                 if set_a and set_b:
    #                     similarity = len(set_a & set_b) / len(set_a | set_b)
    #                     value_match_score += similarity
            
    #         value_match_score /= len(common_attrs)
    #       # 属性值匹配率
    #     # # 综合计算相似度
    #     similarity = name_match_rate * 0.6 + value_match_score * 0.4
    #     return similarity

    def _calculate_simple_relation_similarity(self, rel_a: List, rel_b: List) -> float:
        """改进的关系相似度计算"""
        if not rel_a and not rel_b:
            return 1.0
        if not rel_a or not rel_b:
            return 0.0
        # rel 是 list(dict(a:[b, c]))
 
        #print("rel_a, rel_b", rel_a, rel_b)
        
        # 提取关系名称（小写）
        rel_a_names = [k.lower() for d in rel_a for k in d.keys()]
        rel_b_names = [k.lower() for d in rel_b for k in d.keys()]
        
        #print("rel_a_names:", rel_a_names)
        #print("rel_b_names:", rel_b_names)
        
        common_rels = set(rel_a_names) & set(rel_b_names)
        all_rels = set(rel_a_names) | set(rel_b_names)
        
        if not all_rels:
            return 0.0
            
        name_match_rate = len(common_rels) / len(all_rels)
        
        # 关系值匹配率
        value_match_score = 0.0
        if common_rels:
            for rel_name in common_rels:
                #print("处理关系:", rel_name)
                
                # 从原始关系中查找对应的值
                val_a_list = []
                val_b_list = []
                
                # 查找关系A的值
                for rel_dict in rel_a:
                    for key, value in rel_dict.items():
                        if key.lower() == rel_name:
                            if isinstance(value, list):
                                val_a_list.extend(value)
                            else:
                                val_a_list.append(str(value))
                
                # 查找关系B的值
                for rel_dict in rel_b:
                    for key, value in rel_dict.items():
                        if key.lower() == rel_name:
                            if isinstance(value, list):
                                val_b_list.extend(value)
                            else:
                                val_b_list.append(str(value))
                
                # 计算关系值相似度
                if val_a_list and val_b_list:
                    # 转换为小写字符串进行比较
                    val_a_set = set(str(v).lower() for v in val_a_list)
                    val_b_set = set(str(v).lower() for v in val_b_list)
                    
                    if val_a_set and val_b_set:
                        # 使用Jaccard相似度
                        similarity = len(val_a_set & val_b_set) / len(val_a_set | val_b_set)
                        value_match_score += similarity
                        #print(f"关系 {rel_name} 相似度: {similarity}")
                else:
                    # 如果一方有值而另一方没有，相似度为0
                    value_match_score += 0.0
            
            value_match_score /= len(common_rels)
        
        # 综合计算相似度
        similarity = name_match_rate * 0.6 + value_match_score * 0.4
        #print(f"最终相似度: 名称匹配率={name_match_rate}, 值匹配率={value_match_score}, 综合={similarity}")
        return similarity

    
    # def _are_neighbors_aligned(self, neighbor_a: str, neighbor_b: str) -> bool:
    #     """判断两个邻居实体是否在当前轮次的对齐中匹配"""
    #     # 如果两个邻居实体ID相同，直接返回True
    #     if neighbor_a == neighbor_b:
    #         return True
        
    #     # 检查当前轮次的对齐信息
    #     if neighbor_a in self._current_round_alignments:
    #         aligned_entity = self._current_round_alignments[neighbor_a]
    #         return aligned_entity == neighbor_b
        
    #     if neighbor_b in self._current_round_alignments:
    #         aligned_entity = self._current_round_alignments[neighbor_b]
    #         return aligned_entity == neighbor_a
        
    #     return False
    
    # def _calculate_neighborhood_alignment_boost(self, entity_a, entity_b) -> float:
    #     """计算邻域对齐提升（基于当前轮次的对齐信息判断共同邻居）"""
    #     rel_a = self._safe_get_relations(entity_a)
    #     rel_b = self._safe_get_relations(entity_b)
        
    #     if not rel_a or not rel_b:
    #         return 0.0
        
    #     # 提取邻居实体
    #     neighbors_a = set()
    #     neighbors_b = set()
        
    #     for r in rel_a:
    #         if isinstance(r, dict):
    #             target = r.get('target', '')
    #             if target:
    #                 # 处理列表类型的target
    #                 if isinstance(target, list):
    #                     for t in target:
    #                         if t:
    #                             neighbors_a.add(str(t))
    #                 else:
    #                     neighbors_a.add(str(target))
        
    #     for r in rel_b:
    #         if isinstance(r, dict):
    #             target = r.get('target', '')
    #             if target:
    #                 # 处理列表类型的target
    #                 if isinstance(target, list):
    #                     for t in target:
    #                         if t:
    #                             neighbors_b.add(str(t))
    #                 else:
    #                     neighbors_b.add(str(target))
        
    #     # 计算基于当前轮次对齐的共同邻居
    #     aligned_common_neighbors = 0
    #     for neighbor_a in neighbors_a:
    #         for neighbor_b in neighbors_b:
    #             if self._are_neighbors_aligned(neighbor_a, neighbor_b):
    #                 aligned_common_neighbors += 1
    #                 break  # 每个neighbor_a只需要匹配一个neighbor_b
        
    #     if aligned_common_neighbors == 0:
    #         return 0.0
        
    #     min_neighbors = min(len(neighbors_a), len(neighbors_b))
    #     return aligned_common_neighbors / min_neighbors if min_neighbors > 0 else 0.0




class NeighborhoodSimilarity_EA_Tool(BaseTool):
    """邻域相似度工具（直接计算工具），基于一跳和二跳邻域中对齐实体的数量计算相似度"""
    def __init__(self):
        super().__init__("NeighborhoodSimilarity_EA_Tool", "direct")
        self.description = "Computes entity similarity based on neighborhood alignment. Analyzes 1-hop and 2-hop neighborhoods to count aligned entities, considering structural similarity in knowledge graphs."
        
    def score(self, id_a: str, id_b: str) -> Optional[float]:
        """基于邻域对齐计算两个实体的相似度"""
        cache_key = f"{id_a}_{id_b}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # 获取实体信息
        entity_a = self.get_entity(id_a)
        entity_b = self.get_entity(id_b)

        if not entity_a or not entity_b:
            return None  # 实体不存在时返回None
        
        # 计算邻域对齐相似度
        similarity = self._calculate_neighborhood_similarity(entity_a, entity_b)
        
        # 确保在[0,1]范围内
        similarity = max(0.0, min(1.0, similarity))
        self._cache[cache_key] = similarity
        return similarity
    
    def _get_neighbors(self, entity, hops=1):
        """获取指定跳数的邻居实体"""
        if not hasattr(entity, 'relations') or not entity.relations:
            return set(), set()  # 返回一跳和二跳邻居
        
        # 提取一跳邻居
        direct_neighbors = set()
        if isinstance(entity.relations, dict):
            for targets in entity.relations.values():
                if isinstance(targets, list):
                    direct_neighbors.update([str(t) for t in targets if t])
                elif targets:  # 如果是单个值
                    direct_neighbors.add(str(targets))
        elif isinstance(entity.relations, list):
            for relation in entity.relations:
                if isinstance(relation, dict):
                    target = relation.get('target', '')
                    if target:
                        if isinstance(target, list):
                            direct_neighbors.update([str(t) for t in target if t])
                        else:
                            direct_neighbors.add(str(target))

        if hops == 1:
            return direct_neighbors, set()
        
        # 计算二跳邻居
        second_hop_neighbors = set()
        for neighbor_id in direct_neighbors:
            neighbor_entity = self.get_entity(neighbor_id)
            if neighbor_entity and hasattr(neighbor_entity, 'relations') and neighbor_entity.relations:
                if isinstance(neighbor_entity.relations, dict):
                    for targets in neighbor_entity.relations.values():
                        if isinstance(targets, list):
                            second_hop_neighbors.update([str(t) for t in targets if t and str(t) != entity.id and str(t) not in direct_neighbors])
                        elif targets and str(targets) != entity.id and str(targets) not in direct_neighbors:
                            second_hop_neighbors.add(str(targets))
                elif isinstance(neighbor_entity.relations, list):
                    for relation in neighbor_entity.relations:
                        if isinstance(relation, dict):
                            target = relation.get('target', '')
                            if target:
                                if isinstance(target, list):
                                    for t in target:
                                        if t and str(t) != entity.id and str(t) not in direct_neighbors:
                                            second_hop_neighbors.add(str(t))
                                else:
                                    if str(target) != entity.id and str(target) not in direct_neighbors:
                                        second_hop_neighbors.add(str(target))
        
        return direct_neighbors, second_hop_neighbors
    
    def _are_neighbors_aligned(self, neighbor_a: str, neighbor_b: str) -> bool:
        """判断两个邻居实体是否在当前轮次的对齐中匹配"""
        # 如果两个邻居实体ID相同，直接返回True
        if neighbor_a == neighbor_b:
            return True
        
        # 检查当前轮次的对齐信息
        if neighbor_a in self._current_round_alignments:
            aligned_entity = self._current_round_alignments[neighbor_a]
            return aligned_entity == neighbor_b
        
        if neighbor_b in self._current_round_alignments:
            aligned_entity = self._current_round_alignments[neighbor_b]
            return aligned_entity == neighbor_a
        
        return False
    
    def _calculate_neighborhood_similarity(self, entity_a, entity_b) -> float:
        """计算基于邻域对齐的相似度"""
        # 获取实体A和实体B的一跳和二跳邻居
        a_first_hop, a_second_hop = self._get_neighbors(entity_a, hops=2)
        b_first_hop, b_second_hop = self._get_neighbors(entity_b, hops=2)
        
        # 计算一跳邻居中的对齐数量
        aligned_first_hop = 0
        for neighbor_a in a_first_hop:
            for neighbor_b in b_first_hop:
                if self._are_neighbors_aligned(neighbor_a, neighbor_b):
                    aligned_first_hop += 1
                    break  # 每个neighbor_a只需要匹配一个neighbor_b
        
        # 计算二跳邻居中的对齐数量
        aligned_second_hop = 0
        for neighbor_a in a_second_hop:
            for neighbor_b in b_second_hop:
                if self._are_neighbors_aligned(neighbor_a, neighbor_b):
                    aligned_second_hop += 1
                    break  # 每个neighbor_a只需要匹配一个neighbor_b
        
        # 计算总的邻居数量（考虑一跳和二跳）
        total_neighbors_a = len(a_first_hop) + len(a_second_hop)
        total_neighbors_b = len(b_first_hop) + len(b_second_hop)
        
        if total_neighbors_a == 0 and total_neighbors_b == 0:
            return 1.0  # 如果两个实体都没有邻居，认为完全相似
        if total_neighbors_a == 0 or total_neighbors_b == 0:
            return 0.0  # 如果只有一个实体有邻居，认为完全不相似
        
        # 计算对齐的邻居比例
        total_aligned = aligned_first_hop + aligned_second_hop
        total_possible = min(total_neighbors_a, total_neighbors_b)
        
        if total_possible == 0:
            return 0.0
        
        # 根据对齐邻居的比例计算相似度
        similarity = total_aligned / total_possible
        
        # 也可以加入权重，让一跳邻居更重要
        first_hop_weight = 0.7
        second_hop_weight = 0.3
        
        first_hop_similarity = aligned_first_hop / min(len(a_first_hop), len(b_first_hop)) if min(len(a_first_hop), len(b_first_hop)) > 0 else 0.0
        second_hop_similarity = aligned_second_hop / min(len(a_second_hop), len(b_second_hop)) if min(len(a_second_hop), len(b_second_hop)) > 0 else 0.0
        
        weighted_similarity = first_hop_weight * first_hop_similarity + second_hop_weight * second_hop_similarity
        
        # 返回最大值，确保相似度不会因为邻居数量差异而过低
        return max(similarity, weighted_similarity)




# BERTINT_EA 工具类
class BERTINT_EA_Tool(BaseTool):
    """BERTINT_EA 工具（Top-k查询工具）"""
    def __init__(self):
        super().__init__("BERTINT_EA", "top-k")
        self.description = "BERTINT-based Entity Alignment tool that leverages BERT embeddings and interaction models for entity alignment. Computes semantic similarity between entities using advanced BERT-based representations and interaction mechanisms."
    



class TransE_EA_Tool(BaseTool):
    """TransE工具（Top-k查询工具）"""
    def __init__(self):
        super().__init__("TransE_EA", "top-k")
        self.description = "TransE-based entity alignment using translational embedding method. Represents entities and relations as vectors in a unified space, where relation vectors act as translations between entity vectors."
        self.supervision = True
        


class BootEA_EA_Tool(BaseTool):
    """BootEA工具（Top-k查询工具）"""
    def __init__(self):
        super().__init__("BootEA_EA", "top-k")
        self.description = "Bootstrap Embedding Alignment (BootEA) for entity alignment. Uses a bootstrapping process to iteratively improve entity embeddings and alignment results."
        self.supervision = True
        

class GCN_Align_EA_Tool(BaseTool):
    """GCN-Align工具（Top-k查询工具）"""
    def __init__(self):
        super().__init__("GCN-Align_EA", "top-k")
        self.description = "Graph Convolutional Network Alignment (GCN-Align) for entity alignment. Uses graph convolutional networks to learn entity embeddings considering network structure."
        self.supervision = True
        

class DistMult_EA_Tool(BaseTool):
    """DistMult工具（Top-k查询工具）"""
    def __init__(self):
        super().__init__("DistMult_EA", "top-k")
        self.description = "DistMult-based entity alignment using bilinear scoring function. Represents entities and relations as vectors and uses matrix multiplication to capture relational patterns."
        self.supervision = True


class TransEdge_EA_Tool(BaseTool):
    """TransEdge工具（Top-k查询工具）"""
    def __init__(self):
        super().__init__("TransEdge_EA", "top-k")
        self.description = "TransEdge-based entity alignment using edge-based translational embedding method. Projects entities into a low-dimensional space where relations are modeled as translation vectors on edges."
        self.supervision = True
    

class MTransE_EA_Tool(BaseTool):
    """MTransE工具（Top-k查询工具）"""
    def __init__(self):
        super().__init__("MTransE_EA", "top-k")
        self.description = "MTransE-based entity alignment using multi-view translational embedding method."
        self.supervision = True
