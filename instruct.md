### 基于Agent的知识融合（Entity Align & Entity Matching）统一框架
**核心思想**: 将来自知识图谱（KG）的结构化数据和来自表格的结构化数据，统一抽象为“实体-关系-属性”的表示。框架的各个Agent和工具将基于这种统一表示进行操作，而无需关心数据原始来源是KG还是Table。

---

### **第一步：数据处理及加载**
**目标**: 将所有异构数据源（EA的KG数据，EM的Table数据）加载并转换为统一的内部数据结构。
**输入文件**
EA数据格式(位于 `./data/EA/{data_name}/`)
- ent_ids_1, ent_ids_2 表示实体信息（每行都是：{entity_id}{\t}{entity_name}）
- rel_ids_1, rel_ids_2 表示关系信息（每行都是：{relation_id}{\t}{relation_name}）
- rel_triples_1、rel_triples_2 表示关系三元组信息（每行都是：{entity_id}{\t}{relation_id}{\t}{entity_id}）
- attr_triples_1、attri_triples_2 表示属性三元组信息（每行都是：{entity_id}{\t}{attribute_name}{\t}{value}）
- seed_aligns 表示已知对齐信息（每行都是：{entity_id}{\t}{entity_id}）
- test_aligns 表示测试对齐信息（每行都是：{entity_id}{\t}{entity_id}）
- image_pkls 表示实体图片信息（词典，entity_id: image_embedding_vector）
EM 数据格式(位于 `./data/EM/{data_name}/`)
- tableA.csv, tableB.csv 表示实体属性表（第一个属性都是id，两个表的实体id都从0开始标）
- train.csv, test.csv, valid.csv 表示训练、测试、验证例子（每行都是 {ltable_id},{rtable_id},{label}，lable为1表示对齐）

**内部统一数据格式**:
我们将定义一个核心的 `UnifiedEntity` 对象，EA和EM的所有实体都将被解析成此格式。
```python
class UnifiedEntity:
    def __init__(self, entity_id: str, source_name: str):
        self.id = entity_id  # 实体在原始数据源中的ID
        self.source = source_name  # 例如: "KG1", "KG2", "TableA", "TableB"
        self.name: Optional[str] = None  # 实体名称(来自EA 或者 EM 的 name 列)
        self.attributes: Optional[Dict[str, Any]] = {}  # 属性字典 {attr_name: value}
        self.relations: Optional[Dict[str, List[str]]] = {}  # 关系字典 {rel_id: [target_entity_id, ...]}
        self.info_density_score: Optional[float] = None  # 信息密度，关系个数+属性个数
        self.ontology: Optional[List[str]] # 由 LLM 从我给出的 ontology_list中选最可能的两个
```

**加载数据**:
加载实体
- 为每个实体创建 `UnifiedEntity` 对象，填充id , source, name, attributes, relations，生成 `source_A_entities` 和 `source_B_entities`。
加载对齐信息
- 创建统一的 `AlignmentPair` 格式。
- 从 `seed_aligns` / `valid.csv` + `train.csv` 加载，合并为 `seed_alignments`。
- 从 `test_aligns` / `test.csv` 加载为 `test_alignments`。

---

### **第二步：数据分析 Agent**

**目标**: 深入理解统一后的数据，为后续的对齐策略提供高层次的洞察，补全`UnifiedEntity`中缺失的部分。

**输入**:  所有 `UnifiedEntity`

**分析任务与输出**:
1. **模态分析**:
    - **任务**: 检查两个数据源中是否存在名称、关系、属性、图片等信息。
    - **实现**: 遍历 `UnifiedEntity` 对象，检查当前数据集是否包含名称等信息, 可以用到 LLM 来判断是名称还是 ID。
    - **输出**: `DataModalReport` 对象
```python
        class DataModalReport:
            has_names: bool
            has_relations: bool  # 区分是EA任务还是EM任务的关键
            has_relation_names: bool
            has_attributes: bool
            has_attribute_names: bool
            has_images: bool # 针对 EA, 检查是否存在文件./data/EA/data_dir/image_pkls
            is_ea_task: bool  # has_relations为True时为True
```
2. **信息密度评估**:
    - **任务**: 为每个实体计算信息密度分数 `UnifiedEntity.info_density_score`。
    - **实现**:
        - 对于EA实体: `density = len(relations) + len(attributes)`
        - 对于EM实体: `density = len(attributes)`
        - 更新每个 `UnifiedEntity.info_density_score`。
    - **输出**: 实体信息密度分数已更新到 `source_A_entities` 和 `source_B_entities` 中。
3. **本体判断**:
    - **任务**: 如果 has_names = True, 为每个实体生成 `UnifiedEntity.ontology`。
    - **实现 :
        - **输入**: `source_A_entities` 和 `source_B_entities`，以及我另外给出的一个 ontology_list
        - **实现**：利用 LLM 实现，对于每个实体，帮我从ontology_list中选择出三个实体最有可能属于的ontology,放到`UnifiedEntity.ontology`中，以List[str] 的结构。我会在 `./ontology_list.txt` 中给出 ontology_list， 每行一个字符串表示一种ontology。
        - **输出**: 实体ontology已更新到 `source_A_entities` 和 `source_B_entities` 中。

---
### **第三步：工具**

**输入**：`DataModalReport` 对象。
**输出**：`available_tools`: 所有可用的工具实例列表。
**位置**：工具全部在  `./tools` 下，文件夹名称后缀 `_EA` 或者 `_EM` 表示专用工具，否则为通用工具
**目标**: 提供一系列计算实体对相似度/对齐置信度的原子工具。所有最终传输给对齐执行 Agent的格式都是： `(tool_name, source_A_entities_id, source_B_entities_id, confidence)`。

**工具分类**
1. **按类别分：**
	- 通用工具：适用于EA & EM
	- EA专用工具
	- EM专用工具
2. **按结果获取分：**
	- 一次性获取结果工具：运行工具run.sh后，可以从 ./tools/{tool_name}/results.txt中一次读取该工具预测出的所有结果，格式如  `(source_A_entities_id, source_B_entities_id, confidence)`。
	- Top-k查询工具：运行工具run.sh后，还需运行predict.sh，给出三个参数，分别是 entity_id, entity_source（A或者B）, top_k，得到与 entity_id 的前 top_k 个对齐结果 `(source_A_entities_id, source_B_entities_id, confidence)`。
	- 单次查询工具：运行工具run.sh后，还需运行predict.sh，给出两个参数，分别是 entity_id_A, entity_id_B，得到两个实体是否对齐，结果为`(source_A_entities_id, source_B_entities_id, confidence)`，其中confidence为0或1。
	- LLM 工具：调用LLM比较两者相似度，直接给出一个[0, 1]的 confidence 值。
	- 直接计算工具：直接计算可得 confidence 值。

**工具实现**:
1. **通用工具 (适用于EA & EM)**:
    - **名称相似度工具1** (`name_similarity_1_tool`, Top-k查询工具)：如果 `DataModalReport.has_names` 为True则启用。利用本地的paraphrase-multilingual-MiniLM-L12-v2 模型和cosine similarity计算。
    - 翻译+字符串比较
    - **名称相似度工具2** (`name_similarity_2_tool`, LLM 工具)：如果 `DataModalReport.has_names` 为True则启用。LLM 比较两实体名称的相似度，直接给出一个[0, 1]的值。
    - **本体相似度工具** (`ontology_similarity_tool`, LLM 工具)：如果 `DataModalReport.has_names` 为True则启用。LLM 比较两实体 ontology的相似度，直接给出一个[0, 1]的值。
    - **属性相似度工具** (`attribute_similarity_tool`, LLM 工具)：如果 `DataModalReport.has_attributes` 为True则启用。LLM 比较两实体 attributes的相似度，直接给出一个[0, 1]的值。
    - **关系相似度工具** (`relation_similarity_tool`, LLM 工具)：如果 `DataModalReport.has_relation_names` 为True则启用， LLM 比较两实体 relations的相似度，直接给出一个[0, 1]的值。
    - **信息密度相似度工具** (`info_densitysimilarity_tool`，直接计算工具)：默认启用，1 - abs(两者的info_density_score的差)。
2. **EA专用工具**:
    - **Dual-AMN工具** (`dual_amn_tool`, Top-k查询工具)：如果 `DataModalReport.is_kg_task` 为True则启用。实际运行时要运行一个指定的`run.sh`文件后读取另外一个`result.txt`文件得到结果，`result.txt`中结果的格式为 `(id_a, id_b, score)`。
    - **PARIS+工具** (`paris_plus_tool`，一次性获取结果工具)：如果 `DataModalReport.is_kg_task` 且 `has_names` 为True则启用。实际运行时要运行一个指定的`run.sh`文件后读取另外一个`result.txt`文件得到结果，`result.txt`中结果的格式为 `(id_a, id_b, score)`。
    - **邻域相似度工具** (`neighborhood_similarity_tool`, LLM 工具)：如果 `DataModalReport.is_kg_task` 为True则启用。利用 `seed_alignments` 中已知的对齐，利用 LLM 比较两个实体的领域相似度，直接给出一个[0, 1]的值。
3. **EM专用工具**:
	- **DeepMatcher 工具** (`deepmatcher_tool`, 单次查询工具) : 如果 `DataModalReport.has_attributes` 为True则启用。
	- **Ditto 工具**  (`ditto_tool`, 单次查询工具)：如果`DataModalReport.has_attributes` 为True则启用。
---

### **第四步：对齐执行 Agent**

**目标**:  
综合调用所有可用工具，以迭代方式逐步发现并确认高质量的实体对齐结果。通过候选生成、多工具融合打分、冲突消解与记忆更新机制，在保证效率的前提下提升对齐精度。


### **输入**:

- `source_A_entities`: List[UnifiedEntity] —— 来自源A的所有统一实体
- `source_B_entities`: List[UnifiedEntity] —— 来自源B的所有统一实体
- `seed_alignments`: List[Tuple[str, str]] —— 初始已知对齐（训练/验证集）
- `test_alignments`: List[Tuple[str, str]] —— 测试集对齐（用于最终评估）
- `available_tools`: List[Tool] —— 第三步中根据 `DataModalReport` 筛选出的所有可用工具实例
- 超参数：
    - `max_rounds`: 最大迭代轮数（默认为10）
    - `convergence_threshold`: 收敛阈值（默认为0.1，即新对齐数量 < 当前记忆中对齐总数 × 10% 时停止）

### **输出**:

- 每轮生成的对齐结果文件：`./output/round_{i}_alignments.txt`，格式为 `(entity_id_A, entity_id_B, confidence)`
- 所有工具原始输出记录：`./output/tool_outputs.pkl`，保存为 List[Tuple[str, str, str, float]]，即 `(tool_name, id_A, id_B, confidence)`
- 最终对齐结果：`./output/final_alignments.txt`
- 评估指标：Precision, Recall, F1（基于 `test_alignments`）

### **核心流程设计**

#### **1. 初始化记忆池（Memory Pool）**

- 创建一个字典 `memory: Dict[str, Tuple[str, float]]`，用于维护一对一映射：
    - key: source_A 的 entity_id
    - value: (source_B 的 entity_id, confidence)
- 同时维护反向映射 `reverse_memory: Dict[str, str]`，用于快速检测冲突。
- 将 `seed_alignments` 中所有对齐加入 `memory`，置信度设为 1.0。

#### **2. 工具并发预执行（仅一次性获取 & Top-k 工具）**

- 对于以下两类工具，提前运行其 `run.sh` 并缓存结果：
    - **一次性获取结果工具**（如 `paris_plus_tool`）：直接读取全部预测结果。
    - **Top-k 查询工具**（如 `name_similarity_1_tool`, `dual_amn_tool`）：对所有 A 实体批量调用 Top-k 查询（例如 k=50），缓存每个 A 实体的候选 B 实体列表。
- **注意**：LLM 工具、单次查询工具（如 DeepMatcher）不在此阶段运行，而是在后续按需调用。
- 所有工具运行完毕后，统一提示：“所有预执行工具已完成，开始对齐迭代。”

#### **3. 迭代对齐主循环**

```text
for round in range(max_rounds):
    new_alignments_this_round = []
    
    # Step 3.1: 候选生成
    candidate_pairs = set()
    for tool in available_tools:
        if tool.type in ["one-time", "top-k"]:
            # 从预缓存结果中提取所有 (id_A, id_B) 对
            candidate_pairs.update(tool.get_all_candidates())
        # 注意：LLM / 单次工具暂不用于候选生成（效率低）
    
    # Step 3.2: 多工具融合打分
    scored_candidates = []
    for (id_A, id_B) in candidate_pairs:
        if id_A in memory and memory[id_A][0] == id_B:
            continue  # 已确认，跳过
        if id_A in memory or id_B in reverse_memory:
            continue  # 违反一对一，暂不处理（后续冲突消解阶段处理）
        
        total_score = 0.0
        tool_count = 0
        
        for tool in available_tools:
            conf = tool.predict(id_A, id_B)  # 自动适配工具类型（缓存 or 实时调用）
            if conf is not None:
                total_score += conf
                tool_count += 1
        
        if tool_count > 0:
            avg_conf = total_score / tool_count  # 当前先等权重平均
            scored_candidates.append((id_A, id_B, avg_conf))
            # 同时记录原始工具输出
            record_tool_outputs(tool.name, id_A, id_B, conf)
    
    # Step 3.3: 排序 & 选择高置信对齐
    scored_candidates.sort(key=lambda x: x[2], reverse=True)
    threshold = 0.7  # 可配置，或动态调整
    high_conf_pairs = [(a, b, c) for (a, b, c) in scored_candidates if c >= threshold]
    
    # Step 3.4: 冲突检测与消解（一对一约束）
    for (id_A, id_B, conf) in high_conf_pairs:
        conflict = False
        if id_A in memory:
            old_idB, old_conf = memory[id_A]
            if conf > old_conf:
                # 替换旧对齐
                del reverse_memory[old_idB]
                memory[id_A] = (id_B, conf)
                reverse_memory[id_B] = id_A
                new_alignments_this_round.append((id_A, id_B, conf))
            else:
                conflict = True
        elif id_B in reverse_memory:
            old_idA = reverse_memory[id_B]
            old_conf = memory[old_idA][1]
            if conf > old_conf:
                del memory[old_idA]
                memory[id_A] = (id_B, conf)
                reverse_memory[id_B] = id_A
                new_alignments_this_round.append((id_A, id_B, conf))
            else:
                conflict = True
        else:
            # 无冲突，直接加入
            memory[id_A] = (id_B, conf)
            reverse_memory[id_B] = id_A
            new_alignments_this_round.append((id_A, id_B, conf))
    
    # Step 3.5: 保存本轮结果
    save_to_file(f"./output/round_{round+1}_alignments.txt", new_alignments_this_round)
    
    # Step 3.6: 收敛判断
    if len(new_alignments_this_round) < len(memory) * convergence_threshold:
        print(f"Converged at round {round+1}. Stopping.")
        break
```

#### **4. 最终评估**

- 从 `memory` 中提取所有对齐对 `(id_A, id_B)`（忽略置信度）
- 与 `test_alignments`（ground truth）比较：
    - TP = 正确预测且在测试集中
    - FP = 预测但不在测试集中
    - FN = 测试集中未被预测
- 计算：
    - Precision = TP / (TP + FP)
    - Recall = TP / (TP + FN)
    - F1 = 2 * P * R / (P + R)
- 输出到 `./output/evaluation.txt`

### **工具调用策略说明**

- **主要候选生成工具必须是 Top-k 或一次性工具**（如 `dual_amn_tool`, `name_similarity_1_tool`），因其可高效覆盖大规模实体对。
- **LLM 工具和单次查询工具仅用于精打分**，不在候选生成阶段全量调用，避免 O(n²) 开销。
- 所有工具的 `run.sh` 在迭代开始前**并行启动**（使用 subprocess + wait），确保预计算完成后再进入主循环。
- 工具的 `predict(id_A, id_B)` 方法内部自动判断：
    - 若为缓存型工具 → 查表返回
    - 若为 LLM/单次工具 → 实时调用（可加缓存避免重复）

### **记忆机制与冲突处理原则**

- 强制维护**一对一映射**（bijection）。
- 冲突时**保留高置信度对齐**，低置信度被覆盖。
- 被覆盖的对齐**不回滚**，但记录日志供分析（可选）。




第一步和第二部分已经实现在 #data_loader.py 和 #data_analyzer.py 中,帮我实现第三部分和第四部分，同时修改总体控制流程 #main.py