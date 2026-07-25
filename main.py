import argparse
import os
import time
import sys
import logging
from typing import Dict, Any

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    # handlers=[
    #     logging.FileHandler("./run.log"),
    #     logging.StreamHandler()
    # ]
     handlers=[
        logging.FileHandler("./run.log")
    ]
)
logger = logging.getLogger(__name__)

from data_loader import DataLoader, UnifiedEntity, AlignmentPair
from data_analyzer import DataAnalyzer
from tools import ToolsManager
from alignment_agent import AlignmentAgent
from evaluation import evaluate_alignments, print_evaluation_results


def ensure_output_directory():
    """确保输出目录存在"""
    os.makedirs("./output", exist_ok=True)
    logger.info("输出目录已准备就绪")


def validate_data_path(data_path):
    """验证数据路径是否存在"""
    if not os.path.exists(data_path):
        logger.error(f"数据路径不存在: {data_path}")
        raise ValueError(f"数据路径不存在: {data_path}")
    logger.info(f"数据路径验证通过: {data_path}")


def main():
    """主函数"""
    try:
        # 确保输出目录存在
        ensure_output_directory()
        
        # 解析命令行参数
        parser = argparse.ArgumentParser(description='基于Agent的知识融合统一框架')
        parser.add_argument('--task_type', type=str, default='EA', choices=['EA', 'EM'], help='任务类型')
        parser.add_argument('--data_path', type=str, help='数据路径')
        parser.add_argument('--confidence_threshold', type=float, default=0.7, help='置信度阈值')
        parser.add_argument('--max_iterations', type=int, default=100, help='最大迭代次数')
        parser.add_argument('--convergence_threshold', type=float, default=-1, help='收敛阈值')
        parser.add_argument('--force_analyze', action='store_true', help='强制重新分析数据，不使用缓存')
        parser.add_argument('--top_k', type=int, default=5, help='top-k搜索的k值')
        parser.add_argument('--supervision', type=bool, default=True, help='是否使用监督模式')
        parser.add_argument('--schema_alignment_round', type=int, default=10, help='模式对齐轮数')
        parser.add_argument('--percentage_per_round', type=float, default=0.1, help='每轮模式对齐的百分比')
        args = parser.parse_args()
        
        # 如果没有指定数据路径，使用默认路径
        if not args.data_path:
            if args.task_type == 'EA':
                args.data_path = './data/EA/dbp15k_zh_en'
            else:
                args.data_path = './data/EM/Abt-Buy'
        
        # 验证数据路径
        validate_data_path(args.data_path)
        
        logger.info("===== 基于Agent的知识融合统一框架 =====")
        logger.info(f"任务类型: {args.task_type}")
        logger.info(f"数据路径: {args.data_path}")
        logger.info(f"置信度阈值: {args.confidence_threshold}")
        logger.info(f"最大迭代次数: {args.max_iterations}")
        logger.info(f"收敛阈值: {args.convergence_threshold}")
        logger.info(f"Top-k值: {args.top_k}")
        logger.info(f"是否使用监督模式: {args.supervision}")
        logger.info(f"模式对齐轮数: {args.schema_alignment_round}")
        logger.info(f"每轮模式对齐的百分比: {args.percentage_per_round}")
        logger.info("开始执行...")
        
        # 打印到控制台
        print("===== 基于Agent的知识融合统一框架 =====")
        print(f"任务类型: {args.task_type}")
        print(f"数据路径: {args.data_path}")
        print(f"置信度阈值: {args.confidence_threshold}")
        print(f"最大迭代次数: {args.max_iterations}")
        print(f"收敛阈值: {args.convergence_threshold}")
        print(f"Top-k值: {args.top_k}")
        print(f"是否使用监督模式: {args.supervision}")
        print(f"模式对齐轮数: {args.schema_alignment_round}")
        print(f"每轮模式对齐的百分比: {args.percentage_per_round}")
        print("开始执行...")
        print()
        
        # 计时开始
        start_time = time.time()
        
        # 1. 数据加载
        logger.info("[1/5] 开始数据加载...")
        print("[1/5] 开始数据加载...")
        try:
            loader = DataLoader(args.data_path, args.task_type)
            source_A_entities, source_B_entities, seed_alignments, valid_alignments, test_alignments = loader.load_data()
            # 过滤出有效的对齐（label=1）
            seed_alignments = [a for a in seed_alignments if a.label == 1]
            valid_alignments = [a for a in valid_alignments if a.label == 1]
            test_alignments = [a for a in test_alignments if a.label == 1]
            
            if not args.supervision:
                # 监督模式下，无种子对齐和验证对齐
                seed_alignments = []
                valid_alignments = []

            logger.info(f"数据加载完成: {len(source_A_entities)}个A实体, {len(source_B_entities)}个B实体")
            logger.info(f"种子对齐: {len(seed_alignments)}, 验证对齐: {len(valid_alignments)}, 测试对齐: {len(test_alignments)}")
            print(f"数据加载完成: {len(source_A_entities)}个A实体, {len(source_B_entities)}个B实体")
            print(f"种子对齐: {len(seed_alignments)}, 验证对齐: {len(valid_alignments)}, 测试对齐: {len(test_alignments)}")
        except Exception as e:
            logger.error(f"数据加载失败: {str(e)}")
            print(f"数据加载失败: {e}")
            return
        print()
        
        # 2. 数据分析
        logger.info("[2/5] 开始数据分析...")
        print("[2/5] 开始数据分析...")
        try:
            analyzer = DataAnalyzer(ontology_list_file='./ontology_list.txt', schema_alignment_round=args.schema_alignment_round)
            
            # 如果强制重新分析或者缓存不存在，则执行分析
            if args.force_analyze:
                # 如果需要强制重新分析，先删除可能存在的缓存文件
                save_path = analyzer.get_analysis_save_path(args.data_path) if hasattr(analyzer, 'get_analysis_save_path') else None
                if save_path and os.path.exists(save_path):
                    os.remove(save_path)
                    logger.info(f"已删除旧缓存文件: {save_path}")
                    print(f"已删除旧缓存文件: {save_path}")
                    
            # 分析数据
            report, schema_alignments, source_A_entities, source_B_entities = analyzer.analyze(source_A_entities, source_B_entities, args.data_path)
            logger.info(f"模态分析结果:")
            logger.info(f"  有名称: {report.has_names}")
            logger.info(f"  有关系: {report.has_relations}")
            logger.info(f"  有属性: {report.has_attributes}")
            logger.info(f"  有图片: {report.has_images}")
            logger.info(f"  是EA任务: {report.is_ea_task}")
            
            print(f"模态分析结果:")
            print(f"  有名称: {report.has_names}")
            print(f"  有关系: {report.has_relations}")
            print(f"  有属性: {report.has_attributes}")
            print(f"  有图片: {report.has_images}")
            print(f"  是EA任务: {report.is_ea_task}")
        except Exception as e:
            logger.error(f"数据分析失败: {str(e)}")
            print(f"数据分析失败: {e}")
            return
        print()
        
        # 3. 工具初始化
        logger.info("[3/5] 开始工具初始化...")
        print("[3/5] 开始工具初始化...")
        try:
            tools_manager = ToolsManager()
            # 传递实体信息给工具管理器，这样工具可以访问实体数据
            available_tools = tools_manager.get_available_tools(
                report, 
                args.data_path,
                source_A_entities,  # 传递实体信息
                source_B_entities   # 传递实体信息
            )
            logger.info(f"可用工具数量: {len(available_tools)}")
            print(f"可用工具数量: {len(available_tools)}")
            for tool in available_tools:
                logger.info(f"  - {tool.name} (类型: {tool.tool_type})")
                print(f"  - {tool.name}")
        except Exception as e:
            logger.error(f"工具初始化失败: {str(e)}")
            print(f"工具初始化失败: {e}")
            return
        print()
        # 4. 对齐执行
        logger.info("[4/5] 开始对齐执行...")
        print("[4/5] 开始对齐执行...")
        try:
            agent = AlignmentAgent(
                data_name=args.data_path.split('/')[-1],
                data_modal_report=report,
                confidence_threshold=args.confidence_threshold,
                max_iterations=args.max_iterations,
                convergence_threshold = args.convergence_threshold,
                top_k = args.top_k,
                percentage_per_round = args.percentage_per_round,
                data_path=args.data_path,  # 传递数据路径
                seed_alignments=seed_alignments,
                test_alignments=test_alignments,  # 传递测试对齐用于每轮评估
                supervision=args.supervision,  # 传递监督模式参数
                unsupervised_seed_count = 0.1
            )
            final_alignments = agent.run(
                source_A_entities, 
                source_B_entities, 
                available_tools
            )
            logger.info(f"对齐执行完成，最终对齐数: {len(final_alignments)}")
            print(f"对齐执行完成，最终对齐数: {len(final_alignments)}")
        except Exception as e:
            logger.error(f"对齐执行失败: {str(e)}")
            print(f"对齐执行失败: {e}")
            return
        print()
        
        # 5. 评估
        logger.info("[5/5] 开始评估...")
        print("[5/5] 开始评估...")
        try:
            # 读取种子对齐文件
            seed_alignments = []
            seed_alignments_file = "./output/round_0_alignments_all.txt"
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
                    logger.error(f"读取种子对齐文件失败: {e}")
                    print(f"读取种子对齐文件失败: {e}")
            
            # 评估最终对齐结果（排除种子对齐）
            precision, recall, f1_score = evaluate_alignments(final_alignments, seed_alignments, test_alignments)
            
            # 打印评估结果
            logger.info(f"评估结果 - 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1分数: {f1_score:.4f}")
            print_evaluation_results(precision, recall, f1_score)
            
            # 保存评估结果到文件
            with open("./output/evaluation_results.txt", 'w', encoding='utf-8') as f:
                f.write(f"精确率: {precision:.4f}\n")
                f.write(f"召回率: {recall:.4f}\n")
                f.write(f"F1分数: {f1_score:.4f}\n")
            logger.info("评估结果已保存到 ./output/evaluation_results.txt")
            print("评估结果已保存到 ./output/evaluation_results.txt")
        except Exception as e:
            logger.error(f"评估失败: {str(e)}")
            print(f"评估失败: {e}")
        
        # 计时结束
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"总耗时: {total_time:.2f} 秒")
        print(f"总耗时: {total_time:.2f} 秒")
        
        # 保存运行配置和统计信息
        stats = {
            "task_type": args.task_type,
            "data_path": args.data_path,
            "confidence_threshold": float(args.confidence_threshold),  # 转换为Python float
            "max_iterations": args.max_iterations,
            "total_time_seconds": float(total_time),  # 转换为Python float
            "entity_a_count": len(source_A_entities),
            "entity_b_count": len(source_B_entities),
            "seed_alignment_count": len(seed_alignments),
            "valid_alignment_count": len(valid_alignments),
            "test_alignment_count": len(test_alignments),
            "final_alignment_count": len(final_alignments)
        }
        
        with open("./output/run_stats.json", 'w', encoding='utf-8') as f:
            import json
            # 导入自定义JSON编码器
            from alignment_agent import CustomJSONEncoder
            json.dump(stats, f, ensure_ascii=False, indent=2, cls=CustomJSONEncoder)
        
        logger.info("运行完成！所有结果已保存到 ./output 目录")
        print("\n运行完成！所有结果已保存到 ./output 目录")
        
    except KeyboardInterrupt:
        logger.error("程序被用户中断")
        print("\n程序被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}")
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()