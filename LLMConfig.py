from typing import List, Optional, Dict, Any, Set
# 大模型统一配置
class LLMConfig:
    """大模型统一配置类"""
    
    # 默认配置
    DEFAULT_CONFIG = {
        "model": "Qwen3-8B",
        "api_key": "local-model-key",
        "api_base": "http://localhost:8000/v1",
        #"api_key": "sk-8ebAAD1ohQCKf7pGKDbGH5msuC48qI56U24OgXC3fGksBkjq",
        #"api_base": "https://api.chatfire.cn/v1",
        "temperature": 0.1,  # 默认温度
        "max_tokens": 100,    # 默认最大token数
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
    }
    
    # 特定场景配置（覆盖默认配置）
    SCENE_CONFIGS = {
        "meaningful_name_check": {
            "temperature": 0.1,
            "max_tokens": 10
        },
        "ontology_generation": {
            "temperature": 0.3,
            "max_tokens": 300
        },
        "schema_alignment": {
            "temperature": 0.7,
            "max_tokens": 100
        },
        "weight_assignment": {
            "temperature": 0.1,
            "max_tokens": 200
        },
        "tool_selection": {
            "temperature": 0.1,
            "max_tokens": 500
        },
        "threshold_assignment": {
            "temperature": 0.1,
            "max_tokens": 20
        }
    }
    
    @classmethod
    def get_config(cls, scene: str = None) -> Dict[str, Any]:
        """获取指定场景的配置"""
        config = cls.DEFAULT_CONFIG.copy()
        if scene and scene in cls.SCENE_CONFIGS:
            config.update(cls.SCENE_CONFIGS[scene])
        return config
    
    @classmethod
    def get_openai_client_config(cls, scene: str = None) -> Dict[str, Any]:
        """获取OpenAI客户端配置"""
        config = cls.get_config(scene)
        return {
            "api_key": config["api_key"],
            "base_url": config["api_base"]
        }
    
    @classmethod
    def get_completion_config(cls, scene: str = None) -> Dict[str, Any]:
        """获取对话补全配置"""
        config = cls.get_config(scene)
        return {
            "model": config["model"],
            "temperature": config["temperature"],
            "max_tokens": config["max_tokens"],
            "extra_body": config["extra_body"]
        }
