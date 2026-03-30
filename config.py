"""
精简版配置中心
保留核心配置项，去掉生产级非必要特性
"""
from typing import Optional, Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """LLM 相关配置"""
    model_config = SettingsConfigDict(env_prefix="LLM_")

    provider: Literal["zhipu", "openai", "volcengine"] = Field("zhipu", description="LLM 提供商")
    model: str = Field("glm-4-flash", description="模型名称")
    api_key: Optional[str] = Field(None, description="API Key")
    base_url: Optional[str] = Field("https://open.bigmodel.cn/api/paas/v4/", description="API 基础 URL")
    temperature: float = Field(0.1, ge=0.0, le=2.0, description="温度参数")
    max_tokens: int = Field(2048, description="最大输出 tokens")
    timeout: int = Field(60, description="请求超时时间（秒）")
    max_retries: int = Field(3, description="最大重试次数")


class CacheSettings(BaseSettings):
    """缓存层配置"""
    model_config = SettingsConfigDict(env_prefix="CACHE_")

    backend: Literal["memory", "redis"] = Field("memory", description="缓存后端")
    max_size: int = Field(1000, description="内存缓存最大容量")
    default_ttl: int = Field(300, description="默认缓存 TTL（秒）")
    query_ttl: int = Field(300, description="查询结果缓存 TTL（秒）")
    skill_ttl: int = Field(600, description="Skill 调用结果缓存 TTL（秒）")
    redis_url: str = Field("redis://localhost:6379/0", description="Redis 连接 URL（可选）")


class MemorySettings(BaseSettings):
    """记忆系统配置（简化版，只用内存）"""
    model_config = SettingsConfigDict(env_prefix="MEMORY_")
    max_trips: int = Field(20, description="保留的历史行程数")


class PipelineSettings(BaseSettings):
    """Pipeline 层配置"""
    model_config = SettingsConfigDict(env_prefix="PIPELINE_")
    enabled: bool = Field(True, description="是否启用 Pipeline 层")
    min_confidence: float = Field(0.8, description="路由到 Pipeline 的最低置信度")


class LoggingSettings(BaseSettings):
    """日志配置（简化版）"""
    model_config = SettingsConfigDict(env_prefix="LOG_")
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field("INFO", description="日志级别")


class APISettings(BaseSettings):
    """API 服务配置"""
    model_config = SettingsConfigDict(env_prefix="API_")
    host: str = Field("127.0.0.1", description="监听地址")
    port: int = Field(8000, description="监听端口")
    reload: bool = Field(False, description="是否启用热重载（开发环境）")


class Settings(BaseSettings):
    """全局配置入口"""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 环境标识
    env: Literal["dev", "prod"] = Field("dev", description="运行环境")
    debug: bool = Field(False, description="调试模式")

    # 子配置
    llm: LLMSettings = Field(default_factory=LLMSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    api: APISettings = Field(default_factory=APISettings)

    @field_validator("debug")
    @classmethod
    def auto_debug_for_dev(cls, v: bool, info) -> bool:
        if info.data.get("env") == "dev" and not v:
            return True
        return v


# 全局单例配置
settings = Settings()
