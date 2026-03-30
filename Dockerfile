# 多阶段构建 Dockerfile
# 生产级容器化配置

# ========== 构建阶段 ==========
FROM python:3.11-slim-bookworm AS builder

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY pyproject.toml ./

# 创建虚拟环境并安装依赖
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 安装生产依赖
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ".[dev]"


# ========== 运行阶段 ==========
FROM python:3.11-slim-bookworm AS runtime

WORKDIR /app

# 环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    ENV="prod"

# 从构建阶段复制虚拟环境
COPY --from=builder /opt/venv /opt/venv

# 复制应用代码
COPY . .

# 创建非root用户
RUN groupadd -r railway && useradd -r -g railway railway && \
    mkdir -p /app/data /app/logs && \
    chown -R railway:railway /app

USER railway

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 暴露端口
EXPOSE 8000 9090

# 启动命令
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
