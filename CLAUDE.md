# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 🏗️ 高层面架构
**项目核心设计：分层架构Agent系统，解决纯LLM Agent延迟高、成本高的痛点**
```
用户请求 → LLM意图路由（提取意图+参数，正则兜底）→ 缓存层（命中直接返回）→ Pipeline层（简单查询硬编码）→ Agent层（复杂查询ReAct推理）
```
1. **核心调度**：`hybrid_agent.py`是分层架构入口，统一调度各层
2. **LLM路由**：路由层默认用轻量级LLM做意图识别+参数提取，支持任意句式，失败自动fallback到正则，简单查询覆盖率>90%
3. **分层原则**：90%的简单查询（直达车次/车次/车站信息查询）不需要调用大Agent，走Pipeline层直接返回，延迟<300ms；只有中转、多轮对话、推荐类复杂请求才走Agent层
4. **功能模块化**：所有业务功能都是独立Skill，存放在`skills/`目录下，新增功能不需要修改核心Agent代码
5. **知识图谱**：使用NetworkX内存图存储列车/车站信息，毫秒级查询

---

## 📦 常用命令
### 开发环境
```bash
# 安装依赖
pip install -r requirements.txt

# 启动开发服务（热重载）
uvicorn api_server:app --reload

# 运行测试
python -m pytest tests/
```

### 接口文档
- 服务启动后访问：http://localhost:8000/docs 查看Swagger UI
- 核心接口：POST `/api/v4/plan` 接收自然语言查询

### 容器部署
```bash
# 配置.env中的LLM_API_KEY后一键启动
docker-compose up -d
```

---

## ⚙️ 开发规则（项目特有）
1. **新增功能优先做Skill**：在`skills/`下新建目录实现，在`skill_loader.py`中注册，不要修改`agent.py`/`hybrid_agent.py`等核心文件
2. **路由规则修改**：简单查询的匹配规则在`core/router.py`中添加，匹配成功的请求会走Pipeline层，无需调用LLM
3. **缓存策略**：缓存TTL配置在`config.py`中，相同查询默认缓存5分钟
4. **LLM调用**：统一使用`core/retry.py`的`llm_retry`装饰器，自动指数退避重试
