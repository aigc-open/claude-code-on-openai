# Claude Code on OpenAI

将 Claude API 请求代理到 OpenAI 兼容的大模型服务

## 🚀 功能特性

- **智能模型映射**: 自动将 Claude 模型名称映射到 OpenAI 模型
- **流式响应支持**: 完全兼容 Claude API 的流式响应格式
- **工具调用支持**: 完整支持 Function Calling 功能
- **OpenAI 专用**: 专门针对 OpenAI API 优化
- **自定义端点**: 支持通过 `OPENAI_BASE_URL` 配置自定义 OpenAI 兼容端点

## 📡 API 接口

### 新的路由结构

```
GET  /claude_code/                           # 项目主页
POST /claude_code/{model_name}/v1/messages  # 创建消息
```

**注意**: 所有模型都通过路径参数 `{model_name}` 获取，不再依赖预定义的模型列表。

### 兼容性接口

```
POST /v1/messages  # 原始接口，保持向后兼容
```

## 🔧 配置

复制 `.env.example` 到 `.env` 并配置相应的 API 密钥：

```bash
# OpenAI API 配置
OPENAI_BASE_URL=https://your-custom-openai-endpoint.com/v1  # 可选，默认使用 OpenAI 官方端点

# 可选配置（如果需要自定义 OpenAI 端点）
```

**注意**: API 密钥现在通过请求头传递，不再使用环境变量。

## 🏃‍♂️ 运行

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
python -m claude_code_on_openai.server

# 或使用 uvicorn
uvicorn claude_code_on_openai:app --host 0.0.0.0 --port 8082
```

## 🛠️ 使用 claude-code 工具

### 环境变量配置

```bash
ANTHROPIC_BASE_URL=http://localhost:8082/claude_code/qwen2.5-7B
ANTHROPIC_API_KEY=sk-xxx
```

### 安装 claude-code

```bash
npm install -g @anthropic-ai/claude-code
```

### 启动 claude-code

```bash
ANTHROPIC_BASE_URL=http://localhost:8082/claude_code/gpt-4o ANTHROPIC_API_KEY=sk-xxxxxx claude
```

## 📝 使用示例

```bash
# 使用 x-api-key 头
curl -X POST "http://localhost:8082/claude_code/qwen2.5-7B/v1/messages" \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-openai-api-key" \
  -d '{
    "model": "claude-3-sonnet-20240229",
    "max_tokens": 1024,
    "messages": [
      {
        "role": "user",
        "content": "Hello, how are you?"
      }
    ]
  }'

# 或使用 Authorization Bearer 头
curl -X POST "http://localhost:8082/claude_code/gpt-3.5-turbo/v1/messages" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-xxxxx" \
  -d '{
    "model": "gpt-4o",
    "max_tokens": 1024,
    "messages": [
      {
        "role": "user",
        "content": "你好"
      }
    ]
  }'
```

## 🎯 模型映射

所有模型都映射到 OpenAI API：

- **所有模型**: 统一使用 OpenAI API
- **直接路由**: 通过 URL 路径中的模型名直接使用对应的 OpenAI 模型

## 🌐 访问主页

启动服务后，访问 `http://localhost:8082/claude_code/` 查看详细的使用说明和 API 文档。
