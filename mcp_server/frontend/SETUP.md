# SketchKit Chat - 快速开始指南

## 📦 项目结构

```
frontend2/
├── src/
│   ├── api/              # API 接口
│   │   ├── chat-api.ts   # 对话 API
│   │   └── original-api.ts  # 复用原有功能 API
│   ├── assets/           # 静态资源
│   ├── components/
│   │   └── chat/         # 对话相关组件
│   │       ├── ChatMessage.vue      # 消息组件
│   │       ├── ChatInputBar.vue     # 输入框组件
│   │       └── ResultCard.vue       # 结果卡片组件
│   ├── stores/
│   │   └── chat.ts       # 对话状态管理
│   ├── types/
│   │   └── chat.ts       # TypeScript 类型定义
│   ├── views/
│   │   └── ChatView.vue  # 主聊天界面
│   ├── router/           # 路由配置
│   ├── App.vue
│   └── main.ts
├── package.json
├── vite.config.ts
└── README.md
```

## 🚀 启动步骤

### 1. 安装依赖

```bash
cd frontend2
npm install
```

### 2. 启动开发服务器

```bash
npm run dev
```

应用将在 `http://localhost:5174` 启动（注意端口是 5174，避免与原 frontend 的 5173 冲突）

### 3. 构建生产版本

```bash
npm run build
```

## 🎨 功能特性

### ✅ 已实现

1. **对话界面**
   - 左右对齐的用户/助手消息
   - 消息时间戳显示
   - 意图识别标签

2. **多模态输入**
   - 文本输入
   - 图片上传（拖拽/选择/粘贴）
   - 快捷操作按钮

3. **结果展示**
   - 图片预览
   - 视频播放
   - 3D 模型查看
   - 下载功能

4. **状态管理**
   - Pinia 状态管理
   - 消息历史记录
   - 加载状态处理

### 🔧 待实现（需要后端支持）

1. **后端 API**
   ```python
   POST /api/chat
   {
     "message": "帮我把这张图片转草图",
     "images": [File],
     "context": []
   }
   
   Response:
   {
     "reply": "我来帮你转换成草图...",
     "intent": "image2sketch",
     "result": {
       "type": "image",
       "preview": "data:image/png;base64,...",
       "download_url": "...",
       "method": "method_name"
     }
   }
   ```

2. **意图识别后端逻辑**
   - 使用 NLP 或 LLM 进行更准确的意图识别
   - 参数提取（如上色的 prompt）

3. **任务执行路由**
   - 根据识别的意图调用对应的模块
   - 整合现有的 10 个功能模块

## 🔌 集成现有模块

项目已经预留了与原有功能模块的集成接口，在 `src/api/original-api.ts` 中复用了原有的 API 函数：

- `runAnimation` - 动画生成
- `runSketchCleanup` - 草图清理
- `runVectorization` - 矢量化
- `runOrdering` - 笔画排序

## 🎯 下一步工作

### 前端优化
1. 添加消息编辑/删除功能
2. 实现会话历史管理
3. 添加参数调整面板
4. 优化移动端体验

### 后端集成
1. 实现 `/api/chat` 端点
2. 添加意图识别服务
3. 创建任务分发器
4. 实现结果缓存

## 📝 使用示例

### 基础对话

1. **转草图**
   ```
   用户: 帮我把这张图片转成草图
   [上传图片]
   
   助手: 我来帮你转换成草图...
   [显示处理结果]
   ```

2. **清理草图**
   ```
   用户: 清理这张草图
   [上传草图]
   
   助手: 正在清理草图...
   [显示清理后的结果]
   ```

3. **上色**
   ```
   用户: 给这个 SVG 上色，用温暖的色调
   [上传 SVG]
   
   助手: 正在为草图添加颜色...
   [显示上色结果]
   ```

## 🐛 调试

### 检查 API 连接

```bash
# 开发环境代理配置在 vite.config.ts
# /api/* 会被代理到 http://localhost:8000
```

### 查看控制台日志

浏览器开发者工具中可以看到：
- API 请求/响应
- 意图识别结果
- 错误信息

## 💡 提示

1. **图片格式**: 支持 PNG, JPG, SVG
2. **文件大小**: 建议不超过 10MB
3. **并发处理**: 目前不支持同时处理多个任务
4. **浏览器兼容**: 推荐使用 Chrome, Edge, Firefox 最新版本

## 📞 技术支持

如有问题，请查看：
- 浏览器控制台错误信息
- 网络请求状态
- 后端服务日志
