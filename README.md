# 🤖 Agentic RAG 私人知识库特工

这是一个基于大模型与检索增强生成（RAG）技术的全栈本地知识库应用。内置智能路由引擎，支持本地私有数据深度问答与全网实时搜索的无缝切换。

## ✨ 核心特性 (Features)

- **📄 强悍的文档解析引擎**：内置 `RapidOCR` 视觉模型，不仅支持 TXT，更能完美硬啃 PDF 及各类老旧扫描版图片内容。
- **🧠 智能体路由 (Agentic Router)**：具备自主决策能力。当判断本地知识库无法回答用户问题时，自动调用 `DuckDuckGo` 进行全网实时搜索并总结。
- **🗂️ 动态数据管理 (CRUD)**：基于 `ChromaDB` 向量数据库，引入 Metadata 标签技术，支持多文件隔离共存，并可在网页端一键永久删库。
- **✂️ 记忆“外科手术”**：持久化保存历史对话（本地 JSON），并首创“记忆修剪”功能，支持在侧边栏精细化剔除单条历史冗余对话，防止上下文污染。
- **💻 现代化流式 UI**：使用 `Streamlit` 构建极客风 Web 界面，支持文件拖拽解析、类似 ChatGPT 的打字机流式输出特效以及参考原文折叠溯源。

## 🛠️ 技术栈 (Tech Stack)

- **LLM 核心**：DeepSeek API / LangChain
- **向量化模型 (Embedding)**：SentenceTransformers (all-MiniLM-L6-v2)
- **向量数据库**：ChromaDB
- **文档解析与 OCR**：PyMuPDF (fitz) + RapidOCR
- **前端交互框架**：Streamlit

## 🚀 快速开始 (Quick Start)

### 1. 安装依赖环境
请确保您已安装 Python 环境，然后在终端运行以下命令安装核心组件：
```bash
pip install -r requirements.txt
