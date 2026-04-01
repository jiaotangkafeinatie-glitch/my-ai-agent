import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
from rapidocr_onnxruntime import RapidOCR
import os
import json
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from streamlit.runtime.scriptrunner import get_script_run_ctx

# ================= 1. 用户隔离逻辑 =================
def get_user_id():
    ctx = get_script_run_ctx()
    return ctx.session_id if ctx else "default_user"

USER_ID = get_user_id()
USER_DIR = f"storage/{USER_ID}"
DB_PATH = f"{USER_DIR}/vector_db"
HISTORY_PATH = f"{USER_DIR}/chat_history.json"

os.makedirs(DB_PATH, exist_ok=True)

# ================= 2. 系统核心加载 =================
@st.cache_resource
def load_global_assets():
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    ocr_engine = RapidOCR()
    return embed_model, ocr_engine

embed_model, ocr_engine = load_global_assets()

def get_user_session():
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name="user_docs")
    DEEPSEEK_KEY = st.secrets["DEEPSEEK_KEY"]
    llm = ChatOpenAI(
        model='deepseek-chat', 
        openai_api_key=DEEPSEEK_KEY, 
        openai_api_base='https://api.deepseek.com/v1',
        streaming=True
    )
    return collection, llm

collection, llm = get_user_session()

# ================= 3. 侧边栏：管理与介绍 =================
with st.sidebar:
    st.title("📂 个人私有知识库")
    
    # --- ✨ 详细的操作指南模块 ---
    with st.expander("📖 首次使用必看：操作说明", expanded=True):
        st.markdown(f"""
        **您的专属 ID:** `{USER_ID[:8]}`
        
        ### 🚀 快速上手
        1. **喂养 AI**：在下方上传 **TXT/PDF**，点击 **[存入我的库]**。
        2. **独立空间**：您的文件和对话都存在您的专属路径下，其他用户**完全不可见**。
        3. **精准对谈**：在主页面选择具体文件，AI 会优先从该文件中寻找答案。
        4. **溯源/联网**：
            - 📚 匹配成功：显示原文供核对。
            - 🌐 匹配失败：自动启动全网搜索补充知识。
        
        ---
        *注：免费服务器若长时间无人访问可能会重置，重要资料请在本地留存。*
        """)

    st.divider()

    # 文件上传
    uploaded_file = st.file_uploader("📤 上传新文档", type=['txt', 'pdf'])
    if st.button("🚀 存入我的库"):
        if uploaded_file:
            file_name = uploaded_file.name
            with st.spinner(f'正在解析 {file_name}...'):
                raw_text = ""
                if file_name.lower().endswith(".txt"):
                    raw_text = uploaded_file.getvalue().decode("utf-8")
                else:
                    temp_p = f"{USER_DIR}/temp_{file_name}"
                    with open(temp_p, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    doc = fitz.open(temp_p)
                    for page in doc:
                        raw_text += page.get_text()
                    os.remove(temp_p)
                
                if raw_text.strip():
                    from langchain_text_splitters import RecursiveCharacterTextSplitter
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                    chunks = text_splitter.split_text(raw_text)
                    ids = [f"{file_name}_{i}" for i in range(len(chunks))]
                    metas = [{"source": file_name} for _ in range(len(chunks))]
                    embs = embed_model.encode(chunks).tolist()
                    collection.upsert(embeddings=embs, documents=chunks, metadatas=metas, ids=ids)
                    st.success(f"✅ {file_name} 已就绪")
                    st.rerun()

    # 文件清理
    db_res = collection.get(include=["metadatas"])
    my_files = list(set([m["source"] for m in db_res["metadatas"] if m])) if db_res["metadatas"] else []
    
    if my_files:
        st.divider()
        st.subheader("🗑️ 库文件清理")
        to_del = st.selectbox("选择要删除的文件：", my_files)
        if st.button("从我的库中永久移除"):
            collection.delete(where={"source": to_del})
            st.rerun()

    if st.button("🧼 清空当前聊天记录"):
        if os.path.exists(HISTORY_PATH):
            os.remove(HISTORY_PATH)
        st.session_state.messages = []
        st.rerun()

# ================= 4. 对话中心 =================
st.title("🤖 个人私密 AI 特工")

if "messages" not in st.session_state:
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            st.session_state.messages = json.load(f)
    else:
        st.session_state.messages = []

# 筛选器
target_doc = st.selectbox("🎯 针对哪个文档提问？", ["全选"] + my_files)

# 历史渲染
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 交互逻辑
if prompt := st.chat_input("向我的特工提问..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 搜索与过滤
        search_filter = {"source": target_doc} if target_doc != "全选" else None
        res = collection.query(
            query_embeddings=embed_model.encode([prompt]).tolist(),
            n_results=3,
            where=search_filter
        )
        context = "\n".join(res['documents'][0]) if res['documents'] and res['documents'][0] else ""
        
        response = ""
        placeholder = st.empty()
        
        if not context.strip():
            st.warning("🕵️‍♂️ 本地无答案，正在全网搜索最新资料...")
            search_content = DuckDuckGoSearchRun().run(prompt)
            final_prompt = f"资料：{search_content}\n问题：{prompt}\n回答："
        else:
            with st.expander("📚 个人库溯源"):
                st.write(context)
            final_prompt = f"资料：{context}\n问题：{prompt}\n回答："

        for chunk in llm.stream(final_prompt):
            response += chunk.content
            placeholder.markdown(response + "▌")
        
        placeholder.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        # 实时保存历史
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(st.session_state.messages, f, ensure_ascii=False, indent=4)
