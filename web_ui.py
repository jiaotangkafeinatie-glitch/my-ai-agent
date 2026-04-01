import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
from rapidocr_onnxruntime import RapidOCR
import os
import json
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun

# ================= 1. 核心系统加载 =================
@st.cache_resource
def load_core_systems():
    # 秘密钥匙读取
    DEEPSEEK_KEY = st.secrets["DEEPSEEK_KEY"]
    
    # 向量库与模型
    client = chromadb.PersistentClient(path="./my_vector_db")
    collection = client.get_or_create_collection(name="my_docs")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # OCR 与 LLM
    ocr_engine = RapidOCR()
    llm = ChatOpenAI(
        model='deepseek-chat', 
        openai_api_key=DEEPSEEK_KEY, 
        openai_api_base='https://api.deepseek.com/v1',
        streaming=True
    )
    return collection, embed_model, ocr_engine, llm

collection, embed_model, ocr_engine, llm = load_core_systems()

# ================= 2. 功能函数库 =================
def extract_text_from_txt(file_bytes):
    return file_bytes.decode("utf-8")

def extract_text_with_ocr(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        img_list = page.get_images()
        if img_list:
            pix = page.get_pixmap()
            img_path = f"temp_page.png"
            pix.save(img_path)
            result, _ = ocr_engine(img_path)
            if result:
                full_text += "\n".join([line[1] for line in result])
            os.remove(img_path)
        else:
            full_text += page.get_text()
    return full_text

def split_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def save_chat_history(messages):
    with open("chat_history.json", "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=4)

def load_chat_history():
    if os.path.exists("chat_history.json"):
        with open("chat_history.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# ================= 3. 左侧边栏：管理中心 =================
with st.sidebar:
    st.title("📂 知识库管理")
    
    # --- 模块 A：使用指南 (放在侧边栏最上方) ---
    # --- 模块 A：使用指南 (深度增强版) ---
    with st.expander("🚀 快速上手：特工操作指南", expanded=True):
        st.markdown("""
        ### 1️⃣ 第一步：喂养 AI (入库)
        - 在下方选择 **TXT** 或 **PDF** 文件。
        - 点击 **[🚀 开始解析]**。
        - *💡 即使是图片格式的 PDF，我也能通过 OCR 视觉引擎读懂它。*

        ### 2️⃣ 第二步：精准定位 (筛选)
        - 在主界面中间的下拉菜单中选择你要聊的 **具体文件名**。
        - 选“全选”则会在所有已存文档中寻找答案。

        ### 3️⃣ 第三步：提问与溯源
        - **本地回答**：我会优先从你的文档里找答案，并提供 **[📚 溯源原文]** 供你核对。
        - **自动联网**：如果文档里没写，我会自动切换到 **[🌐 全网搜索]** 模式为你寻找最新资讯。

        ### 4️⃣ 第四步：记忆管理
        - 觉得我回答得不对？在左侧 **[✂️ 记忆清理]** 选择该条对话并删除，防止我被错误信息误导。
        
        ---
        **⚠️ 注意：** 云端环境为临时存储，重要文档请在本地备份。
        """)

    st.divider()

    # --- 模块 B：上传与解析 ---
    uploaded_file = st.file_uploader("支持 TXT 和 PDF", type=['txt', 'pdf'])
    if st.button("🚀 开始解析并存入知识库"):
        if uploaded_file:
            file_name = uploaded_file.name
            with st.spinner(f'正在解析 {file_name}...'):
                raw_text = ""
                if file_name.lower().endswith(".txt"):
                    raw_text = extract_text_from_txt(uploaded_file.getvalue())
                elif file_name.lower().endswith(".pdf"):
                    temp_pdf_path = f"temp_{file_name}"
                    with open(temp_pdf_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    raw_text = extract_text_with_ocr(temp_pdf_path)
                    os.remove(temp_pdf_path)
                
                if raw_text.strip():
                    documents = split_text(raw_text)
                    ids = [f"{file_name}_{i}" for i in range(len(documents))]
                    metadatas = [{"source": file_name} for _ in range(len(documents))]
                    embeddings = embed_model.encode(documents).tolist()
                    collection.upsert(embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids)
                    st.success(f"✅ {file_name} 已入库")
                    st.rerun()
        else:
            st.warning("请先选择文件")

    st.divider()
    
    # --- 模块 C：文件删除 ---
    try:
        db_data = collection.get(include=["metadatas"])
        all_sources = list(set([meta["source"] for meta in db_data["metadatas"] if meta]))
    except:
        all_sources = []

    if all_sources:
        st.subheader("⚠️ 文件清理")
        file_to_del = st.selectbox("选择要删除的文件：", all_sources)
        if st.button("🗑️ 确认从库中永久删除"):
            collection.delete(where={"source": file_to_del})
            st.warning(f"已删除：{file_to_del}")
            st.rerun()

    st.divider()
    
    # --- 模块 D：记忆清理 ---
    st.subheader("✂️ 记忆清理")
    if "messages" in st.session_state and st.session_state.messages:
        user_queries = []
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                display_text = f"问: {msg['content'][:15]}..."
                user_queries.append((i, display_text))
                
        if user_queries:
            query_options = {text: idx for idx, text in user_queries}
            selected_query = st.selectbox("选择要剔除的对话：", list(query_options.keys()))
            if st.button("🔪 仅删除选中对话框"):
                idx = query_options[selected_query]
                st.session_state.messages.pop(idx)
                if idx < len(st.session_state.messages) and st.session_state.messages[idx]["role"] == "assistant":
                    st.session_state.messages.pop(idx)
                save_chat_history(st.session_state.messages)
                st.rerun()

    if st.button("🧼 清空所有聊天记录"):
        st.session_state.messages = []
        save_chat_history([])
        st.rerun()

# ================= 4. 右侧主页面：对话中心 =================
st.title("🤖 私人知识库 & 智能特工")

if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

# 筛选器
selected_doc = st.selectbox("🎯 请选择要针对哪个文档进行提问：", ["全选"] + all_sources)

# 渲染历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 提问逻辑
if prompt := st.chat_input(f"向 {selected_doc} 提问..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 1. 向量搜索
        q_emb = embed_model.encode([prompt]).tolist()
        search_filter = {"source": selected_doc} if selected_doc != "全选" else None
        results = collection.query(query_embeddings=q_emb, n_results=3, where=search_filter)
        
        context = "\n".join(results['documents'][0]) if results['documents'] else ""
        
        # 2. 策略判断与回答
        response_content = ""
        placeholder = st.empty()
        
        if not context.strip():
            st.warning("🕵️‍♂️ 本地无答案，正在联网搜索...")
            search_tool = DuckDuckGoSearchRun()
            context = search_tool.run(prompt)
            final_prompt = f"用户问题：{prompt}\n\n联网搜索结果：{context}\n\n请根据搜索结果回答。"
        else:
            with st.expander("📚 溯源原文（已匹配本地知识库）"):
                st.write(context)
            final_prompt = f"本地知识库内容：{context}\n\n用户提问：{prompt}\n\n请基于库内容回答。"

        for chunk in llm.stream(final_prompt):
            response_content += chunk.content
            placeholder.markdown(response_content + "▌")
        
        placeholder.markdown(response_content)
        st.session_state.messages.append({"role": "assistant", "content": response_content})
        save_chat_history(st.session_state.messages)
        
        with st.expander(f"🔍 溯源原文 ({source_label})"):
            st.write(final_context)
    
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    save_chat_history(st.session_state.messages)
