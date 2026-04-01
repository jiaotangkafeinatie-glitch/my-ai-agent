import streamlit as st
import chromadb
import fitz
import os
import json
from sentence_transformers import SentenceTransformer
from rapidocr_onnxruntime import RapidOCR
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.tools import DuckDuckGoSearchRun 

st.set_page_config(page_title="私人知识库 & 智能特工", page_icon="🤖", layout="wide") 

CHAT_FILE = "chat_history.json"

# ================= 0. 日记本功能 =================
def load_chat_history():
    if os.path.exists(CHAT_FILE):
        with open(CHAT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_chat_history(messages):
    with open(CHAT_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

# ================= 1. 模型加载 =================
@st.cache_resource
def load_core_systems():
    # 🚩 你的 DeepSeek API Key
    DEEPSEEK_KEY = "sk-a6d03b1fc88a4f53a89e2ac09f4068a4"
    
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path="./my_vector_db")
    collection = client.get_or_create_collection(name="pdf_knowledge_base")
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=DEEPSEEK_KEY,
        base_url="https://api.deepseek.com",
        max_tokens=1024
    )
    ocr_engine = RapidOCR() 
    return embed_model, collection, llm, ocr_engine

embed_model, collection, llm, ocr_engine = load_core_systems()

# ================= 2. 文件解析工具 =================
def extract_text_from_txt(txt_bytes):
    return txt_bytes.decode('utf-8', errors='ignore')

def extract_text_with_ocr(pdf_path):
    doc = fitz.open(pdf_path)
    all_text = ""
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=150)
        img_path = f"temp_page_{i}.png"
        pix.save(img_path)
        result, _ = ocr_engine(img_path)
        if result:
            for line in result:
                all_text += line[1] + "\n"
        if os.path.exists(img_path):
            os.remove(img_path)
    return all_text

def split_text(text, chunk_size=500):
    if not text:
        return []
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# ================= 3. 左侧边栏：管理中心 =================
with st.sidebar:
    st.title("📂 知识库管理")
    
    # --- 模块 A：上传新文件 ---
    uploaded_file = st.file_uploader("支持 TXT 和 PDF", type=['txt', 'pdf'])
    if st.button("🚀 开始解析并存入知识库"):
        if uploaded_file is not None:
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
                    ids = [f"{file_name}_chunk_{i}" for i in range(len(documents))]
                    metadatas = [{"source": file_name} for _ in range(len(documents))]
                    embeddings = embed_model.encode(documents).tolist()
                    collection.upsert(embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids)
                    st.success(f"✅ {file_name} 已入库")
                    st.rerun()
        else:
            st.warning("请先选择文件")

    st.markdown("---")
    
    # --- 模块 B & C：文件删除区 ---
    try:
        db_data = collection.get(include=["metadatas"])
        all_sources = list(set([meta["source"] for meta in db_data["metadatas"] if meta is not None]))
    except:
        all_sources = []

    if all_sources:
        st.subheader("⚠️ 文件清理")
        file_to_del = st.selectbox("选择要删除的文件：", all_sources, key="del_box")
        if st.button("🗑️ 确认从库中永久删除"):
            collection.delete(where={"source": file_to_del})
            st.warning(f"已删除：{file_to_del}")
            st.rerun()

    st.markdown("---")
    
    # --- 模块 D：✂️ 聊天记录精确清理区 ---
    st.subheader("✂️ 记忆清理")
    
    # 提取所有用户的提问，用来做下拉菜单
    if "messages" in st.session_state and st.session_state.messages:
        user_queries = []
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                # 截取前15个字作为标题显示
                display_text = f"问: {msg['content'][:15]}..." if len(msg['content']) > 15 else f"问: {msg['content']}"
                user_queries.append((i, display_text))
                
        if user_queries:
            # 制作一个字典，把显示的文本映射到它在列表里的真实索引
            query_options = {text: idx for idx, text in user_queries}
            selected_query = st.selectbox("选择要剔除的对话：", list(query_options.keys()))
            
            if st.button("🔪 仅删除选中对话"):
                idx = query_options[selected_query]
                
                # 删除用户的提问
                st.session_state.messages.pop(idx)
                # 紧接着判断下一条是不是 AI 的回答，如果是，连带 AI 的回答一并删掉
                if idx < len(st.session_state.messages) and st.session_state.messages[idx]["role"] == "assistant":
                    st.session_state.messages.pop(idx)
                    
                save_chat_history(st.session_state.messages) # 更新本地日记本
                st.rerun()

    if st.button("🧼 清空所有聊天记录"):
        st.session_state.messages = []
        save_chat_history([])
        st.rerun()
# ================= 4. 右侧主页面：聊天区 =================
st.title("🤖 联网智能特工 (Agentic RAG)")

if not all_sources:
    st.info("👈 知识库是空的，请先上传资料！")
    st.stop()

selected_file = st.selectbox("🎯 当前查询文档：", all_sources)

if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_question := st.chat_input(f"向 {selected_file} 提问..."):
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})
    save_chat_history(st.session_state.messages)

    with st.chat_message("assistant"):
        # 1. 检索与路由判断
        question_embedding = embed_model.encode([user_question]).tolist()
        results = collection.query(query_embeddings=question_embedding, n_results=1, where={"source": selected_file})
        retrieved_doc = results['documents'][0][0] if len(results['documents'][0]) > 0 else ""
        
        router_prompt = f"判断资料能否解答问题。能则输出YES，不能则输出NO。\n资料: {retrieved_doc}\n问题: {user_question}"
        decision = llm.invoke([HumanMessage(content=router_prompt)]).content
        
        if "NO" in decision.upper() or not retrieved_doc:
            st.warning("🕵️‍♂️ 本地无答案，正在联网搜索...")
            search_tool = DuckDuckGoSearchRun()
            final_context = search_tool.run(user_question)
            source_label = "🌐 全网搜索"
        else:
            st.success("📂 已锁定本地知识。")
            final_context = retrieved_doc
            source_label = f"📁 本地库 ({selected_file})"

        # 2. 生成回答
        llm_messages = [SystemMessage(content=f"基于资料回答，来源：{source_label}")]
        for msg in st.session_state.messages[:-1]:
            llm_messages.append(HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]))
        llm_messages.append(HumanMessage(content=f"资料: {final_context}\n问题: {user_question}"))
        
        stream_response = llm.stream(llm_messages)
        def extract_chunks(stream):
            for chunk in stream: yield chunk.content
        ai_response = st.write_stream(extract_chunks(stream_response))
        
        with st.expander(f"🔍 溯源原文 ({source_label})"):
            st.write(final_context)
    
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    save_chat_history(st.session_state.messages)