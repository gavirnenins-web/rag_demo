import asyncio
from pathlib import Path
import streamlit as st

from rag import build_index, index_exists, get_index_info, ask_agent

st.set_page_config(page_title="PDF Chat", page_icon="📄", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1050px; padding-top: 2rem;}
.title {font-size: 2rem; font-weight: 650; margin-bottom: 0.15rem;}
.sub {color: #777; margin-bottom: 1.5rem;}
.card {padding: 1rem 1.1rem; border: 1px solid #ddd; border-radius: 10px; background: #fafafa;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">PDF Chat</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Upload your documents and ask questions about them.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Documents")
    files = st.file_uploader(
        "Choose up to 5 PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if len(files) > 5:
        st.error("Please select no more than 5 PDFs.")
        files = files[:5]

    if files:
        st.write(f"{len(files)} PDF(s) selected")
        for f in files:
            st.caption(f"• {f.name}")

        if st.button("Build index", type="primary", use_container_width=True):
            with st.spinner("Reading PDFs and building the index..."):
                try:
                    result = build_index(files)
                    st.success(
                        f"Done. {result['pages']} pages, {result['chunks']} chunks indexed."
                    )
                    st.session_state.messages = []
                except Exception as e:
                    st.error(str(e))

    st.divider()
    if index_exists():
        info = get_index_info()
        st.caption(f"Indexed documents: {info['documents']}")
        st.caption(f"Stored chunks: {info['chunks']}")
        st.caption("Vector store: local ChromaDB")
        st.caption("Model: Ollama")
    else:
        st.caption("No index yet. Upload PDFs and click Build index.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if not index_exists():
    st.info("Start by uploading your PDFs in the sidebar.")

question = st.chat_input("Ask something about your PDFs...")

if question:
    if not index_exists():
        st.warning("Please build the index first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = asyncio.run(ask_agent(question))
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(str(e))
