import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import streamlit as st
from pathlib import Path
from rag.config import UPLOADS_DIR
from rag.pipeline import ingest_pdf
from rag.retriever import search
from rag.llm import answer_with_gemini
from rag.reranker import rerank

st.set_page_config(page_title="Intelligent Doc Q&A (RAG)", layout="wide")
st.title("📄 Intelligent Document Q&A (RAG)")

with st.sidebar:
    st.header("Upload")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded is not None:
        dest = UPLOADS_DIR / uploaded.name
        with open(dest, "wb") as f:
            f.write(uploaded.read())
        st.success(f"Saved: {dest.name}")
        with st.spinner("Ingesting... (extract → chunk → embed → index)"):
            chunks = ingest_pdf(dest)
        st.session_state['corpus'] = chunks
        st.success(f"Ingested {len(chunks)} chunks.")

st.subheader("Ask a question")
query = st.text_input("Your question")
btn = st.button("Search & Answer")

if btn and query.strip():
    corpus = st.session_state.get('corpus', None)
    if corpus is None:
        st.info("Please upload a PDF first to build the index for this session.")
    else:
        with st.spinner("Retrieving and reranking..."):
            results = search(query, corpus_texts=corpus)
            results = rerank(query, results, top_n=3)
        if not results:
            st.warning("No results found. Did you ingest a document this session?")
        else:
            st.write("### Top matches")
            for r in results:
                st.markdown(f"**{results.index(r) + 1}.** _{r['meta'].get('source','unknown')}_ — page {r['meta'].get('page', '?')}")
                full_text = r['text']
                first_period = full_text.find('. ')
                if first_period > 0 and first_period < 150:
                    clean_start = full_text[first_period + 2:]
                else:
                    clean_start = full_text
                excerpt = clean_start[:300]
                last_period = excerpt.rfind('.')
                display_text = excerpt[:last_period + 1] if last_period > 80 else excerpt
                st.caption(display_text)

            st.write("### LLM Answer")
            context = [r["text"] for r in results]
            ans = answer_with_gemini(query, context_chunks=context)
            st.success(ans)
