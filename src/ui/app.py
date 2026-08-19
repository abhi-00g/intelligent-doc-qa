import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import streamlit as st
from pathlib import Path
from rag.config import UPLOADS_DIR
from rag.pipeline import ingest_pdf, IngestionError
from rag.retriever import search
from rag.llm import answer_with_gemini
from rag.reranker import rerank
from rag.telemetry import get_session_stats, TELEMETRY_FILE

st.set_page_config(page_title="Intelligent Doc Q&A (RAG)", layout="wide")
st.title("\U0001f4c4 Intelligent Document Q&A (RAG)")

# ── Sidebar: Upload + Telemetry ──────────────────────────────
with st.sidebar:
    st.header("Upload")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded is not None:
        dest = UPLOADS_DIR / uploaded.name
        with open(dest, "wb") as f:
            f.write(uploaded.read())
        st.success(f"Saved: {dest.name}")

        try:
            with st.spinner("Ingesting... (extract \u2192 chunk \u2192 embed \u2192 index)"):
                chunks = ingest_pdf(dest)
            st.session_state["corpus"] = chunks
            st.success(f"Ingested {len(chunks)} chunks.")
        except IngestionError as e:
            st.error(str(e))

    # ── Telemetry Stats ──
    st.divider()
    st.header("\U0001f4ca Usage & Cost")
    stats = get_session_stats()

    if stats["total_calls"] > 0:
        col1, col2 = st.columns(2)
        col1.metric("API Calls", stats["total_calls"])
        col2.metric("Total Cost", f"${stats['total_cost_usd']:.4f}")

        col3, col4 = st.columns(2)
        col3.metric("Total Tokens", f"{stats['total_tokens']:,}")
        col4.metric("Avg Latency", f"{stats['avg_latency_ms']:.0f}ms")

        with st.expander("Detailed breakdown"):
            st.write(f"**Input tokens:** {stats['total_input_tokens']:,}")
            st.write(f"**Output tokens:** {stats['total_output_tokens']:,}")
            st.write(f"**Successful:** {stats['successful_calls']}")
            st.write(f"**Failed:** {stats['failed_calls']}")

        if st.button("Reset stats"):
            if TELEMETRY_FILE.exists():
                TELEMETRY_FILE.unlink()
            st.rerun()
    else:
        st.caption("No API calls yet. Ask a question to start tracking.")


# ── Main: Q&A ────────────────────────────────────────────────
st.subheader("Ask a question")
query = st.text_input("Your question")
btn = st.button("Search & Answer")

if btn and query.strip():
    corpus = st.session_state.get("corpus", None)

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
                page = r["meta"].get("page", "?")
                source = r["meta"].get("source", "unknown")
                score = r.get("rerank_score", None)
                score_str = f" (relevance: {score:.2f})" if score is not None else ""

                st.markdown(
                    f"**{results.index(r) + 1}.** _{source}_ \u2014 page {page}{score_str}"
                )

                full_text = r["text"]
                first_period = full_text.find(". ")
                if 0 < first_period < 150:
                    clean_start = full_text[first_period + 2 :]
                else:
                    clean_start = full_text
                excerpt = clean_start[:300]
                last_period = excerpt.rfind(".")
                display_text = excerpt[: last_period + 1] if last_period > 80 else excerpt
                st.caption(display_text)

            st.write("### LLM Answer")
            context = [r["text"] for r in results]
            ans = answer_with_gemini(query, context_chunks=context)
            st.success(ans)

            # Show per-query telemetry inline
            latest = get_session_stats()
            st.caption(
                f"\U0001f4ca {latest['total_tokens']:,} total tokens \u2022 "
                f"${latest['total_cost_usd']:.4f} total cost \u2022 "
                f"{latest['avg_latency_ms']:.0f}ms avg latency"
            )