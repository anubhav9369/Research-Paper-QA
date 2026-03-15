# app.py — Research Paper Q&A Engine (RAG) 
import os
import json
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Streamlit Cloud secrets support
try:
    for key in ["GEMINI_API_KEY", "PINECONE_API_KEY", "GROQ_API_KEY"]:
        if key in st.secrets:
            os.environ[key] = st.secrets[key]
except Exception:
    pass

from src.pdf_parser import parse_paper, find_relevant_figures
from src.rag_pipeline import (
    get_embedder, get_pinecone_index, chunk_text,
    upsert_paper, semantic_search, make_paper_id, delete_paper
)
from src.llm_qa import get_groq_client, answer_question, generate_paper_summary

# ═══════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════
st.set_page_config(
    page_title="PaperMind — Research Paper Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background: #05080f !important;
    font-family: 'DM Sans', sans-serif !important;
    color: #e2e8f0;
}

.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 32px;
    background: rgba(10, 14, 26, 0.97);
    border-bottom: 1px solid #1a2540;
    position: sticky;
    top: 0;
    z-index: 100;
    margin: -1rem -1rem 0 -1rem;
}
.nav-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 18px;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.3px;
}
.nav-pill {
    background: #0d1a2e;
    border: 1px solid #1e3a5f44;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 11px;
    color: #60a5fa;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.card {
    background: #0a0e1a;
    border: 1px solid #1a2540;
    border-radius: 16px;
    padding: 24px;
}
.card-glow {
    background: linear-gradient(135deg, #0a1628 0%, #0a0e1a 100%);
    border: 1px solid #1e3a5f55;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 0 40px #1e3a5f22;
}
.sec-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #334155;
    display: block;
    margin-bottom: 10px;
}
.paper-card {
    background: #0a0e1a;
    border: 1px solid #1a2540;
    border-left: 3px solid #3b82f6;
    border-radius: 10px;
    padding: 14px 16px;
    margin: 6px 0;
}
.paper-card.active {
    border-left-color: #8b5cf6;
    background: #0f1a2e;
}
.kw-tag {
    display: inline-block;
    background: #0d1a2e;
    color: #60a5fa;
    border: 1px solid #1e3a5f;
    border-radius: 5px;
    padding: 3px 10px;
    margin: 2px;
    font-size: 11px;
    font-weight: 500;
    font-family: 'DM Mono', monospace;
}
.finding-item {
    padding: 8px 12px;
    background: #071a10;
    border: 1px solid #14532d;
    border-radius: 8px;
    margin: 4px 0;
    font-size: 13px;
    color: #86efac;
}
.source-chunk {
    background: #080c18;
    border: 1px solid #1a2540;
    border-left: 3px solid #7c3aed;
    border-radius: 8px;
    padding: 12px 14px;
    margin: 6px 0;
    font-size: 12px;
    color: #64748b;
    font-family: 'DM Mono', monospace;
    line-height: 1.6;
}
.score-badge {
    display: inline-block;
    background: #1a0d30;
    color: #a78bfa;
    border: 1px solid #4c1d95;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'DM Mono', monospace;
}
.diff-beginner { color: #22c55e; background: #052e16; border: 1px solid #166534; border-radius: 6px; padding: 2px 10px; font-size: 11px; font-weight: 600; }
.diff-intermediate { color: #f59e0b; background: #1c1000; border: 1px solid #78350f; border-radius: 6px; padding: 2px 10px; font-size: 11px; font-weight: 600; }
.diff-advanced { color: #ef4444; background: #2d0a0a; border: 1px solid #7f1d1d; border-radius: 6px; padding: 2px 10px; font-size: 11px; font-weight: 600; }

.fig-card {
    background: #0a0e1a;
    border: 1px solid #1a2540;
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 14px;
}
.fig-label {
    font-size: 11px;
    font-weight: 700;
    color: #3b82f6;
    letter-spacing: 1px;
    margin-bottom: 4px;
}
.fig-caption {
    font-size: 12px;
    color: #64748b;
    line-height: 1.5;
    margin-bottom: 10px;
}
.fig-match-banner {
    background: #0a1628;
    border: 1px solid #1e3a5f;
    border-left: 3px solid #3b82f6;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 11px;
    font-weight: 700;
    color: #60a5fa;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin: 10px 0 8px 0;
}

.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px #2563eb44 !important;
}

[data-testid="stChatMessage"] {
    background: #0a0e1a !important;
    border: 1px solid #1a2540 !important;
    border-radius: 14px !important;
    margin-bottom: 10px !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: #0d1428 !important;
    border-color: #1e3a5f !important;
}
[data-testid="stChatInput"] {
    background: #0a0e1a !important;
    border: 1px solid #1a2540 !important;
    border-radius: 14px !important;
}
[data-testid="stChatInput"]:focus-within { border-color: #2563eb !important; }
[data-testid="stChatInput"] textarea { background: transparent !important; color: #e2e8f0 !important; font-family: 'DM Sans', sans-serif !important; }

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #080c18 !important;
    border-radius: 12px !important;
    padding: 4px !important;
    border: 1px solid #1a2540 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    color: #475569 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    border: none !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: #1a2540 !important;
    color: #e2e8f0 !important;
}
[data-testid="stMetric"] {
    background: #0a0e1a !important;
    border: 1px solid #1a2540 !important;
    border-radius: 12px !important;
    padding: 16px !important;
}
[data-testid="stMetricLabel"] { color: #475569 !important; font-size: 11px !important; font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: 0.8px !important; }
[data-testid="stMetricValue"] { color: #e2e8f0 !important; font-size: 20px !important; font-weight: 700 !important; }
[data-testid="stProgress"] > div > div { background: linear-gradient(90deg, #2563eb, #7c3aed) !important; border-radius: 4px !important; }

hr { border-color: #1a2540 !important; margin: 20px 0 !important; }
#MainMenu, footer, .stDeployButton { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #05080f; }
::-webkit-scrollbar-thumb { background: #1a2540; border-radius: 2px; }
[data-testid="stSuccess"] { background: #052e16 !important; border: 1px solid #166534 !important; border-radius: 10px !important; }
[data-testid="stError"] { background: #2d0a0a !important; border: 1px solid #7f1d1d !important; border-radius: 10px !important; }
[data-testid="stInfo"] { background: #0c1a2e !important; border: 1px solid #1e3a5f !important; border-radius: 10px !important; }
[data-testid="stSelectbox"] > div > div { background: #0a0e1a !important; border-color: #1a2540 !important; color: #e2e8f0 !important; border-radius: 10px !important; }
[data-testid="stFileUploader"] > div { background: #080c18 !important; border: 2px dashed #1a2540 !important; border-radius: 14px !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════
DEFAULTS = {
    "papers": {},
    "active_paper_id": None,
    "messages": [],
    "embedder": None,
    "index": None,
    "groq_client": None,
    "initialized": False,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════════════════
# INIT CLIENTS (cached)
# ═══════════════════════════════════════════════════════
@st.cache_resource
def init_clients():
    embedder = get_embedder()
    index = get_pinecone_index("research-papers")
    groq = get_groq_client()
    return embedder, index, groq


# ═══════════════════════════════════════════════════════
# NAVBAR
# ═══════════════════════════════════════════════════════
st.markdown("""
<div class="navbar">
    <div class="nav-logo">
        <span style="font-size:22px;">📚</span>
        <span>Paper<span style="color:#3b82f6;">Mind</span></span>
    </div>
    <div style="display:flex; gap:8px; align-items:center;">
    </div>
</div>
<div style="height:24px"></div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# LANDING PAGE
# ═══════════════════════════════════════════════════════
if not st.session_state.papers:

    st.markdown("""
    <div style="text-align:center; padding:28px 0 36px 0;">
        <div style="font-size:12px; color:#3b82f6; font-weight:700; letter-spacing:2.5px;
                    text-transform:uppercase; margin-bottom:14px;">
            Retrieval Augmented Generation
        </div>
        <h1 style="font-size:44px; font-weight:700; color:#f1f5f9; margin:0 0 14px 0;
                   line-height:1.15; letter-spacing:-0.8px;">
            Ask Anything About<br/>
            <span style="background:linear-gradient(135deg,#3b82f6,#8b5cf6);
                         -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                Any Research Paper
            </span>
        </h1>
        <p style="color:#475569; font-size:16px; max-width:500px; margin:0 auto; line-height:1.6;">
            Upload a PDF → chunks get embedded into Pinecone →
            your questions retrieve the most relevant sections →
            LLaMA 3.1 generates precise answers with figures from the paper.
        </p>
    </div>
    """, unsafe_allow_html=True)

    pad_l, main_col, pad_r = st.columns([1, 2, 1])
    with main_col:
        st.markdown("<span class='sec-label'>📄 Upload Research Paper</span>",
                    unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload PDF", type=["pdf"],
                                    label_visibility="collapsed")

        if uploaded:
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            if st.button("🚀 Process & Index Paper", use_container_width=True):
                try:
                    embedder, index, groq_client = init_clients()
                    st.session_state.embedder = embedder
                    st.session_state.index = index
                    st.session_state.groq_client = groq_client

                    with st.status("Processing paper...", expanded=True) as status:
                        st.write("📄 Extracting text and figures from PDF...")
                        paper = parse_paper(uploaded, uploaded.name)

                        st.write(f"✂️ Chunking {paper.word_count:,} words...")
                        chunks = chunk_text(paper.full_text, chunk_size=500, overlap=100)

                        st.write(f"🧠 Generating embeddings for {len(chunks)} chunks...")
                        paper_id = make_paper_id(uploaded.name)
                        count = upsert_paper(
                            paper_id=paper_id,
                            paper_title=paper.title,
                            chunks=chunks,
                            embedder=embedder,
                            index=index
                        )

                        st.write("📊 Generating paper summary...")
                        summary = generate_paper_summary(paper.full_text, groq_client)

                        fig_count = len(paper.figures)
                        status.update(
                            label=f"✅ Paper indexed! {fig_count} figures extracted.",
                            state="complete"
                        )

                    st.session_state.papers[paper_id] = {
                        "title": paper.title,
                        "filename": uploaded.name,
                        "word_count": paper.word_count,
                        "chunk_count": len(chunks),
                        "vectors_upserted": count,
                        "summary": summary,
                        "figures": paper.figures,
                    }
                    st.session_state.active_paper_id = paper_id
                    st.session_state.messages = []
                    st.rerun()

                except Exception as e:
                    st.error(f"Error: {str(e)}")

    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)

    steps = st.columns(4)
    flow = [
        ("📄", "1. Parse", "PDF text + figures extracted with PyMuPDF"),
        ("✂️", "2. Chunk", "Text split into 500-word overlapping chunks"),
        ("🧠", "3. Embed", "Each chunk embedded into 384-dim vector"),
        ("🔍", "4. Retrieve", "Query finds top-K chunks + matching figures"),
    ]
    for col, (icon, title, desc) in zip(steps, flow):
        with col:
            st.markdown(f"""
            <div class="card" style="text-align:center; padding:20px;">
                <div style="font-size:28px; margin-bottom:8px;">{icon}</div>
                <div style="font-weight:600; color:#e2e8f0; font-size:14px; margin-bottom:6px;">{title}</div>
                <div style="color:#334155; font-size:12px; line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# MAIN DASHBOARD
# ═══════════════════════════════════════════════════════
else:
    if not st.session_state.embedder:
        try:
            embedder, index, groq_client = init_clients()
            st.session_state.embedder = embedder
            st.session_state.index = index
            st.session_state.groq_client = groq_client
        except Exception as e:
            st.error(f"Client init error: {e}")
            st.stop()

    active_id = st.session_state.active_paper_id
    active_paper = st.session_state.papers.get(active_id, {})
    summary = active_paper.get("summary", {})
    paper_figures = active_paper.get("figures", [])

    # ── Top bar ──
    top_l, top_r = st.columns([3, 1])
    with top_l:
        fig_count = len(paper_figures)
        st.markdown(f"""
        <div style="padding:4px 0 12px 0;">
            <div style="font-size:20px; font-weight:700; color:#f1f5f9; margin-bottom:2px;">
                {active_paper.get('title', 'Research Paper')[:80]}
            </div>
            <div style="font-size:12px; color:#334155;">
                {active_paper.get('filename', '')} ·
                {active_paper.get('word_count', 0):,} words ·
                {active_paper.get('chunk_count', 0)} chunks ·
                {active_paper.get('vectors_upserted', 0)} vectors ·
                <span style="color:#3b82f6;">{fig_count} figures extracted</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with top_r:
        if st.button("➕ Add Another Paper", use_container_width=True):
            st.session_state.papers = {}
            st.session_state.active_paper_id = None
            st.session_state.messages = []
            st.rerun()

    # ── Summary row ──
    if summary:
        s_col, k_col = st.columns([3, 2])
        with s_col:
            one_liner = summary.get("one_liner", "")
            problem = summary.get("problem", "")
            approach = summary.get("approach", "")
            diff = summary.get("difficulty", "Intermediate")
            field = summary.get("field", "")
            diff_class = f"diff-{diff.lower()}"

            st.markdown(f"""
            <div class="card-glow" style="margin-bottom:16px;">
                <span class="sec-label">Paper Overview</span>
                <div style="font-size:15px; color:#cbd5e1; font-weight:500;
                            margin-bottom:12px; line-height:1.5;">
                    {one_liner}
                </div>
                <div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:10px;">
                    <span class="{diff_class}">{diff}</span>
                    <span style="color:#60a5fa; font-size:12px; padding:2px 10px;
                                 background:#0d1a2e; border-radius:6px; border:1px solid #1e3a5f;">
                        {field}
                    </span>
                </div>
                {"<div style='font-size:13px; color:#64748b; margin-bottom:6px;'><strong style='color:#94a3b8;'>Problem:</strong> " + problem + "</div>" if problem else ""}
                {"<div style='font-size:13px; color:#64748b;'><strong style='color:#94a3b8;'>Approach:</strong> " + approach + "</div>" if approach else ""}
            </div>
            """, unsafe_allow_html=True)

        with k_col:
            findings = summary.get("key_findings", [])
            keywords = summary.get("keywords", [])
            if findings:
                st.markdown("<span class='sec-label'>Key Findings</span>",
                            unsafe_allow_html=True)
                for f in findings[:3]:
                    st.markdown(f"<div class='finding-item'>→ {f}</div>",
                                unsafe_allow_html=True)
            if keywords:
                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                kw_html = "".join([f"<span class='kw-tag'>{k}</span>" for k in keywords])
                st.markdown(kw_html, unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── Tabs ──
    tab_chat, tab_figures, tab_search, tab_stats = st.tabs([
        "💬  Ask Questions",
        f"🖼  Figures ({len(paper_figures)})",
        "🔍  Semantic Search",
        "📊  RAG Stats"
    ])

    # ─────────────────────────────────
    # TAB 1: CHAT
    # ─────────────────────────────────
    with tab_chat:
        if not st.session_state.messages:
            st.markdown("<span class='sec-label'>💡 Try these questions</span>",
                        unsafe_allow_html=True)
            suggestions = [
                "What problem does this paper solve?",
                "What is the main methodology or approach?",
                "What are the key results and findings?",
                "What datasets were used for evaluation?",
                "What are the limitations of this work?",
                "How does this compare to previous approaches?",
            ]
            cols = st.columns(2)
            for i, q in enumerate(suggestions):
                with cols[i % 2]:
                    st.markdown(f"""
                    <div style="background:#080c18; border:1px solid #1a2540; border-radius:8px;
                                padding:9px 14px; margin:3px 0; font-size:12px; color:#475569;">
                        💬 {q}
                    </div>""", unsafe_allow_html=True)
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # Chat history
        for msg in st.session_state.messages:
            avatar = "🧑‍💻" if msg["role"] == "user" else "📚"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    # Show saved matched figures
                    saved_figs = msg.get("matched_figures", [])
                    if saved_figs:
                        st.markdown(
                            "<div class='fig-match-banner'>📌 Related Figures from Paper</div>",
                            unsafe_allow_html=True
                        )
                        fig_cols = st.columns(len(saved_figs))
                        for col, fig in zip(fig_cols, saved_figs):
                            with col:
                                caption = f"{fig.fig_label}"
                                if fig.caption:
                                    caption += f" — {fig.caption}"
                                st.image(
                                    f"data:image/png;base64,{fig.image_base64}",
                                    caption=caption,
                                    use_column_width=True
                                )
                    if "sources" in msg:
                        with st.expander(f"📎 {len(msg['sources'])} source chunks retrieved"):
                            for src in msg["sources"]:
                                st.markdown(f"""
                                <div class="source-chunk">
                                    <span class="score-badge">score: {src['score']}</span>
                                    &nbsp; chunk #{src['chunk_index']}<br/><br/>
                                    {src['text'][:300]}...
                                </div>""", unsafe_allow_html=True)
                    if "meta" in msg:
                        m = msg["meta"]
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Latency", f"{m['latency_ms']:.0f}ms")
                        c2.metric("Tokens", f"{m['tokens_in']}→{m['tokens_out']}")
                        c3.metric("Sources", len(msg.get("sources", [])))

        # Chat input
        if prompt := st.chat_input("Ask anything about this paper..."):
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant", avatar="📚"):
                with st.spinner("Retrieving relevant sections..."):
                    chunks = semantic_search(
                        query=prompt,
                        embedder=st.session_state.embedder,
                        index=st.session_state.index,
                        top_k=5,
                        filter_paper_id=active_id
                    )
                with st.spinner("Generating answer..."):
                    response = answer_question(
                        question=prompt,
                        retrieved_chunks=chunks,
                        client=st.session_state.groq_client
                    )

                st.markdown(response.answer)

                # ── Auto figure matching ──
                matched_figs = find_relevant_figures(prompt, paper_figures, top_k=2)
                if matched_figs:
                    st.markdown(
                        "<div class='fig-match-banner'>📌 Related Figures from Paper</div>",
                        unsafe_allow_html=True
                    )
                    fig_cols = st.columns(len(matched_figs))
                    for col, fig in zip(fig_cols, matched_figs):
                        with col:
                            caption = f"{fig.fig_label}"
                            if fig.caption:
                                caption += f" — {fig.caption}"
                            st.image(
                                f"data:image/png;base64,{fig.image_base64}",
                                caption=caption,
                                use_column_width=True
                            )
                # ─────────────────────────

                with st.expander(f"📎 {len(chunks)} source chunks retrieved"):
                    for src in chunks:
                        st.markdown(f"""
                        <div class="source-chunk">
                            <span class="score-badge">score: {src['score']}</span>
                            &nbsp; chunk #{src['chunk_index']}<br/><br/>
                            {src['text'][:300]}...
                        </div>""", unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("Latency", f"{response.latency_ms:.0f}ms")
                c2.metric("Tokens", f"{response.tokens_in}→{response.tokens_out}")
                c3.metric("Sources", len(chunks))

            st.session_state.messages.append({
                "role": "assistant",
                "content": response.answer,
                "sources": chunks,
                "matched_figures": matched_figs,
                "meta": {
                    "latency_ms": response.latency_ms,
                    "tokens_in": response.tokens_in,
                    "tokens_out": response.tokens_out,
                }
            })

        if st.session_state.messages:
            if st.button("🗑 Clear Chat"):
                st.session_state.messages = []
                st.rerun()

    # ─────────────────────────────────
    # TAB 2: FIGURE GALLERY
    # ─────────────────────────────────
    with tab_figures:
        if not paper_figures:
            st.markdown("""
            <div style="text-align:center; padding:48px; color:#334155;">
                <div style="font-size:32px; margin-bottom:8px;">🖼</div>
                <div style="font-size:14px;">No figures found in this paper.</div>
                <div style="font-size:12px; margin-top:6px; color:#1e2d4a;">
                    Some PDFs store images in formats that can't be extracted.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div style='padding:8px 0 12px 0; font-size:13px; color:#475569;'>"
                f"{len(paper_figures)} figures extracted · Search or browse all</div>",
                unsafe_allow_html=True
            )

            display_figs = paper_figures

            for i in range(0, len(display_figs), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    if i + j < len(display_figs):
                        fig = display_figs[i + j]
                        with col:
                            with st.container():
                                st.markdown(f"""
                                <div class="fig-card">
                                    <div class="fig-label">{fig.fig_label} · Page {fig.page}</div>
                                    <div class="fig-caption">
                                        {fig.caption if fig.caption else "No caption detected"}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                st.image(
                                    f"data:image/png;base64,{fig.image_base64}",
                                    use_column_width=True
                                )
                                st.markdown("<div style='height:8px'></div>",
                                            unsafe_allow_html=True)

    # ─────────────────────────────────
    # TAB 3: SEMANTIC SEARCH
    # ─────────────────────────────────
    with tab_search:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("<span class='sec-label'>Search paper chunks by semantic similarity</span>",
                    unsafe_allow_html=True)

        search_col, k_col = st.columns([4, 1])
        with search_col:
            search_query = st.text_input(
                "Search Query",
                placeholder="e.g. attention mechanism, loss function, dataset...",
                label_visibility="collapsed"
            )
        with k_col:
            top_k = st.selectbox("Top K", [3, 5, 8, 10], index=1,
                                 label_visibility="collapsed")

        if search_query:
            with st.spinner("Searching..."):
                results = semantic_search(
                    query=search_query,
                    embedder=st.session_state.embedder,
                    index=st.session_state.index,
                    top_k=top_k,
                    filter_paper_id=active_id
                )

            st.markdown(f"<div style='font-size:12px; color:#334155; margin:10px 0;'>"
                        f"Found {len(results)} chunks</div>", unsafe_allow_html=True)

            for r in results:
                score_pct = int(r["score"] * 100)
                with st.expander(f"Chunk #{r['chunk_index']} — Score: {r['score']} ({score_pct}% match)"):
                    st.progress(r["score"])
                    st.markdown(f"""
                    <div class="source-chunk" style="max-height:300px; overflow-y:auto;">
                        {r['text']}
                    </div>""", unsafe_allow_html=True)

    # ─────────────────────────────────
    # TAB 4: RAG STATS
    # ─────────────────────────────────
    with tab_stats:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Papers Indexed", len(st.session_state.papers))
        c2.metric("Total Chunks", sum(p.get("chunk_count", 0) for p in st.session_state.papers.values()))
        c3.metric("Vectors in Pinecone", sum(p.get("vectors_upserted", 0) for p in st.session_state.papers.values()))
        c4.metric("Figures Extracted", sum(len(p.get("figures", [])) for p in st.session_state.papers.values()))

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown("<span class='sec-label'>Architecture</span>", unsafe_allow_html=True)
        st.markdown("""
        <div class="card" style="font-family:'DM Mono',monospace; font-size:12px;
                                  color:#475569; line-height:2;">
            PDF Upload → PyMuPDF text + figure extraction<br/>
            → Chunk (500 words, 100 overlap)<br/>
            → sentence-transformers all-MiniLM-L6-v2 (384 dimensions)<br/>
            → Pinecone upsert (cosine similarity index)<br/>
            ────────────────────────────────────<br/>
            Query → embed → Pinecone top-K search<br/>
            → Retrieved chunks + keyword-matched figures<br/>
            → LLaMA 3.1 (Groq) → grounded answer + figures shown
        </div>
        """, unsafe_allow_html=True)