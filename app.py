import streamlit as st
import time
from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_all
from core.rag import build_rag_chain, ask_question

load_dotenv()

st.set_page_config(
    page_title="Meeting Lens",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
    --bg:           #0a0a0f;
    --surface:      #111118;
    --surface-2:    #1a1a25;
    --border:       #2a2a3a;
    --accent:       #7c3aed;
    --accent-glow:  #9f67ff;
    --accent-2:     #06b6d4;
    --text:         #e8e8f0;
    --text-muted:   #7070a0;
    --success:      #10b981;
    --warning:      #f59e0b;
    --danger:       #ef4444;
    --hindi:        #f97316;
}

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background: var(--bg) !important; }

.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        linear-gradient(rgba(124,58,237,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(124,58,237,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

h1,h2,h3,h4,h5,h6 {
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 800;
    line-height: 1.1;
    margin: 0;
    background: linear-gradient(135deg, #ffffff 0%, var(--accent-glow) 50%, var(--accent-2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.card:hover { border-color: var(--accent); }
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, var(--accent), var(--accent-2));
}
.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.card-content {
    font-size: 0.875rem;
    line-height: 1.7;
    color: var(--text);
}

/* ── Badges ── */
.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.badge-purple { background: rgba(124,58,237,0.2); color: var(--accent-glow); border: 1px solid rgba(124,58,237,0.3); }
.badge-cyan   { background: rgba(6,182,212,0.15);  color: var(--accent-2);   border: 1px solid rgba(6,182,212,0.3); }
.badge-green  { background: rgba(16,185,129,0.15); color: var(--success);    border: 1px solid rgba(16,185,129,0.3); }
.badge-orange { background: rgba(249,115,22,0.15); color: var(--hindi);      border: 1px solid rgba(249,115,22,0.3); }

/* ── Inputs / Buttons ── */
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.2) !important;
}
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #5b21b6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(124,58,237,0.4) !important;
}
.stButton > button[kind="secondary"] {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
}

/* ── Sidebar pipeline dots ── */
.status-bar {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.6rem 0.75rem;
    background: var(--surface-2);
    border-radius: 8px;
    margin: 0.3rem 0;
    border: 1px solid var(--border);
    font-size: 0.78rem;
}
.status-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.dot-active  { background: var(--accent-glow); box-shadow: 0 0 8px var(--accent-glow); animation: pulse 1.5s infinite; }
.dot-done    { background: var(--success); }
.dot-pending { background: var(--border); }

/* ── Live pipeline tracker (main area) ── */
.pipeline-tracker {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.25rem;
}
.pipeline-tracker-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 1.25rem;
}
.pipeline-row {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    margin-bottom: 0.85rem;
    position: relative;
}
.pipeline-row:not(:last-child)::after {
    content: '';
    position: absolute;
    left: 11px;
    top: 24px;
    width: 2px;
    height: calc(100% + 0.1rem);
    background: var(--border);
}
.pipeline-row.done-row::after { background: var(--success); opacity: 0.4; }
.pipeline-row.active-row::after { background: var(--accent); opacity: 0.3; }

.pipeline-icon {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.65rem;
    flex-shrink: 0;
    z-index: 1;
    font-weight: 700;
    border: 2px solid var(--border);
}
.icon-pending { background: var(--surface-2); color: var(--text-muted); border-color: var(--border); }
.icon-active  { background: rgba(124,58,237,0.2); color: var(--accent-glow); border-color: var(--accent); animation: pulse 1.5s infinite; }
.icon-done    { background: rgba(16,185,129,0.15); color: var(--success); border-color: var(--success); }

.pipeline-text { padding-top: 2px; }
.pipeline-label {
    font-size: 0.82rem;
    font-weight: 500;
    color: var(--text);
    font-family: 'Syne', sans-serif;
}
.pipeline-label.muted { color: var(--text-muted); }
.pipeline-detail {
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 0.1rem;
    font-family: 'JetBrains Mono', monospace;
}
.pipeline-detail.active-detail { color: var(--accent-glow); }
.pipeline-detail.done-detail   { color: var(--success); }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}

/* ── Chat ── */
.chat-container {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    max-height: 420px;
    overflow-y: auto;
    margin-bottom: 1rem;
}
.chat-msg { margin-bottom: 1rem; display: flex; flex-direction: column; gap: 0.2rem; }
.chat-label { font-size: 0.65rem; font-weight: 700; letter-spacing: 0.15em; text-transform: uppercase; }
.chat-bubble { display: inline-block; padding: 0.6rem 1rem; border-radius: 10px; font-size: 0.85rem; line-height: 1.6; max-width: 90%; }
.user-label  { color: var(--accent-glow); }
.bot-label   { color: var(--accent-2); }
.user-bubble { background: rgba(124,58,237,0.15); border: 1px solid rgba(124,58,237,0.25); align-self: flex-end; }
.bot-bubble  { background: rgba(6,182,212,0.1);   border: 1px solid rgba(6,182,212,0.2);  align-self: flex-start; }

hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1.5rem 0 !important; }

.transcript-box {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.25rem;
    font-size: 0.82rem;
    line-height: 1.8;
    max-height: 300px;
    overflow-y: auto;
    color: var(--text-muted);
    white-space: pre-wrap;
    word-break: break-word;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--surface-2) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}

.stProgress > div > div > div { background: var(--accent) !important; }
.stSpinner > div { border-top-color: var(--accent) !important; }
[data-testid="stMarkdownContainer"] p { color: var(--text) !important; }
label { color: var(--text-muted) !important; font-size: 0.8rem !important; }

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
</style>
""", unsafe_allow_html=True)

# ─── Session State ───────────────────────────────────────────────────────────────
for key, default in {
    "result": None,
    "chat_history": [],
    "processing": False,
    "pipeline_done": False,
    "pipeline_steps": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Pipeline step config ────────────────────────────────────────────────────────
STEPS = [
    ("audio",      "🔊", "Audio Processing",  "Converting file to WAV chunks"),
    ("transcript", "📝", "Transcription",     "Running Whisper on audio chunks"),
    ("title",      "🏷️",  "Title Generation",  "Extracting keywords"),
    ("summary",    "📋", "Summarisation",     "Map-reduce summarisation"),
    ("extract",    "🔍", "Extraction",        "Action items · Decisions · Questions"),
    ("rag",        "🧠", "RAG Engine",        "Building vector store"),
]

STEP_DETAILS = {
    "audio":      {"active": "Converting & chunking audio...", "done": "Audio ready"},
    "transcript": {"active": "Transcribing with Whisper...",   "done": "Transcript complete"},
    "title":      {"active": "Extracting keywords...",         "done": "Title generated"},
    "summary":    {"active": "Summarising transcript...",      "done": "Summary ready"},
    "extract":    {"active": "Extracting insights...",         "done": "Action items, decisions & questions extracted"},
    "rag":        {"active": "Building vector store...",       "done": "RAG engine ready — chat enabled"},
}

# ─── Helpers ────────────────────────────────────────────────────────────────────
def step_css(key):
    s = st.session_state.pipeline_steps.get(key, "pending")
    if s == "active": return "dot-active"
    if s == "done":   return "dot-done"
    return "dot-pending"

def render_sidebar_step(label, key, icon):
    css = step_css(key)
    st.markdown(f"""
    <div class="status-bar">
        <div class="status-dot {css}"></div>
        <span>{icon} {label}</span>
    </div>""", unsafe_allow_html=True)

def render_pipeline_tracker():
    steps_state = st.session_state.pipeline_steps
    html = '<div class="pipeline-tracker">'
    html += '<div class="pipeline-tracker-title">⚡ Pipeline Progress</div>'

    for key, icon, label, _ in STEPS:
        state = steps_state.get(key, "pending")
        detail_map = STEP_DETAILS[key]

        if state == "done":
            icon_css  = "icon-done"
            row_css   = "done-row"
            label_css = ""
            detail    = detail_map["done"]
            detail_css = "done-detail"
            marker    = "✓"
        elif state == "active":
            icon_css  = "icon-active"
            row_css   = "active-row"
            label_css = ""
            detail    = detail_map["active"]
            detail_css = "active-detail"
            marker    = icon
        else:
            icon_css  = "icon-pending"
            row_css   = ""
            label_css = "muted"
            detail    = "Waiting..."
            detail_css = ""
            marker    = "○"

        html += f"""
        <div class="pipeline-row {row_css}">
            <div class="pipeline-icon {icon_css}">{marker}</div>
            <div class="pipeline-text">
                <div class="pipeline-label {label_css}">{label}</div>
                <div class="pipeline-detail {detail_css}">{detail}</div>
            </div>
        </div>"""

    html += '</div>'
    return html

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;
                background:linear-gradient(135deg,#fff 0%,#9f67ff 60%,#06b6d4 100%);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                background-clip:text;margin-bottom:0.25rem;">
        ◎ Meeting Lens
    </div>
    <div style="font-size:0.7rem;color:#7070a0;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:1.5rem;">
        Audio Intelligence
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="badge badge-purple">Upload</span>', unsafe_allow_html=True)
    st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Audio or video file",
        type=["mp3", "mp4", "wav", "m4a", "webm", "mkv"],
        label_visibility="collapsed",
    )

    source = None
    if uploaded_file:
        save_path = f"/tmp/{uploaded_file.name}"
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())
        source = save_path
        st.markdown(f"""
        <div style="margin-top:0.5rem;padding:0.5rem 0.75rem;background:rgba(16,185,129,0.1);
                    border:1px solid rgba(16,185,129,0.25);border-radius:6px;font-size:0.75rem;color:#10b981;">
            ✓ {uploaded_file.name}
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
    st.markdown('<span class="badge badge-cyan">Language</span>', unsafe_allow_html=True)
    st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)

    language = st.selectbox(
        "Language",
        ["english", "hindi"],
        index=0,
        label_visibility="collapsed",
        format_func=lambda x: "🇬🇧 English" if x == "english" else "🇮🇳 Hindi",
    )

    if language == "hindi":
        st.markdown("""
        <div style="margin-top:0.4rem;padding:0.5rem 0.75rem;background:rgba(249,115,22,0.1);
                    border:1px solid rgba(249,115,22,0.25);border-radius:6px;font-size:0.72rem;color:#f97316;">
            Powered by Sarvam AI — summary & extraction in Hindi
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.25rem'></div>", unsafe_allow_html=True)
    run_btn = st.button("⚡  Analyse", use_container_width=True)

    if st.session_state.pipeline_done:
        st.markdown("---")
        st.markdown('<span class="badge badge-green">Pipeline Status</span>', unsafe_allow_html=True)
        st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)
        for key, icon, label, _ in STEPS:
            render_sidebar_step(label, key, icon)

# ─── Main ───────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Meeting Lens</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Transcribe · Summarise · Chat with your meetings</div>', unsafe_allow_html=True)
st.markdown("---")

# ─── Pipeline ───────────────────────────────────────────────────────────────────
if run_btn:
    if not source:
        st.error("Please upload an audio or video file to continue.")
    else:
        st.session_state.pipeline_done = False
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {}

        tracker_placeholder = st.empty()

        def update_step(key, state):
            st.session_state.pipeline_steps[key] = state
            tracker_placeholder.markdown(render_pipeline_tracker(), unsafe_allow_html=True)

        try:
            # Show initial tracker
            tracker_placeholder.markdown(render_pipeline_tracker(), unsafe_allow_html=True)

            update_step("audio", "active")
            chunks = process_input(source)
            update_step("audio", "done")

            update_step("transcript", "active")
            transcript = transcribe_all(chunks, language)
            update_step("transcript", "done")

            update_step("title", "active")
            title = generate_title(transcript)
            update_step("title", "done")

            update_step("summary", "active")
            summary = summarize(transcript, language)
            update_step("summary", "done")

            update_step("extract", "active")
            extracted    = extract_all(transcript, language)
            action_items = extracted["action_items"]
            decisions    = extracted["key_decisions"]
            questions    = extracted["open_questions"]
            update_step("extract", "done")

            update_step("rag", "active")
            rag_chain = build_rag_chain(transcript)
            update_step("rag", "done")

            st.session_state.result = {
                "title":          title,
                "transcript":     transcript,
                "summary":        summary,
                "action_items":   action_items,
                "key_decisions":  decisions,
                "open_questions": questions,
                "rag_chain":      rag_chain,
                "language":       language,
            }
            st.session_state.pipeline_done = True

            # Show completed tracker briefly then rerun
            tracker_placeholder.markdown(render_pipeline_tracker(), unsafe_allow_html=True)
            time.sleep(1)
            st.rerun()

        except Exception as e:
            for k, *_ in STEPS:
                if st.session_state.pipeline_steps.get(k) == "active":
                    st.session_state.pipeline_steps[k] = "pending"
            tracker_placeholder.empty()
            st.error(f"❌ Error: {e}")

# ─── Results ────────────────────────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result
    is_hindi = r.get("language") == "hindi"

    # Title banner
    lang_badge = '<span class="badge badge-orange" style="margin-left:0.75rem">हिंदी</span>' if is_hindi else ""
    st.markdown(f"""
    <div class="card">
        <div class="card-title">📌 Session Title</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:700;color:var(--text);display:flex;align-items:center;flex-wrap:wrap;gap:0.5rem;">
            {r['title']}{lang_badge}
        </div>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2], gap="medium")

    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">📋 {'सारांश' if is_hindi else 'Summary'}</div>
            <div class="card-content">{r['summary']}</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        with st.expander("📝 Full Transcript", expanded=False):
            st.markdown(f'<div class="transcript-box">{r["transcript"]}</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">✅ {'कार्य सूची' if is_hindi else 'Action Items'}</div>
            <div class="card-content">{r['action_items']}</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">🔑 {'मुख्य निर्णय' if is_hindi else 'Key Decisions'}</div>
            <div class="card-content">{r['key_decisions']}</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">❓ {'खुले प्रश्न' if is_hindi else 'Open Questions'}</div>
            <div class="card-content">{r['open_questions']}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    chat_title = "💬 बैठक से चैट करें" if is_hindi else "💬 Chat with your Meeting"
    st.markdown(f'<div style="font-family:\'Syne\',sans-serif;font-size:1.2rem;font-weight:700;margin-bottom:1rem">{chat_title}</div>', unsafe_allow_html=True)

    if st.session_state.chat_history:
        chat_html = '<div class="chat-container">'
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                chat_html += f"""
                <div class="chat-msg" style="align-items:flex-end">
                    <span class="chat-label user-label">You</span>
                    <div class="chat-bubble user-bubble">{msg['content']}</div>
                </div>"""
            else:
                chat_html += f"""
                <div class="chat-msg" style="align-items:flex-start">
                    <span class="chat-label bot-label">🤖 Assistant</span>
                    <div class="chat-bubble bot-bubble">{msg['content']}</div>
                </div>"""
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)
    else:
        placeholder_text = "बैठक के बारे में कुछ भी पूछें" if is_hindi else "Ask anything about your meeting transcript"
        st.markdown(f"""
        <div class="card" style="text-align:center;padding:2rem">
            <div style="font-size:2rem;margin-bottom:0.5rem">💬</div>
            <div style="color:var(--text-muted);font-size:0.85rem">{placeholder_text}</div>
        </div>""", unsafe_allow_html=True)

    chat_col1, chat_col2 = st.columns([5, 1], gap="small")
    with chat_col1:
        q_placeholder = "मुख्य निर्णय क्या थे?" if is_hindi else "What were the main decisions made?"
        user_input = st.text_input("Your question", placeholder=q_placeholder, label_visibility="collapsed")
    with chat_col2:
        send_btn = st.button("Send →", use_container_width=True)

    if send_btn and user_input.strip():
        with st.spinner("Thinking…"):
            answer = ask_question(r["rag_chain"], user_input.strip())
        st.session_state.chat_history.append({"role": "user",      "content": user_input.strip()})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

else:
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:5rem 2rem;text-align:center">
        <div style="font-size:3.5rem;margin-bottom:1rem">◎</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:700;color:var(--text);margin-bottom:0.5rem">
            Ready to Analyse
        </div>
        <div style="color:var(--text-muted);font-size:0.85rem;max-width:400px;line-height:1.7">
            Upload an audio or video file in the sidebar, choose English or Hindi, and hit <strong>Analyse</strong>.
        </div>
        <div style="margin-top:2rem;display:flex;gap:1rem;flex-wrap:wrap;justify-content:center">
            <span class="badge badge-purple">Transcription</span>
            <span class="badge badge-cyan">Summarisation</span>
            <span class="badge badge-green">RAG Chat</span>
            <span class="badge badge-orange">Hindi Support</span>
        </div>
    </div>""", unsafe_allow_html=True)
