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
    page_title="Video Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Syne:wght@600;700;800&family=JetBrains+Mono:wght@300;400&display=swap');

:root {
    --bg:          #08080f;
    --surface:     #0f0f1a;
    --surface-2:   #161625;
    --surface-3:   #1e1e30;
    --border:      #252538;
    --border-2:    #32324a;
    --accent:      #6d28d9;
    --accent-lite: #7c3aed;
    --glow:        #a78bfa;
    --cyan:        #22d3ee;
    --green:       #10b981;
    --orange:      #f97316;
    --text:        #ededf5;
    --text-2:      #a0a0c0;
    --text-3:      #606080;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background: var(--bg) !important; }

.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: radial-gradient(circle, rgba(109,40,217,0.055) 1px, transparent 1px);
    background-size: 28px 28px;
    pointer-events: none;
    z-index: 0;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: var(--surface-2) !important;
    border: 1px solid var(--border-2) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent-lite) !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.15) !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent-lite) 0%, #4c1d95 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.06em !important;
    padding: 0.6rem 1.4rem !important;
    text-transform: uppercase !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(109,40,217,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 22px rgba(109,40,217,0.5) !important;
}
.stButton > button[kind="secondary"] {
    background: var(--surface-2) !important;
    border: 1px solid var(--border-2) !important;
    box-shadow: none !important;
    color: var(--text-2) !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: var(--glow) !important;
    color: var(--text) !important;
    transform: none !important;
}

[data-testid="stFileUploader"] {
    background: var(--surface-2) !important;
    border: 1px dashed var(--border-2) !important;
    border-radius: 10px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent-lite) !important; }

label {
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
    color: var(--text-3) !important;
}

/* ── Wordmark ── */
.wordmark {
    display: flex; align-items: center; gap: 0.5rem;
    margin-bottom: 0.2rem;
}
.wordmark-icon {
    width: 28px; height: 28px;
    background: linear-gradient(135deg, var(--accent-lite), var(--cyan));
    border-radius: 7px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.85rem; flex-shrink: 0;
}
.wordmark-text {
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem; font-weight: 700;
    letter-spacing: -0.02em; color: var(--text);
}

.section-label {
    font-size: 0.65rem; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--text-3); margin-bottom: 0.5rem;
}

.pill {
    display: inline-flex; align-items: center; gap: 0.3rem;
    padding: 0.22rem 0.65rem; border-radius: 20px;
    font-size: 0.62rem; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
}
.pill-purple { background: rgba(124,58,237,0.15); color: var(--glow);   border: 1px solid rgba(124,58,237,0.25); }
.pill-cyan   { background: rgba(34,211,238,0.1);  color: var(--cyan);   border: 1px solid rgba(34,211,238,0.2); }
.pill-green  { background: rgba(16,185,129,0.1);  color: var(--green);  border: 1px solid rgba(16,185,129,0.2); }
.pill-orange { background: rgba(249,115,22,0.1);  color: var(--orange); border: 1px solid rgba(249,115,22,0.2); }

/* ── Hero ── */
.hero { padding: 3rem 0 2rem; }
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem; letter-spacing: 0.2em;
    text-transform: uppercase; color: var(--glow);
    margin-bottom: 1rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.hero-eyebrow::before {
    content: ''; display: inline-block;
    width: 20px; height: 1px; background: var(--glow);
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.5rem, 5vw, 4rem);
    font-weight: 800; line-height: 1.05;
    letter-spacing: -0.03em; margin: 0 0 1.25rem;
    background: linear-gradient(135deg, #fff 0%, var(--glow) 45%, var(--cyan) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-desc {
    font-size: 1rem; line-height: 1.75;
    color: var(--text-2); max-width: 560px;
    margin-bottom: 1.75rem; font-weight: 400;
}
.hero-desc strong { color: var(--text); font-weight: 500; }
.hero-pills { display: flex; gap: 0.6rem; flex-wrap: wrap; }
.hero-divider {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 2.5rem 0 2rem !important;
}

/* ── Feature cards (landing) ── */
.feat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.3rem;
    height: 100%;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
}
.feat-card:hover { border-color: var(--border-2); transform: translateY(-2px); }
.feat-card::after {
    content: '';
    position: absolute; top: 0; left: 0;
    width: 100%; height: 2px;
    background: linear-gradient(90deg, var(--accent-lite), var(--cyan));
}
.feat-icon {
    font-size: 1.4rem; margin-bottom: 0.75rem;
    display: block;
}
.feat-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem; font-weight: 700;
    color: var(--text); margin-bottom: 0.4rem;
}
.feat-desc {
    font-size: 0.8rem; line-height: 1.65;
    color: var(--text-3);
}

/* ── Pipeline tracker ── */
.tracker {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.5rem;
}
.tracker-head {
    font-family: 'Syne', sans-serif;
    font-size: 0.68rem; font-weight: 700;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--text-3); margin-bottom: 1.25rem;
}
.t-row {
    display: flex; align-items: flex-start;
    gap: 1rem; margin-bottom: 0.9rem; position: relative;
}
.t-row:not(:last-child)::after {
    content: ''; position: absolute;
    left: 11px; top: 26px;
    width: 2px; height: calc(100% - 2px);
    background: var(--border); border-radius: 1px;
}
.t-row.t-done::after   { background: rgba(16,185,129,0.3); }
.t-row.t-active::after { background: rgba(124,58,237,0.25); }

.t-dot {
    width: 24px; height: 24px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.6rem; font-weight: 700;
    flex-shrink: 0; z-index: 1;
    border: 2px solid var(--border); transition: all 0.3s;
}
.t-dot.pending { background: var(--surface-2); color: var(--text-3); }
.t-dot.active  { background: rgba(124,58,237,0.2); color: var(--glow); border-color: var(--accent-lite); animation: gpulse 1.4s ease infinite; }
.t-dot.done    { background: rgba(16,185,129,0.15); color: var(--green); border-color: var(--green); }

@keyframes gpulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(124,58,237,0); }
    50%      { box-shadow: 0 0 0 4px rgba(124,58,237,0.2); }
}
.t-label { font-size: 0.82rem; font-weight: 500; color: var(--text); }
.t-label.dim { color: var(--text-3); }
.t-detail { font-size: 0.7rem; margin-top: 0.1rem; font-family: 'JetBrains Mono', monospace; color: var(--text-3); }
.t-detail.active { color: var(--glow); }
.t-detail.done   { color: var(--green); }

/* ── Result cards ── */
.card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.4rem;
    margin-bottom: 1rem; position: relative;
    overflow: hidden; transition: border-color 0.2s, box-shadow 0.2s;
    height: 100%;
}
.card:hover { border-color: var(--border-2); box-shadow: 0 8px 30px rgba(0,0,0,0.3); }
.card-top-bar {
    position: absolute; top: 0; left: 0;
    width: 100%; height: 2px;
    background: linear-gradient(90deg, var(--accent-lite), var(--cyan));
}
.card-head { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.9rem; }
.card-ico {
    width: 28px; height: 28px; border-radius: 7px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.82rem; flex-shrink: 0;
}
.ico-p { background: rgba(124,58,237,0.15); }
.ico-c { background: rgba(34,211,238,0.1); }
.ico-g { background: rgba(16,185,129,0.1); }
.ico-o { background: rgba(249,115,22,0.1); }

.card-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-2);
}
.card-body { font-size: 0.85rem; line-height: 1.75; color: var(--text-2); }

/* ── Title banner ── */
.title-banner {
    background: linear-gradient(135deg, var(--surface) 0%, var(--surface-3) 100%);
    border: 1px solid var(--border-2); border-radius: 12px;
    padding: 1.2rem 1.5rem; margin-bottom: 1.5rem;
    display: flex; align-items: center; gap: 1rem;
    position: relative; overflow: hidden;
}
.title-banner::before {
    content: ''; position: absolute; top: 0; left: 0;
    width: 4px; height: 100%;
    background: linear-gradient(180deg, var(--accent-lite), var(--cyan));
}
.tb-label {
    font-size: 0.62rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--text-3); white-space: nowrap;
}
.tb-text {
    font-family: 'Syne', sans-serif;
    font-size: 1.15rem; font-weight: 700;
    letter-spacing: -0.02em; color: var(--text);
}

/* ── Transcript ── */
.transcript-box {
    background: var(--surface-2); border: 1px solid var(--border);
    border-radius: 8px; padding: 1.1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem; line-height: 1.85; color: var(--text-3);
    max-height: 280px; overflow-y: auto;
    white-space: pre-wrap; word-break: break-word;
}

/* ── Chat ── */
.chat-wrap {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.25rem;
    max-height: 400px; overflow-y: auto; margin-bottom: 0.75rem;
}
.chat-msg { margin-bottom: 1rem; display: flex; flex-direction: column; }
.chat-sender {
    font-size: 0.62rem; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.3rem;
}
.sender-you { color: var(--glow); }
.sender-bot { color: var(--cyan); }
.bubble {
    display: inline-block; padding: 0.65rem 1rem;
    border-radius: 10px; font-size: 0.84rem;
    line-height: 1.65; max-width: 88%;
}
.bubble-user {
    background: rgba(124,58,237,0.12);
    border: 1px solid rgba(124,58,237,0.22); align-self: flex-end;
}
.bubble-bot {
    background: rgba(34,211,238,0.07);
    border: 1px solid rgba(34,211,238,0.15); align-self: flex-start;
}
.chat-empty {
    padding: 2rem; text-align: center; color: var(--text-3);
    font-size: 0.82rem; background: var(--surface);
    border: 1px solid var(--border); border-radius: 12px; margin-bottom: 0.75rem;
}

/* ── Sidebar pipeline dots ── */
.sb-step {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.5rem 0.75rem;
    background: var(--surface-2); border: 1px solid var(--border);
    border-radius: 8px; margin-bottom: 0.3rem;
    font-size: 0.77rem; color: var(--text-2);
}
.sb-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.sb-dot.pending { background: var(--border-2); }
.sb-dot.active  { background: var(--glow); box-shadow: 0 0 6px var(--glow); animation: blink 1.3s infinite; }
.sb-dot.done    { background: var(--green); }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.35} }

hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1.75rem 0 !important; }
.stProgress > div > div > div { background: var(--accent-lite) !important; }
.stSpinner > div { border-top-color: var(--accent-lite) !important; }
[data-testid="stMarkdownContainer"] p { color: var(--text-2) !important; font-size: 0.875rem !important; }
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-2); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-lite); }
</style>
""", unsafe_allow_html=True)

# ─── Session state ────────────────────────────────────────────────────────────
for key, default in {
    "result": None,
    "chat_history": [],
    "pipeline_done": False,
    "pipeline_steps": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Pipeline config ──────────────────────────────────────────────────────────
STEPS = [
    ("audio",      "🔊", "Audio Processing"),
    ("transcript", "📝", "Transcription"),
    ("title",      "🏷️",  "Title Generation"),
    ("summary",    "📋", "Summarisation"),
    ("extract",    "🔍", "Extraction"),
    ("rag",        "🧠", "RAG Engine"),
]
STEP_DETAIL = {
    "audio":      {"active": "Converting & chunking audio...",  "done": "Audio chunks ready"},
    "transcript": {"active": "Transcribing with Whisper...",    "done": "Transcript complete"},
    "title":      {"active": "Generating title with LLM...",    "done": "Title generated"},
    "summary":    {"active": "Summarising transcript...",       "done": "Summary ready"},
    "extract":    {"active": "Extracting insights...",          "done": "Action items · Decisions · Questions done"},
    "rag":        {"active": "Building vector store...",        "done": "Chat enabled"},
}

def get_state(key):
    return st.session_state.pipeline_steps.get(key, "pending")

def render_tracker():
    html = '<div class="tracker"><div class="tracker-head">⚡ Pipeline Progress</div>'
    for key, icon, label in STEPS:
        s = get_state(key)
        d = STEP_DETAIL[key]
        if s == "done":
            dot, row, lbl_css, det, det_css, marker = "done", "t-done", "", d["done"], "done", "✓"
        elif s == "active":
            dot, row, lbl_css, det, det_css, marker = "active", "t-active", "", d["active"], "active", icon
        else:
            dot, row, lbl_css, det, det_css, marker = "pending", "", "dim", "Waiting...", "", "○"
        html += f"""
        <div class="t-row {row}">
            <div class="t-dot {dot}">{marker}</div>
            <div>
                <div class="t-label {lbl_css}">{label}</div>
                <div class="t-detail {det_css}">{det}</div>
            </div>
        </div>"""
    html += '</div>'
    return html

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="wordmark">
        <div class="wordmark-icon">◈</div>
        <div class="wordmark-text">Video Intelligence</div>
    </div>
    <div style="font-size:0.68rem;color:var(--text-3);margin-bottom:1.5rem;
                font-family:'JetBrains Mono',monospace;letter-spacing:0.05em;">
        audio · video · meetings
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Upload File</div>', unsafe_allow_html=True)
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
        <div style="margin-top:0.5rem;padding:0.45rem 0.75rem;
                    background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.2);
                    border-radius:7px;font-size:0.72rem;color:var(--green);
                    font-family:'JetBrains Mono',monospace;
                    overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">
            ✓ {uploaded_file.name}
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.1rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Language</div>', unsafe_allow_html=True)
    language = st.selectbox(
        "Language", ["english", "hindi"], index=0,
        label_visibility="collapsed",
        format_func=lambda x: "🇬🇧 English" if x == "english" else "🇮🇳 Hindi",
    )
    if language == "hindi":
        st.markdown("""
        <div style="margin-top:0.4rem;padding:0.45rem 0.75rem;
                    background:rgba(249,115,22,0.08);border:1px solid rgba(249,115,22,0.2);
                    border-radius:7px;font-size:0.7rem;color:var(--orange);">
            Sarvam AI · Hindi output
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.25rem'></div>", unsafe_allow_html=True)
    run_btn = st.button("⚡  Analyse", use_container_width=True)

    if st.session_state.pipeline_done:
        st.markdown("---")
        st.markdown('<div class="section-label">Status</div>', unsafe_allow_html=True)
        for key, icon, label in STEPS:
            s = get_state(key)
            st.markdown(f"""
            <div class="sb-step">
                <div class="sb-dot {s}"></div>
                <span>{icon} {label}</span>
            </div>""", unsafe_allow_html=True)

# ─── Main area ───────────────────────────────────────────────────────────────
if not st.session_state.result and not run_btn:
    # Landing / hero
    st.markdown("""
    <div class="hero">
        <div class="hero-eyebrow">AI-Powered Analysis</div>
        <h1 class="hero-title">Video Intelligence</h1>
        <p class="hero-desc">
            Drop any meeting, lecture, or video recording and get an instant
            <strong>transcript</strong>, <strong>summary</strong>, action items and
            key decisions — then <strong>chat with it</strong> like a document.
            Fully supported in <strong>English</strong> and <strong>Hindi</strong>.
        </p>
        <div class="hero-pills">
            <span class="pill pill-purple">◎ Transcription</span>
            <span class="pill pill-cyan">◈ Summarisation</span>
            <span class="pill pill-green">✦ RAG Chat</span>
            <span class="pill pill-orange">अ Hindi</span>
        </div>
    </div>
    <hr class="hero-divider">
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    feats = [
        ("🎙️", "Transcribe",  "Whisper-powered transcription for any audio or video file, with language hints for accuracy."),
        ("📋", "Summarise",   "Map-reduce summarisation that distils hours of recording into clean bullet points."),
        ("💬", "Chat",        "Ask anything about your recording. Answers come directly from the transcript via RAG."),
        ("🇮🇳", "Hindi Support","Full pipeline in Hindi via Sarvam AI — from transcription through to summary and extraction."),
    ]
    ico_css = ["ico-p", "ico-c", "ico-g", "ico-o"]
    for col, (ico, title, desc), icss in zip([c1, c2, c3, c4], feats, ico_css):
        with col:
            st.markdown(f"""
            <div class="feat-card">
                <span class="feat-icon">{ico}</span>
                <div class="feat-title">{title}</div>
                <div class="feat-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

# ─── Pipeline run ─────────────────────────────────────────────────────────────
if run_btn:
    if not source:
        st.error("Please upload an audio or video file to continue.")
    else:
        st.session_state.pipeline_done = False
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {}

        tracker_ph = st.empty()

        def update(key, state):
            st.session_state.pipeline_steps[key] = state
            tracker_ph.markdown(render_tracker(), unsafe_allow_html=True)

        try:
            tracker_ph.markdown(render_tracker(), unsafe_allow_html=True)

            update("audio", "active")
            chunks = process_input(source)
            update("audio", "done")

            update("transcript", "active")
            transcript = transcribe_all(chunks, language)
            update("transcript", "done")

            update("title", "active")
            title = generate_title(transcript)
            update("title", "done")

            update("summary", "active")
            summary = summarize(transcript, language)
            update("summary", "done")

            update("extract", "active")
            extracted    = extract_all(transcript, language)
            action_items = extracted["action_items"]
            decisions    = extracted["key_decisions"]
            questions    = extracted["open_questions"]
            update("extract", "done")

            update("rag", "active")
            rag_chain = build_rag_chain(transcript)
            update("rag", "done")

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
            time.sleep(0.8)
            st.rerun()

        except Exception as e:
            for k, *_ in STEPS:
                if st.session_state.pipeline_steps.get(k) == "active":
                    st.session_state.pipeline_steps[k] = "pending"
            tracker_ph.empty()
            st.error(f"❌ {e}")

# ─── Results ──────────────────────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result
    is_hindi = r.get("language") == "hindi"

    lang_tag = '<span class="pill pill-orange" style="margin-left:0.6rem;vertical-align:middle;font-size:0.58rem;">अ Hindi</span>' if is_hindi else ""
    st.markdown(f"""
    <div class="title-banner">
        <div>
            <div class="tb-label">Session Title</div>
            <div class="tb-text">{r['title']}{lang_tag}</div>
        </div>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2], gap="medium")
    with col1:
        lbl = "सारांश" if is_hindi else "Summary"
        st.markdown(f"""
        <div class="card">
            <div class="card-top-bar"></div>
            <div class="card-head">
                <div class="card-ico ico-p">📋</div>
                <div class="card-label">{lbl}</div>
            </div>
            <div class="card-body">{r['summary']}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        with st.expander("📝 Full Transcript", expanded=False):
            st.markdown(f'<div class="transcript-box">{r["transcript"]}</div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")
    three_cards = [
        (c1, "ico-g", "✅", "कार्य सूची" if is_hindi else "Action Items",    r["action_items"]),
        (c2, "ico-c", "🔑", "मुख्य निर्णय" if is_hindi else "Key Decisions", r["key_decisions"]),
        (c3, "ico-o", "❓", "खुले प्रश्न" if is_hindi else "Open Questions",  r["open_questions"]),
    ]
    for col, icss, ico, title, body in three_cards:
        with col:
            st.markdown(f"""
            <div class="card">
                <div class="card-top-bar"></div>
                <div class="card-head">
                    <div class="card-ico {icss}">{ico}</div>
                    <div class="card-label">{title}</div>
                </div>
                <div class="card-body">{body}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    chat_title = "बैठक से चैट करें" if is_hindi else "Chat with your Meeting"
    st.markdown(f"""
    <div style="font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:700;
                letter-spacing:-0.02em;margin-bottom:1rem;color:var(--text);">
        💬 {chat_title}
    </div>""", unsafe_allow_html=True)

    if st.session_state.chat_history:
        chat_html = '<div class="chat-wrap">'
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                chat_html += f"""
                <div class="chat-msg" style="align-items:flex-end">
                    <div class="chat-sender sender-you">You</div>
                    <div class="bubble bubble-user">{msg['content']}</div>
                </div>"""
            else:
                chat_html += f"""
                <div class="chat-msg" style="align-items:flex-start">
                    <div class="chat-sender sender-bot">Assistant</div>
                    <div class="bubble bubble-bot">{msg['content']}</div>
                </div>"""
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)
    else:
        ph = "बैठक के बारे में कुछ भी पूछें..." if is_hindi else "Ask anything about your meeting or recording..."
        st.markdown(f'<div class="chat-empty">💬 {ph}</div>', unsafe_allow_html=True)

    cc1, cc2 = st.columns([5, 1], gap="small")
    with cc1:
        qph = "मुख्य निर्णय क्या थे?" if is_hindi else "What were the main decisions made?"
        user_input = st.text_input("Q", placeholder=qph, label_visibility="collapsed")
    with cc2:
        send = st.button("Send →", use_container_width=True)

    if send and user_input.strip():
        with st.spinner("Thinking…"):
            answer = ask_question(r["rag_chain"], user_input.strip())
        st.session_state.chat_history.append({"role": "user",      "content": user_input.strip()})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()