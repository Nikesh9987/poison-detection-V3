try:
    import streamlit as st
except ImportError as e:
    raise ImportError(
        "streamlit is required to run this app. Install it with `pip install streamlit`."
    ) from e

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from code.toxin_matcher import ToxinMatcher


# ==========================
# Load model
# ==========================

BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "toxin_matcher.pkl"
)

matcher = ToxinMatcher.load(MODEL_PATH)


# ==========================
# Load toxin reference data (severity + confirmatory tests)
# Single source of truth: data/toxin_master.csv + data/confirmatory_tests.csv
# Update those CSVs (and re-run train_similarity_model.py) to add/change
# toxins -- no code changes needed here.
# ==========================

@st.cache_data
def load_toxic_db():
    master = pd.read_csv(os.path.join(BASE_DIR, "data", "toxin_master.csv"))
    conf = pd.read_csv(os.path.join(BASE_DIR, "data", "confirmatory_tests.csv"))
    merged = master.merge(conf, on="toxin_id", suffixes=("", "_conf"))

    db = {}
    for _, row in merged.iterrows():
        name = str(row["toxin_name"]).strip()
        db[name] = {
            "description": str(row["description"]),
            "severity": row["severity"],
            "confirmatory": str(row["confirmatory_test"]),
            "sample": str(row["priority_sample"]),
            "screening": str(row["screening_test"]),
            "critical_note": str(row["critical_note"]),
        }
    return db


def confidence_label(score: float) -> str:
    """Bucket the matcher's cosine-similarity score into a display label."""
    if score >= 0.30:
        return "HIGH"
    if score >= 0.12:
        return "MODERATE"
    return "LOW"


# ==========================
# Page Config
# ==========================

st.set_page_config(
    page_title="VishAI · Forensic Toxicology DSS",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==========================
# CSS Design System
# ==========================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap');

/* ─── Root Tokens ─────────────────────────────────── */
:root {
  --bg-base:       #080c14;
  --bg-surface:    #0d1525;
  --bg-elevated:   #111d33;
  --bg-card:       #0f1a2e;
  --border:        rgba(99, 160, 255, 0.10);
  --border-bright: rgba(99, 160, 255, 0.22);
  --accent-blue:   #4d8bff;
  --accent-cyan:   #00d4e8;
  --accent-teal:   #00c9a7;
  --red-critical:  #ff4d6d;
  --amber-severe:  #ffb830;
  --green-mod:     #22d3a5;
  --text-primary:  #e8f0ff;
  --text-secondary:#8da4cc;
  --text-muted:    #4a607e;
  --font-display:  'Syne', sans-serif;
  --font-body:     'DM Sans', sans-serif;
  --font-mono:     'DM Mono', monospace;
  --radius-sm:     8px;
  --radius-md:     14px;
  --radius-lg:     20px;
  --shadow-card:   0 4px 32px rgba(0,0,0,0.45), 0 1px 0 rgba(99,160,255,0.06) inset;
  --shadow-glow:   0 0 40px rgba(77,139,255,0.12);
}

/* ─── Global Reset ────────────────────────────────── */
html, body, [class*="css"] {
  font-family: var(--font-body) !important;
  color: var(--text-primary) !important;
}

.stApp {
  background: var(--bg-base) !important;
  background-image:
    radial-gradient(ellipse 80% 50% at 50% -10%, rgba(77,139,255,0.07) 0%, transparent 70%),
    radial-gradient(ellipse 40% 30% at 85% 20%, rgba(0,212,232,0.04) 0%, transparent 60%) !important;
}

/* hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
  padding: 0 2.5rem 4rem !important;
  max-width: 1280px !important;
}

/* ─── Typography ──────────────────────────────────── */
h1, h2, h3 {
  font-family: var(--font-display) !important;
  letter-spacing: -0.02em !important;
}

/* ─── Scrollbar ───────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 3px; }

/* ─── Hero ────────────────────────────────────────── */
.hero-wrap {
  padding: 3.5rem 0 2.5rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 2.5rem;
}
.hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: rgba(77,139,255,0.10);
  border: 1px solid rgba(77,139,255,0.25);
  color: #7eb8ff;
  font-family: var(--font-mono);
  font-size: 0.72rem;
  font-weight: 500;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  padding: 5px 14px;
  border-radius: 100px;
  margin-bottom: 1.4rem;
}
.hero-badge::before {
  content: '';
  width: 6px; height: 6px;
  background: var(--accent-cyan);
  border-radius: 50%;
  box-shadow: 0 0 8px var(--accent-cyan);
  animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
  0%,100% { opacity: 1; }
  50%      { opacity: 0.35; }
}
.hero-title {
  font-family: var(--font-display) !important;
  font-size: 3.2rem !important;
  font-weight: 800 !important;
  letter-spacing: -0.035em !important;
  line-height: 1.08 !important;
  background: linear-gradient(135deg, #e8f0ff 0%, #7eb8ff 55%, #00d4e8 100%);
  -webkit-background-clip: text !important;
  -webkit-text-fill-color: transparent !important;
  background-clip: text !important;
  margin: 0 0 0.75rem !important;
}
.hero-sub {
  font-size: 1.05rem;
  color: var(--text-secondary);
  max-width: 560px;
  line-height: 1.65;
  margin: 0;
}
.hero-disclaimer {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 1.6rem;
  background: rgba(255,184,48,0.06);
  border: 1px solid rgba(255,184,48,0.18);
  border-radius: var(--radius-sm);
  padding: 10px 16px;
  font-size: 0.82rem;
  color: #c99a40;
  max-width: 640px;
  font-family: var(--font-mono);
}

/* ─── Section Labels ──────────────────────────────── */
.section-label {
  font-family: var(--font-mono);
  font-size: 0.7rem;
  font-weight: 500;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--text-muted);
  margin-bottom: 1.2rem;
  display: flex;
  align-items: center;
  gap: 10px;
}
.section-label::after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border);
}

/* ─── Input Cards ─────────────────────────────────── */
.input-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 1.6rem 1.8rem;
  box-shadow: var(--shadow-card);
  transition: border-color 0.25s ease;
  height: 100%;
}
.input-card:hover {
  border-color: var(--border-bright);
}
.card-icon-title {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 1.2rem;
}
.card-icon {
  width: 34px; height: 34px;
  border-radius: 9px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1rem;
  flex-shrink: 0;
}
.icon-neuro  { background: rgba(255,77,109,0.12); }
.icon-resp   { background: rgba(0,212,232,0.10); }
.icon-system { background: rgba(0,201,167,0.10); }
.card-title {
  font-family: var(--font-display) !important;
  font-size: 0.92rem !important;
  font-weight: 600 !important;
  color: var(--text-primary) !important;
  margin: 0 !important;
  letter-spacing: -0.01em !important;
}

/* ─── Streamlit widget overrides ─────────────────── */
div[data-testid="stMultiSelect"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stCheckbox"] label {
  font-family: var(--font-body) !important;
  font-size: 0.82rem !important;
  font-weight: 500 !important;
  color: var(--text-secondary) !important;
  letter-spacing: 0.02em !important;
  text-transform: uppercase !important;
}
div[data-testid="stMultiSelect"] > div > div,
div[data-testid="stSelectbox"] > div > div {
  background: rgba(13,21,37,0.8) !important;
  border: 1px solid var(--border-bright) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-primary) !important;
}
div[data-testid="stMultiSelect"] > div > div:focus-within,
div[data-testid="stSelectbox"] > div > div:focus-within {
  border-color: var(--accent-blue) !important;
  box-shadow: 0 0 0 3px rgba(77,139,255,0.12) !important;
}

/* ─── Predict Button ──────────────────────────────── */
div[data-testid="stButton"] > button {
  font-family: var(--font-display) !important;
  font-weight: 700 !important;
  font-size: 0.95rem !important;
  letter-spacing: 0.01em !important;
  background: linear-gradient(135deg, #2a5ce8 0%, #1a8fcf 100%) !important;
  color: #fff !important;
  border: none !important;
  border-radius: var(--radius-md) !important;
  padding: 0.75rem 2.4rem !important;
  box-shadow: 0 4px 24px rgba(42,92,232,0.35), 0 1px 0 rgba(255,255,255,0.08) inset !important;
  transition: all 0.2s ease !important;
  height: auto !important;
  width: 100% !important;
}
div[data-testid="stButton"] > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 32px rgba(42,92,232,0.48), 0 1px 0 rgba(255,255,255,0.12) inset !important;
}
div[data-testid="stButton"] > button:active {
  transform: translateY(0) !important;
}

/* ─── Results Section ─────────────────────────────── */
.result-banner {
  border-radius: var(--radius-lg);
  padding: 1.8rem 2.2rem;
  margin: 2rem 0 1.5rem;
  display: flex;
  align-items: center;
  gap: 1.4rem;
  border: 1px solid;
  animation: fadeSlideIn 0.4s ease;
}
@keyframes fadeSlideIn {
  from { opacity: 0; transform: translateY(12px); }
  to   { opacity: 1; transform: translateY(0); }
}
.result-banner.critical {
  background: rgba(255,77,109,0.07);
  border-color: rgba(255,77,109,0.30);
}
.result-banner.severe {
  background: rgba(255,184,48,0.07);
  border-color: rgba(255,184,48,0.28);
}
.result-banner.moderate {
  background: rgba(34,211,165,0.07);
  border-color: rgba(34,211,165,0.25);
}
.result-banner.unknown {
  background: rgba(77,139,255,0.07);
  border-color: rgba(77,139,255,0.25);
}
# .result-substance {
#   font-family: var(--font-display);
#   font-size: 2rem;
#   font-weight: 800;
#   letter-spacing: -0.03em;
#   line-height: 1.1;
# }
# .result-subtitle {
#   font-size: 0.8rem;
#   font-family: var(--font-mono);
#   text-transform: uppercase;
#   letter-spacing: 0.1em;
#   margin-top: 3px;
# }

.result-substance {
  font-family: var(--font-display);
  font-size: 1.45rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  line-height: 1.3;
  word-break: break-word;
}
.result-subtitle {
  font-size: 0.66rem;
  font-family: var(--font-mono);
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.16em;
  color: var(--text-muted);
  margin-bottom: 8px;
}
.severity-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 14px;
  border-radius: 100px;
  font-family: var(--font-mono);
  font-size: 0.72rem;
  font-weight: 500;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  white-space: nowrap;
}
.pill-critical { background: rgba(255,77,109,0.15); color: #ff7090; border: 1px solid rgba(255,77,109,0.3); }
.pill-severe   { background: rgba(255,184,48,0.12);  color: #ffc94d; border: 1px solid rgba(255,184,48,0.28); }
.pill-moderate { background: rgba(34,211,165,0.10); color: #2ee8b9; border: 1px solid rgba(34,211,165,0.25); }
.pill-unknown  { background: rgba(77,139,255,0.10); color: #7eb8ff; border: 1px solid rgba(77,139,255,0.25); }

/* ─── Metric Cards ────────────────────────────────── */
.metric-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 1.4rem 1.6rem;
  text-align: center;
  animation: fadeSlideIn 0.5s ease;
}
.metric-label {
  font-family: var(--font-mono);
  font-size: 0.68rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text-muted);
  margin-bottom: 0.5rem;
}
.metric-value {
  font-family: var(--font-display);
  font-size: 1.35rem;
  font-weight: 700;
  letter-spacing: -0.02em;
}
.metric-critical { color: var(--red-critical); }
.metric-severe   { color: var(--amber-severe); }
.metric-moderate { color: var(--green-mod); }
.metric-neutral  { color: var(--accent-cyan); }

/* ─── Detail Cards ────────────────────────────────── */
.detail-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 1.4rem 1.6rem;
  margin-bottom: 1rem;
  animation: fadeSlideIn 0.55s ease;
}
.detail-card-header {
  font-family: var(--font-mono);
  font-size: 0.68rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--text-muted);
  margin-bottom: 0.6rem;
  display: flex;
  align-items: center;
  gap: 8px;
}
.detail-card-body {
  font-size: 0.96rem;
  color: var(--text-primary);
  font-weight: 500;
  line-height: 1.6;
}

/* ─── Symptom Tags ────────────────────────────────── */
.tag-wrap { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
.tag {
  background: rgba(77,139,255,0.09);
  border: 1px solid rgba(77,139,255,0.18);
  color: #7eb8ff;
  border-radius: 6px;
  padding: 3px 10px;
  font-family: var(--font-mono);
  font-size: 0.75rem;
}
.tag-none {
  color: var(--text-muted);
  font-family: var(--font-mono);
  font-size: 0.78rem;
  font-style: italic;
}

/* ─── Case Summary ────────────────────────────────── */
.summary-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 2rem 2.2rem;
  margin-top: 1rem;
  animation: fadeSlideIn 0.6s ease;
}
.summary-row {
  display: grid;
  grid-template-columns: 160px 1fr;
  align-items: start;
  padding: 0.85rem 0;
  border-bottom: 1px solid var(--border);
  gap: 1rem;
}
.summary-row:last-child { border-bottom: none; }
.summary-key {
  font-family: var(--font-mono);
  font-size: 0.72rem;
  letter-spacing: 0.10em;
  text-transform: uppercase;
  color: var(--text-muted);
  padding-top: 3px;
}
.summary-val {
  font-size: 0.92rem;
  color: var(--text-primary);
  line-height: 1.5;
}

/* ─── Footer Disclaimer ───────────────────────────── */
.footer-disclaimer {
  margin-top: 3rem;
  padding: 1.4rem 1.8rem;
  background: rgba(255,184,48,0.05);
  border: 1px solid rgba(255,184,48,0.15);
  border-radius: var(--radius-md);
  display: flex;
  gap: 14px;
  align-items: flex-start;
  animation: fadeSlideIn 0.65s ease;
}
.footer-disclaimer-icon {
  font-size: 1.2rem;
  flex-shrink: 0;
  margin-top: 1px;
}
.footer-disclaimer-text {
  font-size: 0.83rem;
  color: #9e7a2e;
  line-height: 1.65;
}
.footer-disclaimer-text strong {
  color: #c99a40;
  font-weight: 600;
}

/* ─── Dividers ────────────────────────────────────── */
.vis-divider {
  height: 1px;
  background: var(--border);
  margin: 2rem 0;
}

/* ─── Streamlit spinner override ─────────────────── */
div[data-testid="stSpinner"] > div {
  border-top-color: var(--accent-blue) !important;
}

/* ─── Misc overrides ──────────────────────────────── */
div[data-testid="stVerticalBlock"] { gap: 0 !important; }
.stAlert { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ==========================
# Helper: render tag chips
# ==========================

def render_tags(items):
    if not items:
        return '<span class="tag-none">None selected</span>'
    return "".join(f'<span class="tag">{i}</span>' for i in items)


def severity_class(sev):
    s = str(sev).upper()
    if "CRITICAL" in s: return "critical"
    if "SEVERE"   in s: return "severe"
    if "MODERATE" in s: return "moderate"
    return "unknown"


def metric_color_class(label, value):
    v = str(value).upper()
    if "CRITICAL" in v: return "metric-critical"
    if "SEVERE"   in v: return "metric-severe"
    if "MODERATE" in v: return "metric-moderate"
    return "metric-neutral"


def pill_class(value):
    v = str(value).upper()
    if "CRITICAL" in v: return "pill-critical"
    if "SEVERE"   in v: return "pill-severe"
    if "MODERATE" in v: return "pill-moderate"
    return "pill-unknown"


# ==========================
# Hero
# ==========================

st.markdown("""
<div class="hero-wrap">
  <div class="hero-badge">⬡ VishAI · Forensic Decision Support.</div>
  <h1 class="hero-title">Decision Support System - VishAI</h1>
  <p class="hero-sub">AI assisted forensic toxicology decision support tool for probable poison identification.</p>
</div>
""", unsafe_allow_html=True)


# ==========================
# INPUT SECTION
# ==========================

st.markdown('<p class="section-label">01 — Symptom Profile</p>', unsafe_allow_html=True)

col_a, col_b, col_c = st.columns([1, 1, 1], gap="medium")

# ── Card A: Neurological & Cardiac ──────────────────
with col_a:
    st.markdown("""
    <div class="input-card">
      <div class="card-icon-title">
        <div class="card-icon icon-neuro">🧠</div>
        <p class="card-title">Neurological &amp; Cardiac</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    cns = st.multiselect(
        "CNS Symptoms",
        ["confusion", "coma", "delirium", "anxiety", "seizures", "headache",
         "restlessness", "dizziness", "drowsiness", "ataxia", "miosis (pinpoint pupils)",
         "mydriasis (dilated pupils)", "hallucinations", "ptosis", "dysarthria",
         "nystagmus", "tremors", "agitation"],
        key="cns"
    )

    cvs = st.multiselect(
        "CVS Symptoms",
        ["tachycardia", "bradycardia", "hypotension", "cardiac arrest",
         "arrhythmia", "hypertension", "cardiovascular collapse",
         "ventricular fibrillation", "shock"],
        key="cvs"
    )

# ── Card B: Respiratory & GI ────────────────────────
with col_b:
    st.markdown("""
    <div class="input-card">
      <div class="card-icon-title">
        <div class="card-icon icon-resp">🫁</div>
        <p class="card-title">Respiratory &amp; Gastrointestinal</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    resp = st.multiselect(
        "Respiratory Symptoms",
        ["dyspnoea", "tachypnoea", "respiratory failure", "pulmonary oedema",
         "apnoea", "cyanosis", "respiratory arrest", "respiratory depression",
         "bronchospasm", "aspiration pneumonia", "chest tightness"],
        key="resp"
    )

    gi = st.multiselect(
        "GI Symptoms",
        ["vomiting", "nausea", "abdominal pain", "diarrhoea", "hypersalivation",
         "haematemesis", "jaundice", "dysphagia", "metallic taste",
         "burning in mouth/throat", "epigastric burning", "constipation"],
        key="gi"
    )

# ── Card C: Systemic & Clinical Context ─────────────
with col_c:
    st.markdown("""
    <div class="input-card">
      <div class="card-icon-title">
        <div class="card-icon icon-system">⚕️</div>
        <p class="card-title">Systemic &amp; Clinical Context</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    other = st.multiselect(
        "Other / Systemic Symptoms",
        ["fever", "cyanosis", "renal failure", "burning sensation", "diaphoresis",
         "multiorgan failure", "hypothermia", "hyperthermia", "rhabdomyolysis",
         "weakness", "flaccid paralysis", "urinary retention",
         "yellow vision (xanthopsia)", "dry mouth", "flushing", "bleeding / ecchymosis",
         "oliguria / anuria"],
        key="other"
    )

    odour = st.selectbox(
        "Odour",
        ["Unknown", "Petroleum / Kerosene", "Garlic / Fishy", "Sweet / Alcoholic",
         "Bitter Almond", "Carbolic / Antiseptic", "None"],
        key="odour"
    )

st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)
st.markdown('<p class="section-label">02 — Exposure &amp; Sample</p>', unsafe_allow_html=True)

col_d, col_e, col_f = st.columns([1, 1, 1], gap="medium")

with col_d:
    route = st.selectbox(
        "Route of Exposure",
        ["Ingestion", "Inhalation", "Injection", "Dermal", "Unknown"],
        key="route"
    )

with col_e:
    timeline = st.selectbox(
        "Symptom Timeline",
        ["Rapid", "Acute", "Subacute", "Delayed", "Chronic"],
        key="timeline"
    )

with col_f:
    sample = st.multiselect(
        "Available Sample",
        ["Blood", "Urine", "Gastric Lavage", "Hair", "Nails", "Viscera"],
        key="sample"
    )


# ==========================
# PREDICT BUTTON
# ==========================

st.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)

btn_col, _ = st.columns([1, 2])
with btn_col:
    predict_clicked = st.button("⬡  Predict Toxic Substance", key="predict_btn")


# ==========================
# PREDICT + RESULTS
# ==========================

if predict_clicked:

    with st.spinner("Analysing symptom profile…"):

        # ── Backend logic — untouched ────────────────
        symptoms = (
            " ".join(cns)
            + " "
            + " ".join(cvs)
            + " "
            + " ".join(resp)
            + " "
            + " ".join(gi)
            + " "
            + " ".join(other)
            + " "
            + odour
        )

        best_label, match_confidence, ranked_matches = matcher.predict(symptoms)

        toxic_db = load_toxic_db()
        matched = toxic_db.get(best_label) if best_label else None

        if best_label and matched:
            prediction = f"{best_label} ({matched['description']})"
            matched = dict(matched)  # copy so we can add the dynamic confidence
            matched["confidence"] = confidence_label(match_confidence)
        else:
            prediction = "No confident match \u2014 symptoms too ambiguous / not enough data"


    # ── Results Header ───────────────────────────────
    st.markdown('<p class="section-label" style="margin-top:2.2rem">03 — Analysis Result</p>', unsafe_allow_html=True)

    sev_val    = matched["severity"]   if matched else "Unknown"
    conf_val   = matched["confidence"] if matched else "–"
    sev_cls    = severity_class(sev_val)
    pill_cls   = pill_class(sev_val)

    # st.markdown(f"""
    # <div class="result-banner {sev_cls}">
    #   <div style="flex:1">
    #     <div class="result-subtitle" style="color:var(--text-muted); margin-bottom:6px">Identified Toxic Substance</div>
    #     <div class="result-substance" style="color:var(--text-primary)">{prediction}</div>
    #   </div>
    #   <div class="severity-pill {pill_cls}">
    #     {'◉' if sev_cls == 'critical' else '◎'}&nbsp;{sev_val}
    #   </div>
    # </div>
    # """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-banner {sev_cls}">
      <div style="flex:1; min-width:0">
        <div class="result-subtitle">Identified Toxic Substance</div>
        <div class="result-substance" style="color:var(--text-primary)">{prediction}</div>
        <div style="margin-top:10px">
          <span class="severity-pill {pill_cls}">
            {'◉' if sev_cls == 'critical' else '◎'}&nbsp;{sev_val}
          </span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


    # ── Metric Cards ────────────────────────────────
    if matched:
        mc1, mc2, mc3 = st.columns(3, gap="medium")

        conf_color = "metric-neutral" if matched["confidence"] == "HIGH" else "metric-severe"

        with mc1:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">AI Confidence</div>
              <div class="metric-value {conf_color}">{matched["confidence"]}</div>
            </div>
            """, unsafe_allow_html=True)

        with mc2:
            sev_color = metric_color_class("severity", matched["severity"])
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">Severity Level</div>
              <div class="metric-value {sev_color}">{matched["severity"]}</div>
            </div>
            """, unsafe_allow_html=True)

        with mc3:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">Symptom Timeline</div>
              <div class="metric-value metric-neutral">{timeline}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)

        # ── Recommendation Cards ─────────────────────
        st.markdown('<p class="section-label">04 — Forensic Recommendations</p>', unsafe_allow_html=True)

        rec1, rec2 = st.columns(2, gap="medium")

        with rec1:
            st.markdown(f"""
            <div class="detail-card">
              <div class="detail-card-header">🔬&nbsp; Recommended Confirmatory Test</div>
              <div class="detail-card-body">{matched["confirmatory"].replace(chr(10), "<br/>")}</div>
            </div>
            """, unsafe_allow_html=True)

        with rec2:
            st.markdown(f"""
            <div class="detail-card">
              <div class="detail-card-header">🧪&nbsp; Optimal Biological Sample</div>
              <div class="detail-card-body">{matched["sample"]}</div>
            </div>
            """, unsafe_allow_html=True)


    # ── Case Summary ─────────────────────────────────
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">05 — Case Summary Report</p>', unsafe_allow_html=True)

    cns_tags  = render_tags(cns)
    cvs_tags  = render_tags(cvs)
    resp_tags = render_tags(resp)
    gi_tags   = render_tags(gi)
    oth_tags  = render_tags(other)
    smp_tags  = render_tags(sample)

    st.markdown(f"""
    <div class="summary-card">
      <div class="summary-row">
        <span class="summary-key">Substance</span>
        <span class="summary-val" style="font-weight:700;color:#7eb8ff">{prediction}</span>
      </div>
      <div class="summary-row">
        <span class="summary-key">Route</span>
        <span class="summary-val">{route}</span>
      </div>
      <div class="summary-row">
        <span class="summary-key">Timeline</span>
        <span class="summary-val">{timeline}</span>
      </div>
      <div class="summary-row">
        <span class="summary-key">Odour</span>
        <span class="summary-val">{odour}</span>
      </div>
      <div class="summary-row">
        <span class="summary-key">CNS</span>
        <span class="summary-val"><div class="tag-wrap">{cns_tags}</div></span>
      </div>
      <div class="summary-row">
        <span class="summary-key">CVS</span>
        <span class="summary-val"><div class="tag-wrap">{cvs_tags}</div></span>
      </div>
      <div class="summary-row">
        <span class="summary-key">Respiratory</span>
        <span class="summary-val"><div class="tag-wrap">{resp_tags}</div></span>
      </div>
      <div class="summary-row">
        <span class="summary-key">Gastrointestinal</span>
        <span class="summary-val"><div class="tag-wrap">{gi_tags}</div></span>
      </div>
      <div class="summary-row">
        <span class="summary-key">Systemic</span>
        <span class="summary-val"><div class="tag-wrap">{oth_tags}</div></span>
      </div>
      <div class="summary-row">
        <span class="summary-key">Samples Available</span>
        <span class="summary-val"><div class="tag-wrap">{smp_tags}</div></span>
      </div>
    </div>
    """, unsafe_allow_html=True)


    # ── Footer Disclaimer ────────────────────────────
    st.markdown("""
    <div class="footer-disclaimer">
      <div class="footer-disclaimer-icon"></div>
      <div class="footer-disclaimer-text">
        <strong>Medico-legal Notice:</strong> This is a preliminary forensic decision-support output.
        Laboratory confirmation is mandatory before any medico-legal interpretation.
        <br/><br/>Powered by <strong>VishAI Decision Support System</strong> · Forensic Toxicology Intelligence
      </div>
    </div>
    """, unsafe_allow_html=True)