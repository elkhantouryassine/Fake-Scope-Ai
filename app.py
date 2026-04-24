"""
FakeScope — Interface de détection de deepfakes
Design Terminal Monochrome + Lime — Space Mono + Space Grotesk
"""
import os, sys, warnings, json, threading
import numpy as np
import cv2
from PIL import Image
import gradio as gr
import tensorflow as tf
from tensorflow import keras
from mtcnn import MTCNN

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ── Chemins ───────────────────────────────────────────────
BASE_DIR   = r"C:\Users\Lenovo\Desktop\deepshield"
MODEL_PATH = os.path.join(BASE_DIR, "models", "deepshield_best.keras")
CFG_PATH   = os.path.join(BASE_DIR, "data", "processed", "config.json")

sys.path.append(BASE_DIR)
from gradcam_utils import get_gradcam_heatmap, overlay_gradcam

with open(CFG_PATH) as f:
    cfg = json.load(f)
IMG_SIZE  = cfg["IMG_SIZE"]
THRESHOLD = 0.5

print("⏳ Chargement du modèle...")
model    = keras.models.load_model(MODEL_PATH)
detector = MTCNN()
model.predict(np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype='float32'), verbose=0)
print("✅ Modèle prêt !")


# ══════════════════════════════════════════════════════════
# CSS — Terminal Monochrome + Lime
# ══════════════════════════════════════════════════════════
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');

/* ═══ ROOT ═══ */
:root {
    --ink:   #0d0d0d;
    --ink2:  #1a1a1a;
    --ink3:  #2a2a2a;
    --wire:  #3a3a3a;
    --mist:  #888888;
    --fog:   #aaaaaa;
    --paper: #f5f4f0;
    --lime:  #c8ff00;
    --lime2: #b4e800;
    --red:   #ff4c4c;
    --mono: 'Space Mono', monospace;
    --sans: 'Space Grotesk', sans-serif;
}

/* ═══ BASE ═══ */
html { scroll-behavior: smooth; }

body,
.gradio-container,
#root {
    background: var(--ink) !important;
    font-family: var(--sans) !important;
    color: var(--paper) !important;
    min-height: 100vh;
}

/* Grid de fond */
.gradio-container::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(var(--ink2) 1px, transparent 1px),
        linear-gradient(90deg, var(--ink2) 1px, transparent 1px);
    background-size: 40px 40px;
    opacity: 0.5;
    pointer-events: none;
    z-index: 0;
}
.gradio-container > * { position: relative; z-index: 1; }

/* ═══ SPLASH ═══ */
#splash-overlay {
    position: fixed;
    inset: 0;
    z-index: 99999;
    background: var(--ink);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px;
    transition: opacity 0.55s ease, transform 0.55s ease;
}
#splash-overlay.hidden {
    opacity: 0;
    pointer-events: none;
    transform: scale(0.97);
}

/* grid overlay on splash */
#splash-overlay::before {
    content: '';
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(var(--ink2) 1px, transparent 1px),
        linear-gradient(90deg, var(--ink2) 1px, transparent 1px);
    background-size: 40px 40px;
    opacity: 0.5;
    pointer-events: none;
    z-index: 0;
}
#splash-overlay > * { position: relative; z-index: 1; }

.splash-corner {
    position: absolute;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--wire);
    letter-spacing: 0.08em;
    z-index: 2;
}
.splash-corner.tl { top: 20px; left: 20px; }
.splash-corner.tr { top: 20px; right: 20px; }
.splash-corner.bl { bottom: 20px; left: 20px; }
.splash-corner.br { bottom: 20px; right: 20px; }

#splash-eyebrow {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.15em;
    color: var(--mist);
    text-transform: uppercase;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}
#splash-eyebrow::before,
#splash-eyebrow::after {
    content: '';
    width: 32px;
    height: 1px;
    background: var(--wire);
    display: inline-block;
}

#splash-logo {
    font-family: var(--mono);
    font-size: clamp(52px, 8vw, 80px);
    font-weight: 700;
    color: var(--paper);
    line-height: 1;
    letter-spacing: -4px;
    margin-bottom: 4px;
    text-align: center;
}
#splash-logo span { color: var(--lime); }

#splash-sub {
    font-size: 14px;
    color: var(--mist);
    font-weight: 300;
    letter-spacing: 0.04em;
    text-align: center;
    line-height: 1.8;
    max-width: 340px;
    margin: 14px 0 32px;
}

#splash-tags {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    justify-content: center;
    margin-bottom: 40px;
}
.splash-tag {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.1em;
    color: var(--mist);
    border: 1px solid var(--wire);
    padding: 5px 10px;
    text-transform: uppercase;
}

#splash-btn {
    background: var(--lime);
    color: var(--ink);
    border: none;
    font-family: var(--mono);
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 16px 44px;
    cursor: pointer;
    transition: background 0.15s;
    position: relative;
}
#splash-btn:hover { background: var(--lime2); }
#splash-btn::after { content: ' →'; }

/* ═══ HERO ═══ */
#fakescope-hero {
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: center;
    gap: 40px;
    padding: 48px 32px 36px;
    border-bottom: 1px solid var(--wire);
    margin-bottom: 28px;
}

.hero-eyebrow {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.15em;
    color: var(--mist);
    text-transform: uppercase;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.hero-eyebrow::before {
    content: '';
    width: 20px;
    height: 1px;
    background: var(--lime);
    display: inline-block;
    flex-shrink: 0;
}

.hero-title {
    font-family: var(--mono);
    font-size: clamp(34px, 4.5vw, 54px);
    font-weight: 700;
    letter-spacing: -3px;
    line-height: 1;
    color: var(--paper);
    margin-bottom: 14px;
}
.hero-title span { color: var(--lime); }

.hero-sub {
    font-size: 14px;
    font-weight: 300;
    color: var(--mist);
    letter-spacing: 0.02em;
    line-height: 1.8;
    margin-bottom: 22px;
}

.stat-row {
    display: flex;
    width: fit-content;
    border: 1px solid var(--wire);
}
.stat-pill {
    padding: 10px 18px;
    border-right: 1px solid var(--wire);
    display: flex;
    flex-direction: column;
    gap: 3px;
}
.stat-pill:last-child { border-right: none; }
.stat-num {
    font-family: var(--mono);
    font-size: 14px;
    font-weight: 700;
    color: var(--lime);
    line-height: 1;
}
.stat-lbl {
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--wire);
}

/* Hero terminal widget */
#hero-terminal {
    width: 210px;
    background: var(--ink2);
    border: 1px solid var(--wire);
    padding: 14px 16px;
    font-family: var(--mono);
    font-size: 10.5px;
    line-height: 1.9;
    color: var(--mist);
    position: relative;
    overflow: hidden;
    flex-shrink: 0;
}
#hero-terminal::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--lime);
}
#hero-terminal .t-lime { color: var(--lime); }
#hero-terminal .t-red  { color: var(--red); }

.cursor {
    display: inline-block;
    width: 7px;
    height: 12px;
    background: var(--lime);
    animation: cur-blink 1s step-end infinite;
    vertical-align: text-bottom;
    margin-left: 1px;
}
@keyframes cur-blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0; }
}

/* ═══ TABS ═══ */
.gradio-container .tab-nav,
.gradio-container [role="tablist"] {
    border-bottom: 1px solid var(--wire) !important;
    background: transparent !important;
    padding: 0 32px !important;
    gap: 0 !important;
    border-radius: 0 !important;
    box-shadow: none !important;
}

button[role="tab"] {
    font-family: var(--mono) !important;
    font-size: 11px !important;
    font-weight: 400 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--mist) !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 14px 20px !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    border-radius: 0 !important;
    margin-bottom: -1px !important;
}
button[role="tab"]:hover {
    color: var(--paper) !important;
    background: transparent !important;
}
button[role="tab"][aria-selected="true"],
button.selected[role="tab"] {
    color: var(--lime) !important;
    background: transparent !important;
    border-bottom-color: var(--lime) !important;
    box-shadow: none !important;
}

.gradio-container .tabitem {
    background: transparent !important;
    border: none !important;
    padding: 24px 32px !important;
}

/* ═══ SECTION LABELS ═══ */
.section-label {
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 400;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--mist);
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--wire);
}

.gradio-container label,
.gradio-container .gr-form > label {
    font-family: var(--mono) !important;
    font-size: 10px !important;
    font-weight: 400 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--mist) !important;
}

/* ═══ BUTTONS ═══ */
#btn-analyze, #btn-start, #btn-video {
    background: var(--lime) !important;
    color: var(--ink) !important;
    border: none !important;
    border-radius: 0 !important;
    font-family: var(--mono) !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 14px 24px !important;
    min-height: 48px !important;
    height: auto !important;
    width: 100% !important;
    cursor: pointer !important;
    box-shadow: none !important;
    transition: background 0.15s !important;
}
#btn-analyze:hover, #btn-start:hover, #btn-video:hover {
    background: var(--lime2) !important;
    transform: none !important;
    box-shadow: none !important;
}

#btn-stop {
    background: transparent !important;
    border: 1px solid var(--wire) !important;
    border-radius: 0 !important;
    color: var(--mist) !important;
    font-family: var(--mono) !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 12px 16px !important;
    min-height: 44px !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
}
#btn-stop:hover {
    border-color: var(--red) !important;
    color: var(--red) !important;
    background: rgba(255,76,76,0.05) !important;
}

#btn-refresh {
    background: transparent !important;
    border: 1px solid var(--wire) !important;
    border-radius: 0 !important;
    color: var(--mist) !important;
    font-family: var(--mono) !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 12px 16px !important;
    min-height: 44px !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
}
#btn-refresh:hover {
    border-color: var(--lime) !important;
    color: var(--lime) !important;
    background: rgba(200,255,0,0.04) !important;
}

/* ═══ FORM ELEMENTS ═══ */
.gradio-container input[type="text"],
.gradio-container textarea,
.gradio-container .gr-textbox textarea {
    background: var(--ink) !important;
    border: 1px solid var(--wire) !important;
    border-radius: 0 !important;
    color: var(--lime) !important;
    font-family: var(--mono) !important;
    font-size: 12px !important;
    padding: 12px 14px !important;
    letter-spacing: 0.03em !important;
    line-height: 1.8 !important;
    transition: border-color 0.2s !important;
}
.gradio-container input:focus,
.gradio-container textarea:focus {
    border-color: var(--lime) !important;
    box-shadow: 0 0 0 1px rgba(200,255,0,0.15) !important;
    outline: none !important;
}

.gradio-container .gr-textbox,
.gradio-container [data-testid="textbox"] {
    background: var(--ink) !important;
    border: 1px solid var(--wire) !important;
    border-radius: 0 !important;
}

/* Slider */
.gradio-container input[type="range"] {
    -webkit-appearance: none !important;
    height: 3px !important;
    background: var(--wire) !important;
    border: none !important;
    border-radius: 0 !important;
    cursor: pointer !important;
    width: 100% !important;
}
.gradio-container input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none !important;
    width: 14px !important;
    height: 14px !important;
    background: var(--lime) !important;
    border: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    cursor: pointer !important;
}

/* ═══ IMAGE BOXES ═══ */
.gradio-container .gr-image,
.gradio-container [data-testid="image"] {
    border-radius: 0 !important;
    border: 1px solid var(--wire) !important;
    background: var(--ink2) !important;
    transition: border-color 0.2s !important;
    box-shadow: none !important;
}
.gradio-container .gr-image:hover,
.gradio-container [data-testid="image"]:hover {
    border-color: var(--lime) !important;
}
.gradio-container .upload-container,
.gradio-container [data-testid="image"] .upload-btn-wrapper {
    background: var(--ink2) !important;
    border: 1px dashed var(--wire) !important;
    border-radius: 0 !important;
}
.gradio-container .upload-container:hover {
    border-color: var(--lime) !important;
    background: rgba(200,255,0,0.02) !important;
}

/* ═══ INFO BANNER ═══ */
.info-banner {
    background: rgba(200,255,0,0.04);
    border: 1px solid rgba(200,255,0,0.2);
    padding: 12px 16px;
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.04em;
    color: var(--lime);
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 20px;
}
.info-banner::before { content: '//'; opacity: 0.5; }

/* ═══ ACCORDION ═══ */
.gradio-container .gr-accordion {
    background: var(--ink2) !important;
    border: 1px solid var(--wire) !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    overflow: hidden;
    margin-top: 24px;
}
.gradio-container .gr-accordion > .label-wrap {
    padding: 14px 20px !important;
    font-family: var(--mono) !important;
    font-size: 11px !important;
    font-weight: 400 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--mist) !important;
    transition: background 0.2s;
}
.gradio-container .gr-accordion > .label-wrap:hover {
    background: var(--ink3) !important;
}

/* ═══ HOW IT WORKS ═══ */
.how-grid {
    display: grid;
    grid-template-columns: repeat(3,1fr);
    gap: 0;
    border: 1px solid var(--wire);
    margin: 16px 0 8px;
}
.how-card {
    background: var(--ink2);
    border-right: 1px solid var(--wire);
    padding: 22px;
    transition: background 0.2s;
}
.how-card:last-child { border-right: none; }
.how-card:hover { background: var(--ink3); }
.how-step {
    font-family: var(--mono);
    font-size: 28px;
    font-weight: 700;
    color: var(--lime);
    line-height: 1;
    margin-bottom: 12px;
    opacity: 0.35;
}
.how-title {
    font-family: var(--mono);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--paper);
    margin-bottom: 8px;
}
.how-desc {
    font-size: 12px;
    font-weight: 300;
    color: var(--mist);
    line-height: 1.7;
}

/* ═══ FOOTER ═══ */
#fakescope-footer {
    padding: 24px 32px;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--wire);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border-top: 1px solid var(--wire);
    margin-top: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 18px;
    flex-wrap: wrap;
}
.sep {
    width: 3px;
    height: 3px;
    background: var(--wire);
    border-radius: 50%;
    display: inline-block;
    flex-shrink: 0;
}

/* ═══ STATUS DOT ═══ */
.status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--lime);
    display: inline-block;
    animation: dot-blink 2s infinite;
}
@keyframes dot-blink {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.25; }
}

/* ═══ SCROLLBAR ═══ */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--wire); border-radius: 0; }

/* ═══ SELECTION ═══ */
::selection { background: rgba(200,255,0,0.18); color: var(--paper); }

/* ═══ GRADIO OVERRIDES ═══ */
.gradio-container .svelte-1ed2p3z,
.gradio-container footer { display: none !important; }
.gradio-container .prose { color: var(--paper) !important; }
.gradio-container .gr-panel {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
.gradio-container * { border-radius: 0 !important; }
.gradio-container .gr-row { gap: 20px !important; }
.gradio-container .gr-column { gap: 14px !important; }
"""


# ══════════════════════════════════════════════════════════
# HTML BLOCKS
# ══════════════════════════════════════════════════════════

SPLASH_HTML = """
<div id="splash-overlay">
  <span class="splash-corner tl">FS-2025</span>
  <span class="splash-corner tr">v3.0.0</span>
  <span class="splash-corner bl">EMSI // 4IASDR</span>
  <span class="splash-corner br">RABAT · MA</span>

  <div id="splash-eyebrow">Deepfake Detection System</div>

  <div id="splash-logo">Fake<span>Scope</span></div>

  <p id="splash-sub">
    Real-time AI analysis for photos,<br>
    video streams and webcam feeds.
  </p>

  <div id="splash-tags">
    <span class="splash-tag">MobileNetV2</span>
    <span class="splash-tag">MTCNN</span>
    <span class="splash-tag">Grad-CAM</span>
    <span class="splash-tag">140k dataset</span>
  </div>

  <button id="splash-btn" onclick="
    var el = document.getElementById('splash-overlay');
    el.classList.add('hidden');
    setTimeout(function(){ el.style.display='none'; }, 600);
  ">Begin analysis</button>
</div>
"""

TOGGLE_JS = """
<script>
(function(){
    document.body.style.background = '#0d0d0d';
    document.documentElement.style.background = '#0d0d0d';
})();
</script>
"""

HERO_HTML = """
<div id="fakescope-hero">
  <div>
    <div class="hero-eyebrow">Yassuo &nbsp;·&nbsp; EMSI 4IASDR</div>
    <h1 class="hero-title">Fake<span>Scope</span><br>AI</h1>
    <p class="hero-sub">
      Analyse d'images, flux webcam et vidéos.<br>
      MobileNetV2 &nbsp;·&nbsp; MTCNN &nbsp;·&nbsp; Grad-CAM.
    </p>
    <div class="stat-row">
      <div class="stat-pill">
        <span class="stat-num">140k</span>
        <span class="stat-lbl">Images</span>
      </div>
      <div class="stat-pill">
        <span class="stat-num">160px</span>
        <span class="stat-lbl">Résolution</span>
      </div>
      <div class="stat-pill">
        <span class="stat-num">MTCNN</span>
        <span class="stat-lbl">Détection</span>
      </div>
      <div class="stat-pill">
        <span class="stat-num">GradCAM</span>
        <span class="stat-lbl">XAI</span>
      </div>
    </div>
  </div>

  <div id="hero-terminal">
    <span class="t-lime">$ fakescope --init</span><br>
    loading model...<br>
    <span class="t-lime">✓ model ready</span><br>
    threshold: 0.50<br>
    img_size:&nbsp; 160px<br>
    detector:&nbsp; MTCNN<br>
    xai:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; grad-cam<br>
    <span class="t-lime">$ <span class="cursor"></span></span>
  </div>
</div>
"""

HOW_HTML = """
<div class="how-grid">
  <div class="how-card">
    <div class="how-step">01</div>
    <div class="how-title">Détection MTCNN</div>
    <div class="how-desc">
      MTCNN localise le visage avec précision sub-pixel
      et l'extrait proprement pour l'analyse.
    </div>
  </div>
  <div class="how-card">
    <div class="how-step">02</div>
    <div class="how-title">Analyse MobileNetV2</div>
    <div class="how-desc">
      Fine-tuné sur 140 000 visages réels et synthétiques
      pour détecter les artefacts de génération.
    </div>
  </div>
  <div class="how-card">
    <div class="how-step">03</div>
    <div class="how-title">Grad-CAM XAI</div>
    <div class="how-desc">
      Visualise les zones suspectes qui ont
      déclenché la décision du modèle.
    </div>
  </div>
</div>
<p style="font-size:10px;color:var(--wire);margin:12px 0 4px;
          font-family:'Space Mono',monospace;letter-spacing:0.08em;text-transform:uppercase;">
  Outil académique &nbsp;·&nbsp; EMSI Rabat &nbsp;·&nbsp; 4IASDR
</p>
"""

FOOTER_HTML = """
<div id="fakescope-footer">
  <span>FakeScope AI</span>
  <span class="sep"></span>
  <span>MobileNetV2 + Transfer Learning</span>
  <span class="sep"></span>
  <span>140k Real &amp; Fake Faces</span>
  <span class="sep"></span>
  <span>Grad-CAM Explainability</span>
  <span class="sep"></span>
  <span>EMSI Rabat 4IASDR</span>
</div>
"""

STATUS_HTML = """
<div style="display:flex;align-items:center;gap:8px;padding:14px 32px 0;
            font-family:'Space Mono',monospace;font-size:11px;color:#888;
            letter-spacing:0.06em;">
  <span class="status-dot"></span>
  MODEL READY &nbsp;·&nbsp; THRESHOLD 0.50 &nbsp;·&nbsp; MTCNN ACTIVE
</div>
"""


# ══════════════════════════════════════════════════════════
# FONCTIONS COMMUNES
# ══════════════════════════════════════════════════════════
def extract_face(img_rgb):
    faces = detector.detect_faces(img_rgb)
    if not faces:
        return cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)), False, 0.0
    best       = max(faces, key=lambda x: x['confidence'])
    x, y, w, h = best['box']
    x, y       = max(0, x), max(0, y)
    face       = img_rgb[y:y+h, x:x+w]
    if face.size == 0:
        return cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)), False, 0.0
    return cv2.resize(face, (IMG_SIZE, IMG_SIZE)), True, best['confidence']


def predict_face(face_img):
    img_norm  = face_img.astype('float32') / 255.0
    img_input = np.expand_dims(img_norm, axis=0)
    score     = float(model.predict(img_input, verbose=0)[0][0])
    is_fake   = score >= THRESHOLD
    conf      = score if is_fake else (1 - score)
    return score, is_fake, conf, img_norm, img_input


def make_gradcam(img_norm, img_input):
    try:
        heatmap = get_gradcam_heatmap(model, img_input)
        overlay = overlay_gradcam(img_norm, heatmap, alpha=0.45)
        return (overlay * 255).astype(np.uint8)
    except:
        return (img_norm * 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════
# MODE 1 — UPLOAD PHOTO
# ══════════════════════════════════════════════════════════
def analyze_photo(pil_image):
    if pil_image is None:
        return None, None, "// awaiting input...", "", ""
    img_rgb                   = np.array(pil_image.convert('RGB'))
    face_img, found, conf_det = extract_face(img_rgb)
    score, is_fake, conf, img_norm, img_input = predict_face(face_img)
    gradcam_img               = make_gradcam(img_norm, img_input)
    verdict = (
        f"VERDICT: DEEPFAKE\nCONFIDANCE: {conf:.1%}\nSCORE: {score:.4f}"
        if is_fake else
        f"VERDICT: AUTHENTIC\nCONFIDANCE: {conf:.1%}\nSCORE: {score:.4f}"
    )
    detection = (
        f"FACE: DETECTED\nMTCNN CONF: {conf_det:.1%}"
        if found else
        "FACE: NOT FOUND\nUSING FULL IMAGE"
    )
    scores = (
        f"AUTHENTIC : {1-score:.4f}  ({1-score:.1%})\n"
        f"DEEPFAKE  : {score:.4f}  ({score:.1%})\n"
        f"THRESHOLD : {THRESHOLD}"
    )
    return (
        Image.fromarray(face_img),
        Image.fromarray(gradcam_img),
        verdict, detection, scores
    )


# ══════════════════════════════════════════════════════════
# MODE 2 — WEBCAM
# ══════════════════════════════════════════════════════════
webcam_running = False
webcam_thread  = None
latest_result  = {"verdict": "En attente...", "score": 0.5, "conf": 0.0}


def webcam_loop(seuil):
    global webcam_running, latest_result
    cap          = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    if not cap.isOpened():
        latest_result["verdict"] = "// camera not found"
        webcam_running = False
        return

    frame_count = 0
    prediction  = 0.5

    while webcam_running:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 5 == 0:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
            if len(faces) > 0:
                faces      = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                x, y, w, h = faces[0]
                margin     = int(0.2 * w)
                x1 = max(0, x - margin);   y1 = max(0, y - margin)
                x2 = min(frame.shape[1], x + w + margin)
                y2 = min(frame.shape[0], y + h + margin)
                face_rgb = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                face_res = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
                score, is_fake, conf, _, _ = predict_face(face_res)
                prediction = score
                label  = "DEEPFAKE" if is_fake else "REAL"
                color  = (255, 76, 76) if is_fake else (200, 255, 0)
                latest_result = {
                    "verdict": f"{'DEEPFAKE' if is_fake else 'REAL'}  ·  {conf:.1%}",
                    "score": score, "conf": conf
                }
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"{label}  {conf*100:.0f}%"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                cv2.rectangle(frame, (x1, y1-th-14), (x1+tw+12, y1), color, -1)
                cv2.putText(frame, text, (x1+6, y1-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (13, 13, 13), 2)

        h_f = frame.shape[0]
        cv2.putText(frame,
                    f"FakeScope | Score: {prediction:.3f} | Threshold: {seuil} | Q=Quit",
                    (12, h_f-14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (136, 136, 136), 1)
        cv2.imshow("FakeScope — Live Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    webcam_running = False
    latest_result["verdict"] = "// stream stopped"


def start_webcam(seuil):
    global webcam_running, webcam_thread
    if webcam_running:
        return "// stream already active"
    webcam_running = True
    webcam_thread  = threading.Thread(target=webcam_loop, args=(seuil,), daemon=True)
    webcam_thread.start()
    return "// stream started\n// opencv window open\n// press Q to quit"


def stop_webcam():
    global webcam_running
    webcam_running = False
    return "// stream stopped"


def get_webcam_result():
    return latest_result.get("verdict", "// awaiting...")


# ══════════════════════════════════════════════════════════
# MODE 3 — VIDÉO
# ══════════════════════════════════════════════════════════
def analyze_video(video_path, seuil, sample_rate):
    if video_path is None:
        return None, "// no video uploaded"

    cap          = cv2.VideoCapture(video_path)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    total_frames    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps             = cap.get(cv2.CAP_PROP_FPS)
    results         = []
    frames_analysed = 0
    frame_idx       = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % int(sample_rate) == 0:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
            if len(faces) > 0:
                faces      = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                x, y, w, h = faces[0]
                x1 = max(0, x - int(0.2*w));  y1 = max(0, y - int(0.2*h))
                x2 = min(frame.shape[1], x+w+int(0.2*w))
                y2 = min(frame.shape[0], y+h+int(0.2*h))
                face = cv2.resize(
                    cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB),
                    (IMG_SIZE, IMG_SIZE)
                )
                score, is_fake, conf, _, _ = predict_face(face)
                results.append({"frame": frame_idx, "score": score,
                                 "is_fake": is_fake, "conf": conf})
                frames_analysed += 1
        frame_idx += 1

    cap.release()

    if not results:
        return None, "// no faces detected in video"

    scores      = [r["score"] for r in results]
    fake_count  = sum(1 for r in results if r["is_fake"])
    real_count  = len(results) - fake_count
    avg_score   = np.mean(scores)
    verdict_vid = "DEEPFAKE DETECTED" if fake_count > real_count else "AUTHENTIC"

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 3.5))
    fig.patch.set_facecolor('#0d0d0d')
    ax.set_facecolor('#1a1a1a')

    frames_list = [r["frame"] for r in results]

    ax.fill_between(frames_list, scores, seuil,
                    where=[s >= seuil for s in scores],
                    color='#ff4c4c', alpha=0.25, label='Deepfake zone')
    ax.fill_between(frames_list, scores, seuil,
                    where=[s < seuil for s in scores],
                    color='#c8ff00', alpha=0.1, label='Authentic zone')
    ax.plot(frames_list, scores,
            color='#c8ff00', lw=1.5, label='Deepfake score',
            zorder=3, solid_capstyle='butt')
    ax.axhline(seuil, color='#ff4c4c', ls='--', lw=1,
               label=f'Threshold {seuil}', zorder=4)

    ax.set_xlabel('FRAME', fontsize=9, color='#3a3a3a', labelpad=8,
                  fontfamily='monospace', fontweight='bold', letter_spacing=0.1)
    ax.set_ylabel('SCORE', fontsize=9, color='#3a3a3a', labelpad=8,
                  fontfamily='monospace', fontweight='bold')
    ax.set_title('DEEPFAKE SCORE TIMELINE',
                 fontsize=11, fontweight='bold', color='#f5f4f0', pad=14,
                 fontfamily='monospace')
    ax.legend(framealpha=0.1, edgecolor='#3a3a3a', fontsize=9,
              loc='upper right', frameon=True, facecolor='#1a1a1a',
              labelcolor='#888888')
    ax.grid(True, alpha=0.08, color='#3a3a3a', linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_color('#3a3a3a')
        spine.set_linewidth(0.5)
    ax.tick_params(colors='#3a3a3a', labelsize=8)
    ax.set_ylim(0, 1)
    plt.tight_layout(pad=1.2)

    chart_path = os.path.join(BASE_DIR, "results", "video_analysis.png")
    os.makedirs(os.path.dirname(chart_path), exist_ok=True)
    plt.savefig(chart_path, dpi=160, bbox_inches='tight',
                facecolor='#0d0d0d', edgecolor='none')
    plt.close()

    summary = (
        f"VERDICT         : {verdict_vid}\n\n"
        f"FRAMES ANALYZED : {frames_analysed}\n"
        f"FAKE FRAMES     : {fake_count}  ({fake_count/frames_analysed:.1%})\n"
        f"REAL FRAMES     : {real_count}  ({real_count/frames_analysed:.1%})\n"
        f"AVG SCORE       : {avg_score:.4f}\n"
        f"DURATION        : {total_frames/fps:.1f}s  @  {fps:.0f} FPS"
    )
    return Image.open(chart_path), summary


# ══════════════════════════════════════════════════════════
# INTERFACE GRADIO
# ══════════════════════════════════════════════════════════
with gr.Blocks(css=CSS, theme=gr.themes.Base(), title="FakeScope AI") as app:

    # 1. Splash screen — injecté en premier
    gr.HTML(SPLASH_HTML)

    # 2. JS background fix
    gr.HTML(TOGGLE_JS)

    # 3. Status bar
    gr.HTML(STATUS_HTML)

    # 4. Hero
    gr.HTML(HERO_HTML)

    # 5. Tabs
    with gr.Tabs():

        # ── Onglet Photo ──────────────────────────────────
        with gr.Tab("📸  Photo"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("<div class='section-label'>Image à analyser</div>")
                    photo_input = gr.Image(
                        type="pil", label="", show_label=False, height=260
                    )
                    photo_btn = gr.Button(
                        "Analyser l'image →", elem_id="btn-analyze"
                    )
                    gr.HTML("""
                    <div style="border:1px solid var(--wire);padding:16px;margin-top:4px;background:var(--ink2)">
                      <div class="section-label">Instructions</div>
                      <p style="font-size:12px;color:var(--mist);line-height:1.8;
                                margin:0;font-family:'Space Mono',monospace;">
                        Upload a portrait or face photo.<br>
                        FakeScope auto-extracts the face<br>
                        and maps artifacts via Grad-CAM.
                      </p>
                    </div>
                    """)

                with gr.Column(scale=1):
                    gr.HTML("<div class='section-label'>Visage extrait</div>")
                    photo_face = gr.Image(label="", show_label=False, height=190)
                    gr.HTML("<div class='section-label' style='margin-top:14px'>Carte Grad-CAM</div>")
                    photo_gradcam = gr.Image(label="", show_label=False, height=190)

                with gr.Column(scale=1):
                    gr.HTML("<div class='section-label'>Verdict</div>")
                    photo_verdict = gr.Textbox(
                        label="", show_label=False, lines=3,
                        value="// awaiting input..."
                    )
                    gr.HTML("<div class='section-label'>Détection visage</div>")
                    photo_detection = gr.Textbox(label="", show_label=False, lines=2)
                    gr.HTML("<div class='section-label'>Scores de probabilité</div>")
                    photo_scores = gr.Textbox(label="", show_label=False, lines=4)

            photo_btn.click(
                fn=analyze_photo, inputs=[photo_input],
                outputs=[photo_face, photo_gradcam,
                         photo_verdict, photo_detection, photo_scores]
            )
            photo_input.upload(
                fn=analyze_photo, inputs=[photo_input],
                outputs=[photo_face, photo_gradcam,
                         photo_verdict, photo_detection, photo_scores]
            )

        # ── Onglet Webcam ─────────────────────────────────
        with gr.Tab("🎥  Webcam"):
            gr.HTML("""
            <div class="info-banner">
                Stream opens in separate OpenCV window.
                Press <strong>Q</strong> in that window to close.
            </div>
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("<div class='section-label'>Seuil de détection</div>")
                    webcam_seuil = gr.Slider(
                        0.1, 0.9, value=0.5, step=0.05,
                        label="", show_label=False
                    )
                    webcam_start = gr.Button(
                        "▶  Start stream", elem_id="btn-start"
                    )
                    with gr.Row():
                        webcam_stop    = gr.Button("Stop",    elem_id="btn-stop")
                        webcam_refresh = gr.Button("Refresh", elem_id="btn-refresh")

                with gr.Column(scale=1):
                    gr.HTML("<div class='section-label'>Statut</div>")
                    webcam_status = gr.Textbox(
                        label="", show_label=False, lines=3,
                        value="// offline"
                    )
                    gr.HTML("<div class='section-label'>Dernier résultat</div>")
                    webcam_result = gr.Textbox(
                        label="", show_label=False, lines=2,
                        value="// awaiting..."
                    )

            webcam_start.click(
                fn=start_webcam, inputs=[webcam_seuil], outputs=[webcam_status]
            )
            webcam_stop.click(fn=stop_webcam, outputs=[webcam_status])
            webcam_refresh.click(fn=get_webcam_result, outputs=[webcam_result])

        # ── Onglet Vidéo ──────────────────────────────────
        with gr.Tab("🎬  Vidéo"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("<div class='section-label'>Fichier vidéo</div>")
                    video_input = gr.Video(label="", show_label=False)
                    gr.HTML("<div class='section-label'>Seuil de détection</div>")
                    video_seuil = gr.Slider(
                        0.1, 0.9, value=0.5, step=0.05,
                        label="", show_label=False
                    )
                    gr.HTML("<div class='section-label'>Échantillonnage (1 frame / N)</div>")
                    video_sample = gr.Slider(
                        1, 30, value=10, step=1,
                        label="", show_label=False
                    )
                    video_btn = gr.Button(
                        "Analyser la vidéo →", elem_id="btn-video"
                    )

                with gr.Column(scale=2):
                    gr.HTML("<div class='section-label'>Score deepfake par frame</div>")
                    video_chart = gr.Image(label="", show_label=False)
                    gr.HTML("<div class='section-label'>Résumé de l'analyse</div>")
                    video_summary = gr.Textbox(
                        label="", show_label=False, lines=8,
                        value="// no analysis yet"
                    )

            video_btn.click(
                fn=analyze_video,
                inputs=[video_input, video_seuil, video_sample],
                outputs=[video_chart, video_summary]
            )

    # 6. How it works
    with gr.Accordion("//  How FakeScope works", open=False):
        gr.HTML(HOW_HTML)

    # 7. Footer
    gr.HTML(FOOTER_HTML)


# ══════════════════════════════════════════════════════════
# 🚀 Lancement
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )