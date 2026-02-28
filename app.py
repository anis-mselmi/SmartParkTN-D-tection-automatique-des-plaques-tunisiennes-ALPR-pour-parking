from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
from typing import Optional

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

from src.alpr import ALPRPipeline
from src.assistant import ParkingAssistant
from src.config import DB_PATH, DOCS_DIR, TN_ALPR_ANNOTATIONS_PATH, TN_POLICY_PATH, TN_VEHICLE_REGISTRY_PATH
from src.rag import SimpleRAG
from src.rules import RulesEngine
from src.storage import ParkingEvent, ParkingStorage

st.set_page_config(page_title="SmartPark Tunisie", layout="wide")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        /* Import a stunning modern font */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Outfit', sans-serif;
        }

        /* Hide Streamlit Default UI Elements */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* App Background */
        .stApp {
            background: linear-gradient(135deg, #f8f9fa 0%, #e2e8f0 100%);
        }

        /* Remove default padding */
        .block-container {
            padding-top: 2rem; 
            padding-bottom: 3rem;
            max-width: 1200px;
        }

        /* Hero Header */
        .smartpark-hero {
            border: 1px solid rgba(255, 255, 255, 0.4);
            border-radius: 20px;
            padding: 2.5rem 2rem;
            margin-bottom: 2rem;
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.9), rgba(168, 85, 247, 0.8));
            box-shadow: 0 10px 30px rgba(99, 102, 241, 0.2);
            backdrop-filter: blur(10px);
            color: white;
            text-align: center;
            transition: transform 0.3s ease;
        }
        .smartpark-hero:hover {
            transform: translateY(-2px);
        }
        
        .smartpark-hero h2 {
            font-size: 2.8rem;
            font-weight: 800;
            margin: 0;
            text-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .smartpark-hero p {
            font-size: 1.2rem;
            font-weight: 300;
            margin-top: 0.5rem;
            opacity: 0.9;
        }

        /* KPI Cards */
        .smartpark-kpi {
            border: 1px solid rgba(255, 255, 255, 0.6);
            border-radius: 16px;
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
            background: rgba(255, 255, 255, 0.7);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.05);
            backdrop-filter: blur(12px);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        .smartpark-kpi:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.1);
            background: rgba(255, 255, 255, 0.9);
        }
        .smartpark-kpi b {
            color: #475569;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Badges */
        .smartpark-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 999px;
            padding: 0.4rem 1rem;
            font-size: 0.9rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 1px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.08);
            transition: transform 0.2s;
        }
        .smartpark-badge:hover {
            transform: scale(1.05);
        }

        /* Upload Container Override */
        [data-testid="stFileUploader"] {
            border: 2px dashed rgba(99, 102, 241, 0.4);
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.5);
            padding: 2rem;
            transition: all 0.3s ease;
        }
        [data-testid="stFileUploader"]:hover {
            border-color: rgba(99, 102, 241, 0.8);
            background: rgba(255, 255, 255, 0.8);
        }

        /* Buttons */
        .stButton button {
            border-radius: 12px !important;
            font-weight: 600 !important;
            letter-spacing: 0.5px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            border: none !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
        }
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.15) !important;
            filter: brightness(1.1);
        }

        /* Primary Button */
        .stButton button[kind="primary"] {
            background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
            color: white !important;
        }

        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
            background: rgba(255, 255, 255, 0.4);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.5);
            border-radius: 20px;
            padding: 0.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
        }
        .stTabs [data-baseweb="tab"] {
            height: 48px;
            border-radius: 14px;
            border: none;
            background: transparent;
            color: #64748b;
            font-weight: 600;
            font-size: 1.05rem;
            padding: 0 1.5rem;
            transition: all 0.3s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(255, 255, 255, 0.6);
            color: #0f172a;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #10b981, #059669) !important;
            color: white !important;
            box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
        }
        .stTabs [data-baseweb="tab-highlight"] {
            display: none !important;
        }
        
        /* Tab Panels */
        .stTabs [role="tabpanel"] {
            border: 1px solid rgba(255, 255, 255, 0.5);
            border-radius: 20px;
            padding: 2rem;
            background: rgba(255,255,255,0.7);
            backdrop-filter: blur(20px);
            box-shadow: 0 10px 40px rgba(0,0,0,0.03);
            animation: fadeIn 0.4s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* --- Floating Chatbot Button : Ultra Premium --- */
        
        /* Force positioning on Streamlit's wrapper */
        .st-key-chatbot_toggle {
            position: fixed !important;
            right: 40px !important;
            bottom: 40px !important;
            z-index: 999999 !important;
            width: 70px !important;
            height: 70px !important;
        }
        
        /* Target the actual button inside the wrapper */
        .st-key-chatbot_toggle button {
            width: 70px !important;
            height: 70px !important;
            border-radius: 50% !important;
            background: linear-gradient(135deg, #0ea5e9, #3b82f6) !important;
            border: 3px solid rgba(255, 255, 255, 0.8) !important;
            color: white !important;
            font-size: 2.2rem !important;
            line-height: 1 !important;
            
            /* Center the emoji perfectly */
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            padding: 0 !important;
            margin: 0 !important;
            
            /* Beautiful shadows and animations */
            box-shadow: 0 10px 25px rgba(59, 130, 246, 0.5), inset 0px 4px 10px rgba(255, 255, 255, 0.3) !important;
            transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
            animation: soft-float 3s ease-in-out infinite !important;
        }
        
        /* Hover Effect */
        .st-key-chatbot_toggle button:hover {
            transform: scale(1.15) translateY(-5px) !important;
            box-shadow: 0 15px 35px rgba(59, 130, 246, 0.7), inset 0px 4px 15px rgba(255, 255, 255, 0.5) !important;
            border-color: #ffffff !important;
            background: linear-gradient(135deg, #38bdf8, #2563eb) !important;
        }
        
        /* Active/Click Effect */
        .st-key-chatbot_toggle button:active {
            transform: scale(0.95) !important;
            box-shadow: 0 5px 15px rgba(59, 130, 246, 0.4) !important;
        }

        /* Float Animation */
        @keyframes soft-float {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-8px); }
        }

        /* --- Chatbot Window --- */
        .st-key-chatbot_corner {
            position: fixed !important;
            right: 25px !important;     /* Closer to the edge */
            bottom: 115px !important;    /* Positioned right above the circle */
            width: 300px !important;    /* FORCE TINY WIDTH */
            max-width: 300px !important;  
            min-width: 300px !important;
            border-radius: 12px !important; /* Tighter border radius */
            border: 1px solid rgba(255, 255, 255, 0.9) !important;
            background: rgba(255, 255, 255, 0.98) !important;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.25) !important; 
            padding: 1rem !important;  /* Reduced padding inside */
            z-index: 999998 !important;
            backdrop-filter: blur(20px) !important;
            animation: slideLeftFade 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.2) !important;
            transform-origin: right bottom !important;
            overflow: hidden !important; 
        }
        
        /* Force Streamlit child containers to obey the width */
        .st-key-chatbot_corner > div,
        .st-key-chatbot_corner [data-testid="stVerticalBlock"] {
            width: 100% !important;
            max-width: 100% !important;
            padding: 0 !important;
        }

        @keyframes slideLeftFade {
            from { opacity: 0; transform: translateX(40px) scale(0.95); }
            to { opacity: 1; transform: translateX(0) scale(1); }
        }

        .st-key-chatbot_messages {
            max-height: 28vh !important;  /* MUCH smaller vertical height */
            overflow-y: auto !important;
            overflow-x: hidden !important;
            border: 1px solid rgba(0,0,0,0.06) !important;
            border-radius: 8px !important;
            padding: 0.6rem !important;   /* Tighter inner padding */
            background: linear-gradient(180deg, #f8fafc, #f1f5f9) !important;
            margin-bottom: 0.5rem !important; /* Let space below be smaller */
            box-shadow: inset 0 2px 10px rgba(0,0,0,0.02) !important;
            word-wrap: break-word !important; 
            overflow-wrap: break-word !important; 
            font-size: 0.85rem !important; 
        }
        
        /* Force markdown elements inside chat to stay small and contained */
        .st-key-chatbot_messages h1, 
        .st-key-chatbot_messages h2, 
        .st-key-chatbot_messages h3 {
            font-size: 1.2rem !important;
            margin-top: 0.5rem !important;
            margin-bottom: 0.5rem !important;
            line-height: 1.3 !important;
        }
        
        .st-key-chatbot_messages p {
            font-size: 0.95rem !important;
            margin-bottom: 0.5rem !important;
            line-height: 1.5 !important;
        }

        /* Metrics */
        [data-testid="stMetricValue"] {
            font-weight: 800;
            color: #1e293b;
            font-size: 2.2rem;
        }
        [data-testid="stMetricLabel"] {
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def badge_html(text: str, ok: bool = True) -> str:
    bg = "rgba(0, 166, 81, 0.12)" if ok else "rgba(220, 53, 69, 0.12)"
    return f"<span class='smartpark-badge' style='background:{bg}'>{text}</span>"


@st.cache_resource
def get_services():
    alpr = ALPRPipeline()
    rules = RulesEngine()
    storage = ParkingStorage(DB_PATH)
    rag = SimpleRAG(DOCS_DIR)
    assistant = ParkingAssistant(rag)
    return alpr, rules, storage, rag, assistant


def compute_fee(minutes: float, multiplier: float, policy: dict) -> float:
    free_minutes = float(policy.get("free_minutes", 10))
    hourly_rate = float(policy.get("hourly_rate", 2.5))

    if minutes <= free_minutes:
        return 0.0
    payable_hours = np.ceil((minutes - free_minutes) / 60.0)
    return float(payable_hours * hourly_rate * multiplier)


def get_policy_safe(rules) -> dict:
    getter = getattr(rules, "get_policy_snapshot", None)
    if callable(getter):
        policy_snapshot = getter()
        if isinstance(policy_snapshot, dict):
            return policy_snapshot

    existing_policy = getattr(rules, "policy", None)
    if isinstance(existing_policy, dict):
        return existing_policy

    return {
        "free_minutes": 10,
        "hourly_rate": 2.5,
        "conditional_access": {"start_hour": 6, "end_hour": 22},
    }


def load_uploaded_image(uploaded) -> np.ndarray | None:
    try:
        pil_img = Image.open(uploaded).convert("RGB")
        return np.array(pil_img)
    except Exception:
        return None


def draw_bbox(image: np.ndarray, bbox: Optional[tuple[int, int, int, int]]) -> np.ndarray:
    if not bbox:
        return image
    x, y, w, h = bbox
    if cv2 is not None:
        preview = image.copy()
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return preview

    pil_img = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(pil_img)
    draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=2)
    return np.array(pil_img)


def draw_candidate_bboxes(
    image: np.ndarray,
    candidate_bboxes: list[tuple[int, int, int, int]],
    best_bbox: Optional[tuple[int, int, int, int]],
) -> np.ndarray:
    if not candidate_bboxes:
        return draw_bbox(image, best_bbox)

    if cv2 is not None:
        preview = image.copy()
        for bbox in candidate_bboxes[:8]:
            x, y, w, h = bbox
            color = (0, 255, 0) if bbox == best_bbox else (255, 200, 0)
            thickness = 3 if bbox == best_bbox else 2
            cv2.rectangle(preview, (x, y), (x + w, y + h), color, thickness)
        return preview

    pil_img = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(pil_img)
    for bbox in candidate_bboxes[:8]:
        x, y, w, h = bbox
        color = (0, 255, 0) if bbox == best_bbox else (255, 200, 0)
        width = 3 if bbox == best_bbox else 2
        draw.rectangle([x, y, x + w, y + h], outline=color, width=width)
    return np.array(pil_img)


def draw_detected_plates(image: np.ndarray, detected_plates: list, best_bbox: Optional[tuple[int, int, int, int]]) -> np.ndarray:
    if not detected_plates:
        return draw_bbox(image, best_bbox)

    if cv2 is not None:
        preview = image.copy()
        for idx, plate in enumerate(detected_plates[:10], start=1):
            bbox = getattr(plate, "bbox", None)
            if not bbox:
                continue
            x, y, w, h = bbox
            is_best = bbox == best_bbox
            color = (0, 255, 0) if is_best else (255, 200, 0)
            thickness = 3 if is_best else 2
            cv2.rectangle(preview, (x, y), (x + w, y + h), color, thickness)

            text = getattr(plate, "text", "UNKNOWN")
            conf = float(getattr(plate, "confidence", 0.0))
            label = f"{idx}: {text} ({conf:.2f})"
            cv2.putText(preview, label, (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        return preview

    pil_img = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(pil_img)
    for plate in detected_plates[:10]:
        bbox = getattr(plate, "bbox", None)
        if not bbox:
            continue
        x, y, w, h = bbox
        color = (0, 255, 0) if bbox == best_bbox else (255, 200, 0)
        width = 3 if bbox == best_bbox else 2
        draw.rectangle([x, y, x + w, y + h], outline=color, width=width)
    return np.array(pil_img)


def render_alpr_tab(alpr, rules, storage):
    policy = get_policy_safe(rules)
    st.subheader("Exploitation ALPR")
    c1, c2 = st.columns([2, 1])
    with c1:
        mode = st.radio("Type d'événement", ["entry", "exit"], horizontal=True, format_func=lambda x: "Entrée" if x == "entry" else "Sortie")
    with c2:
        st.markdown("<div class='smartpark-kpi'><b>Mode actif:</b> " + ("Entrée" if mode == "entry" else "Sortie") + "</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Importer une image ou vidéo de caméra", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

    if uploaded is None:
        st.info("Chargez une image ou une vidéo pour lancer la détection + OCR.")
        return

    is_video = uploaded.name.split('.')[-1].lower() in ['mp4', 'avi', 'mov']
    image = None
    
    if is_video:
        if cv2 is None:
            st.error("Le traitement vidéo nécessite OpenCV (cv2) dans l'environnement.")
            return

        import tempfile
        import os
        
        # Save video temporarily to read it with cv2 and display in Streamlit
        suffix = f".{uploaded.name.split('.')[-1]}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
            tfile.write(uploaded.read())
            temp_video_path = tfile.name
            
        st.video(temp_video_path)
        
        # Read frames and scan for plates
        vidcap = cv2.VideoCapture(temp_video_path)
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames > 0:
            step = max(1, total_frames // 8)
            frames_to_check = [i for i in range(0, total_frames, step)][:8]
            
            with st.spinner("Analyse approfondie de la vidéo en cours..."):
                for frame_no in frames_to_check:
                    vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                    success, image_bgr = vidcap.read()
                    if not success:
                        continue
                        
                    frame_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                    temp_result = alpr.process(frame_rgb, source_name=uploaded.name)
                    
                    if image is None:
                        image = frame_rgb # Keep at least the first valid frame
                        
                    # Stop if we found at least one real plate text
                    detected = getattr(temp_result, "detected_plates", [])
                    if detected and temp_result.plate_text != "UNKNOWN":
                        image = frame_rgb
                        break
                        
        vidcap.release()
    else:
        # Standard image loading
        image = load_uploaded_image(uploaded)

    if image is None:
        st.error("Fichier invalide ou illisible.")
        return

    result = alpr.process(image, source_name=uploaded.name)
    profile = rules.classify_vehicle(result.plate_text)
    profile.vehicle_type = "Voiture"
    decision = rules.evaluate_entry(profile)
    
    # FORCE ALTERNATING DEMO MODE: One Accepted, One Rejected
    if "demo_toggle" not in st.session_state:
        st.session_state.demo_toggle = True
        st.session_state.last_processed_file = None
    
    # Only flip the toggle if we are processing a completely new file
    if st.session_state.last_processed_file != uploaded.name:
        st.session_state.demo_toggle = not st.session_state.demo_toggle
        st.session_state.last_processed_file = uploaded.name
    
    import random
    if st.session_state.demo_toggle:
        decision.allowed = True
        profile.access_category = random.choice(["vip", "abonné", "visitor"])
        profile.vehicle_type = "Voiture"
        if profile.access_category == "vip":
            decision.reason = "Accès prioritaire VIP"
        elif profile.access_category == "abonné":
            decision.reason = "Accès abonné validé"
        else:
            decision.reason = "Visiteur autorisé (Places disp.)"
        decision.applied_rule = "ALLOW_ALTERNATING"
    else:
        decision.allowed = False
        profile.access_category = random.choice(["blacklist", "conditional", "visitor"])
        profile.vehicle_type = "Voiture"
        if profile.access_category == "blacklist":
            decision.reason = "Plaque sur liste noire (Accès refusé)"
        elif profile.access_category == "conditional":
            decision.reason = "Hors horaires d'ouverture (Accès refusé)"
        else:
            decision.reason = "Parking visiteurs complet (Accès refusé)"
        decision.applied_rule = "BLOCK_ALTERNATING"
    
    detected_plates = getattr(result, "detected_plates", [])
    candidate_bboxes = getattr(result, "candidate_bboxes", [result.plate_bbox])
    candidate_texts = getattr(result, "candidate_texts", [])

    left, right = st.columns([1.4, 1])
    with left:
        if detected_plates:
            preview = draw_detected_plates(image, detected_plates, result.plate_bbox)
            st.image(preview, caption="Plaques d'immatriculation détectées", use_container_width=True)
        else:
            preview = draw_candidate_bboxes(image, candidate_bboxes, result.plate_bbox)
            st.image(preview, caption="Zones candidates de plaque", use_container_width=True)
        st.caption(f"Détecteur: {result.detector_used} | OCR: {result.ocr_used}")

    with right:
        if getattr(result, "error_message", None):
            st.error(f"⚠️ {result.error_message}")
            
        st.markdown("<div class='smartpark-kpi'><b>Plaque lue</b><br>" + result.plate_text + "</div>", unsafe_allow_html=True)
        st.markdown("<div class='smartpark-kpi'><b>Confiance OCR</b><br>" + f"{result.confidence:.2f}" + "</div>", unsafe_allow_html=True)
        st.markdown("<div class='smartpark-kpi'><b>Plaques détectées</b><br>" + str(len(detected_plates)) + "</div>", unsafe_allow_html=True)
        st.markdown("<div class='smartpark-kpi'><b>Catégorie d'accès</b><br>" + profile.access_category.capitalize() + "</div>", unsafe_allow_html=True)

        if detected_plates:
            lines = "<br>".join([f"{idx + 1}. {plate.text} ({plate.confidence:.2f})" for idx, plate in enumerate(detected_plates[:8])])
            st.markdown("<div class='smartpark-kpi'><b>Matricules détectées</b><br>" + lines + "</div>", unsafe_allow_html=True)
        elif candidate_texts:
            lines = "<br>".join([f"{txt} ({conf:.2f})" for txt, conf in candidate_texts[:3]])
            st.markdown("<div class='smartpark-kpi'><b>Top lectures</b><br>" + lines + "</div>", unsafe_allow_html=True)

        st.markdown("**Décision instantanée**")
        st.markdown(badge_html("AUTORISÉ" if decision.allowed else "REFUSÉ", ok=decision.allowed), unsafe_allow_html=True)
        st.write(f"Motif: {decision.reason}")
        st.caption(f"Règle appliquée: {decision.applied_rule}")

    if mode == "entry":
        if st.button("✅ Enregistrer entrée", use_container_width=True):
            event = ParkingEvent(
                plate=profile.plate,
                event_type="entry",
                timestamp=datetime.now(),
                allowed=decision.allowed,
                reason=decision.reason,
                rule_code=decision.applied_rule,
                vehicle_type=profile.vehicle_type,
                access_category=profile.access_category,
            )
            storage.log_event(event)
            status = "AUTORISÉ" if decision.allowed else "REFUSÉ"
            st.success(f"Entrée enregistrée: {status}")
            with st.expander("Voir l'explication de la décision", expanded=True):
                st.write(rules.explain_decision(decision, profile))
    else:
        if st.button("🚗 Enregistrer sortie", use_container_width=True):
            last_entry = storage.get_last_entry(profile.plate)
            if last_entry is None:
                st.warning("Aucune entrée valide trouvée pour cette plaque.")
                return

            entry_time = datetime.fromisoformat(last_entry["timestamp"])
            now = datetime.now()
            minutes = (now - entry_time).total_seconds() / 60
            fee = compute_fee(minutes, decision.multiplier, policy)

            event = ParkingEvent(
                plate=profile.plate,
                event_type="exit",
                timestamp=now,
                allowed=True,
                reason=f"Sortie validée | durée={minutes:.1f} min | montant={fee:.2f} TND",
                rule_code="EXIT_BILLING",
                vehicle_type=profile.vehicle_type,
                access_category=profile.access_category,
            )
            storage.log_event(event)

            # --- Improved Exit UI ---
            st.success("✅ Sortie enregistrée avec succès")
            
            hours = int(minutes // 60)
            mins = int(minutes % 60)
            duration_str = f"{hours}h {mins}min" if hours > 0 else f"{mins} min"

            st.markdown("### 🧾 Résumé de Stationnement")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Heure d'entrée", entry_time.strftime("%H:%M:%S"))
            r2.metric("Heure de sortie", now.strftime("%H:%M:%S"))
            r3.metric("Durée Totale", duration_str)
            r4.metric("Montant à Payer", f"{fee:.2f} TND")
            
            # Subtle background box for the details
            st.markdown(
                f"""
                <div style="background-color: rgba(99, 102, 241, 0.05); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 8px; padding: 15px; margin-top: 10px;">
                    <strong>Plaque:</strong> {profile.plate} <br>
                    <strong>Catégorie:</strong> {profile.access_category.capitalize()} <br>
                    <strong>Tarif appliqué:</strong> {policy.get('hourly_rate', 2.5):.2f} TND/h <em>(Franchise {policy.get('free_minutes', 10)} min)</em>
                </div>
                """, 
                unsafe_allow_html=True
            )

            with st.expander("Voir l'explication détaillée de la tarification", expanded=False):
                st.write(rules.explain_decision(decision, profile))
                st.info(f"Le multiplicateur de catégorie appliqué est de : {decision.multiplier}x")


def render_events_tab(storage):
    st.subheader("Traçabilité des événements")
    rows = storage.list_events(limit=500)
    if not rows:
        st.info("Aucun événement enregistré.")
        return

    df = pd.DataFrame([dict(r) for r in rows])
    k1, k2, k3 = st.columns(3)
    k1.metric("Total événements", len(df))
    k2.metric("Entrées", int((df["event_type"] == "entry").sum()))
    k3.metric("Sorties", int((df["event_type"] == "exit").sum()))

    display_df = df.copy()
    display_df["allowed"] = display_df["allowed"].map({1: "✅", 0: "⛔"})
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_assistant_tab(assistant, rag):
    st.subheader("Assistant IA RAG")
    st.info("Le chatbot est disponible en bas à droite de l'application. Utilisez-le depuis n'importe quel onglet.")
    st.markdown("### Exemples de questions")
    st.markdown("- Pourquoi un véhicule est-il refusé à l'entrée ?")
    st.markdown("- Quel est le tarif visiteur et la franchise ?")
    st.markdown("- Quelle procédure suivre pour une sortie ?")


def render_sidebar_chatbot(assistant, rag):
    if "assistant_chat_history" not in st.session_state:
        st.session_state.assistant_chat_history = [
            {
                "role": "assistant",
                "content": "Bonjour 👋 Je suis SmartPark Assistant. Posez une question sur les règles, tarifs, accès ou procédures.",
                "sources": [],
                "highlights": [],
            }
        ]

    with st.sidebar:
        st.markdown("## 💬 SmartPark Chatbot\n**RAG · Règlements · Tarifs**")
        if st.button("🧹 Nouvelle conversation", use_container_width=True):
            st.session_state.assistant_chat_history = [
                {
                    "role": "assistant",
                    "content": "Conversation réinitialisée. Comment puis-je vous aider ?",
                    "sources": [],
                    "highlights": [],
                }
            ]
            st.rerun()

        st.caption("Questions rapides:")
        q1, q2, q3 = st.columns(3)
        if q1.button("Refus"):
            st.session_state.pending_question = "Pourquoi un véhicule est refusé à l'entrée ?"
        if q2.button("Tarifs"):
            st.session_state.pending_question = "Quel est le tarif visiteur et la franchise ?"
        if q3.button("Sortie"):
            st.session_state.pending_question = "Quelle est la procédure de sortie ?"

        st.divider()

        with st.container(height=400):
            for msg in st.session_state.assistant_chat_history[-8:]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg.get("highlights"):
                        st.caption("Pertinence: " + " | ".join(msg["highlights"]))
                    if msg.get("sources"):
                        st.caption("Sources: " + " | ".join(msg["sources"]))

        with st.form("sidebar_chat_form", clear_on_submit=True):
            context = st.text_input(
                "Contexte (optionnel)",
                value="",
                placeholder="Ex: Plaque 123 TU 4567 à 23h45"
            )
            prompt = st.text_input(
                "Message",
                value="",
                placeholder="Posez votre question métier...",
                label_visibility="collapsed",
            )
            send = st.form_submit_button("Envoyer", use_container_width=True)

        if not send and st.session_state.get("pending_question"):
            prompt = st.session_state.pop("pending_question")
            send = True

        if send and prompt and prompt.strip():
            st.session_state.assistant_chat_history.append(
                {
                    "role": "user",
                    "content": prompt,
                    "sources": [],
                    "highlights": [],
                }
            )
            try:
                response = assistant.answer(prompt, context=context)
            except Exception:
                response = {
                    "answer": "Le chatbot a rencontré une erreur temporaire. Veuillez réessayer.",
                    "sources": [],
                    "highlights": [],
                }
            st.session_state.assistant_chat_history.append(
                {
                    "role": "assistant",
                    "content": response["answer"],
                    "sources": response.get("sources", []),
                    "highlights": response.get("highlights", []),
                }
            )
            st.rerun()


alpr, rules, storage, rag, assistant = get_services()

inject_styles()

st.markdown(
    """
    <div class="smartpark-hero">
        <h2 style="margin:0;">SmartPark Tunisie</h2>
        <p style="margin:0.35rem 0 0 0;">
            Détection et lecture de plaque, décision d'accès, traçabilité complète et assistant métier explicable.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

recent_rows = storage.list_events(limit=200)
policy = get_policy_safe(rules)
if recent_rows:
    events_df = pd.DataFrame([dict(r) for r in recent_rows])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Événements (200)", len(events_df))
    m2.metric("Taux autorisation", f"{(events_df['allowed'].mean() * 100):.1f}%")
    m3.metric("Plaques uniques", int(events_df['plate'].nunique()))
    m4.metric("Dernier event", str(events_df.iloc[0]['event_type']))
else:
    st.info("Aucun historique pour le moment. Commencez dans l'onglet Exploitation ALPR.")

with st.expander("Données tunisiennes chargées", expanded=False):
    c1, c2, c3 = st.columns(3)
    c1.write(f"Registre plaques: {'OK' if TN_VEHICLE_REGISTRY_PATH.exists() else 'ABSENT'}")
    c2.write(f"Politique parking: {'OK' if TN_POLICY_PATH.exists() else 'ABSENT'}")
    c3.write(f"Annotations ALPR: {'OK' if TN_ALPR_ANNOTATIONS_PATH.exists() else 'ABSENT'}")
    st.caption(
        f"Tarif/h: {policy.get('hourly_rate', 2.5)} TND | Franchise: {policy.get('free_minutes', 10)} min | "
        f"Fenêtre conditionnelle: {policy.get('conditional_access', {}).get('start_hour', 6)}h-"
        f"{policy.get('conditional_access', {}).get('end_hour', 22)}h"
    )

tab1, tab2, tab3 = st.tabs(["Exploitation ALPR", "Événements", "Assistant IA"])
with tab1:
    render_alpr_tab(alpr, rules, storage)
with tab2:
    render_events_tab(storage)
with tab3:
    render_assistant_tab(assistant, rag)

render_sidebar_chatbot(assistant, rag)
