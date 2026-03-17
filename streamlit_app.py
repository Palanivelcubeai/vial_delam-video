import io
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vial Delamination Detection – Video",
    page_icon="🎬",
    layout="wide",
)

# ── Inspection zone (fractions of frame width/height) ─────────────────────────
ROI_X1_F, ROI_X2_F = 0.02, 0.27
ROI_Y1_F, ROI_Y2_F = 0.05, 0.95

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; }
  .stat-chip { background:#f8fafc; border:1px solid #e2e8f0; border-radius:9px;
               padding:10px 14px; text-align:center; margin-bottom:6px; }
  .stat-chip .num { font-size:1.3rem; font-weight:700; color:#1a202c; }
  .stat-chip .lbl { font-size:0.65rem; color:#94a3b8; text-transform:uppercase; letter-spacing:.4px; }
  .banner-ok   { background:#f0fdf4; color:#15803d; border:1.5px solid #86efac;
                 border-radius:10px; padding:12px 16px; font-weight:700; margin-bottom:8px; }
  .banner-warn { background:#fff7ed; color:#c2410c; border:1.5px solid #fdba74;
                 border-radius:10px; padding:12px 16px; font-weight:700; margin-bottom:8px; }
  .zone-empty  { background:#e0f7fa; color:#0277bd; border:1.5px solid #81d4fa;
                 border-radius:10px; padding:10px 14px; font-size:.85rem; margin-bottom:8px; }
  .zone-active-ok   { background:#f0fdf4; color:#15803d; border:1.5px solid #86efac;
                      border-radius:10px; padding:10px 14px; margin-bottom:8px; }
  .zone-active-warn { background:#fff7ed; color:#c2410c; border:1.5px solid #fdba74;
                      border-radius:10px; padding:10px 14px; margin-bottom:8px; }
  .zone-cls  { font-size:.92rem; font-weight:700; margin-bottom:2px; }
  .zone-conf { font-size:.78rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── Model loader ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading YOLO model…")
def load_model():
    model_path = Path(__file__).parent / "best1.pt"
    if not model_path.exists():
        st.error(f"Model file not found: {model_path}\nPlace 'best .pt' in the same folder.")
        st.stop()
    return YOLO(str(model_path))

model = load_model()

# ── Session state defaults ─────────────────────────────────────────────────────
for key, val in {
    "running": False,
    "stop_requested": False,
    "result_video": None,
    "final_stats": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("## 🎬 Vial Delamination Detection — Video")
st.caption("Upload a video and run live frame-by-frame YOLO inspection.")
st.divider()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Detection Settings")
    conf = st.slider("Confidence Threshold", 0.05, 0.95, 0.25, 0.05)
    iou  = st.slider("IoU Threshold",        0.05, 0.95, 0.45, 0.05)
    st.divider()
    st.caption("🎬 Video Inspection\nPowered by YOLOv8 + Streamlit")

# ── Layout ─────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 2], gap="large")

with left:
    st.markdown("#### Upload Video")
    uploaded = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv", "wmv", "flv"],
        key="video_upload",
    )

    if uploaded:
        st.caption(f"📹 **{uploaded.name}**  ({uploaded.size / 1_048_576:.1f} MB)")

    col_run, col_stop = st.columns(2)
    with col_run:
        run_btn = st.button(
            "▶ Run",
            disabled=not uploaded or st.session_state.running,
            use_container_width=True,
            type="primary",
        )
    with col_stop:
        stop_btn = st.button(
            "■ Stop",
            disabled=not st.session_state.running,
            use_container_width=True,
        )

    if stop_btn:
        st.session_state.stop_requested = True

    # Live stats panel (updated during processing)
    st.markdown("#### Live Stats")
    stat_frame     = st.empty()
    stat_inspected = st.empty()
    stat_zone      = st.empty()

with right:
    st.markdown("#### Live Detection Feed")
    frame_placeholder = st.empty()
    progress_bar      = st.empty()
    progress_text     = st.empty()

    # Placeholder when idle
    if not st.session_state.running and st.session_state.result_video is None:
        frame_placeholder.markdown("""
        <div style="text-align:center;padding:80px 20px;color:#94a3b8;
                    background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;">
          <div style="font-size:3rem;">🧪</div>
          <p><strong style="color:#475569;">No Results Yet</strong></p>
          <p>Upload a video and click <strong>▶ Run</strong> to start.</p>
        </div>
        """, unsafe_allow_html=True)

# ── Processing ─────────────────────────────────────────────────────────────────
if run_btn and uploaded:
    st.session_state.running       = True
    st.session_state.stop_requested = False
    st.session_state.result_video  = None
    st.session_state.final_stats   = None

    # Save uploaded video to a temp file
    suffix = Path(uploaded.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
        tmp_in.write(uploaded.read())
        input_path = tmp_in.name

    # Output video temp file
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_out.close()
    output_path = tmp_out.name

    try:
        cap         = cv2.VideoCapture(input_path)
        fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        bottles_inspected = 0
        prev_zone_active  = False
        all_classes: dict = {}
        frame_idx         = 0
        pb = progress_bar.progress(0, text="Processing…")

        while True:
            if st.session_state.stop_requested:
                progress_text.warning("⏹ Stopped by user.")
                break

            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            rx1 = int(w * ROI_X1_F); rx2 = int(w * ROI_X2_F)
            ry1 = int(h * ROI_Y1_F); ry2 = int(h * ROI_Y2_F)

            results = model.predict(source=frame, conf=conf, iou=iou, verbose=False)
            r = results[0]

            # Best detection whose center is inside the zone
            best = None
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0]
                cx = float((x1 + x2) / 2)
                cy = float((y1 + y2) / 2)
                if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                    if best is None or float(b.conf) > float(best.conf):
                        best = b

            zone_active = best is not None
            if zone_active and not prev_zone_active:
                bottles_inspected += 1
            prev_zone_active = zone_active

            # Draw annotated frame
            annotated  = frame.copy()
            roi_color  = (0, 220, 220)
            det_payload = None

            if best is not None:
                bx1 = int(best.xyxy[0][0]); by1 = int(best.xyxy[0][1])
                bx2 = int(best.xyxy[0][2]); by2 = int(best.xyxy[0][3])
                cls_name  = model.names[int(best.cls)]
                conf_val  = round(float(best.conf), 4)
                is_delam  = "delaminated" in cls_name.lower() and "non" not in cls_name.lower()
                box_color = (0, 0, 220) if is_delam else (0, 200, 0)
                roi_color = box_color

                cv2.rectangle(annotated, (bx1, by1), (bx2, by2), box_color, 2)
                label = f"{cls_name}  {conf_val:.0%}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
                cv2.rectangle(annotated, (bx1, by1 - lh - 10), (bx1 + lw + 8, by1), box_color, -1)
                cv2.putText(annotated, label, (bx1 + 4, by1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

                if cls_name not in all_classes or conf_val > all_classes[cls_name]:
                    all_classes[cls_name] = conf_val
                det_payload = {"class": cls_name, "conf": conf_val}

            cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), roi_color, 3)
            zone_label = "IN ZONE" if zone_active else "INSPECTION ZONE"
            cv2.putText(annotated, zone_label, (rx1 + 5, ry1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, roi_color, 2)

            writer.write(annotated)

            # Update UI every 5 frames to stay fast
            if frame_idx % 5 == 0:
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb, channels="RGB", use_container_width=True)

                pct = (frame_idx + 1) / max(total, 1)
                pb.progress(min(pct, 1.0), text=f"Frame {frame_idx + 1} / {total}")

                # Live stats
                stat_frame.markdown(
                    f'<div class="stat-chip"><div class="num">{frame_idx + 1}</div>'
                    f'<div class="lbl">Frame / {total}</div></div>',
                    unsafe_allow_html=True,
                )
                stat_inspected.markdown(
                    f'<div class="stat-chip"><div class="num">{bottles_inspected}</div>'
                    f'<div class="lbl">Bottles Inspected</div></div>',
                    unsafe_allow_html=True,
                )

                if zone_active and det_payload:
                    is_d = "delaminated" in det_payload["class"].lower() and "non" not in det_payload["class"].lower()
                    css  = "zone-active-warn" if is_d else "zone-active-ok"
                    stat_zone.markdown(
                        f'<div class="{css}"><div class="zone-cls">● {det_payload["class"]}</div>'
                        f'<div class="zone-conf">Confidence: {det_payload["conf"]*100:.1f}%</div></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    stat_zone.markdown(
                        '<div class="zone-empty">○ Zone is empty</div>',
                        unsafe_allow_html=True,
                    )

            frame_idx += 1

        cap.release()
        writer.release()

        # Load processed video for display / download
        if not st.session_state.stop_requested:
            pb.progress(1.0, text="Complete ✓")
            with open(output_path, "rb") as f:
                st.session_state.result_video = f.read()

            is_delaminated = any(
                "delaminated" in k.lower() and "non" not in k.lower()
                for k in all_classes
            )
            st.session_state.final_stats = {
                "frames": frame_idx,
                "bottles": bottles_inspected,
                "is_delaminated": is_delaminated,
                "classes": all_classes,
            }

    finally:
        if os.path.exists(input_path):
            os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)
        st.session_state.running       = False
        st.session_state.stop_requested = False

# ── Final results (shown after processing) ─────────────────────────────────────
if st.session_state.result_video and st.session_state.final_stats:
    stats = st.session_state.final_stats

    with right:
        st.divider()
        st.markdown("#### Final Result")

        if stats["is_delaminated"]:
            st.markdown('<div class="banner-warn">⚠️ Delaminated vial detected in video</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="banner-ok">✅ No delamination detected</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f'<div class="stat-chip"><div class="num">{stats["frames"]}</div>'
                f'<div class="lbl">Total Frames</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="stat-chip"><div class="num">{stats["bottles"]}</div>'
                f'<div class="lbl">Bottles Inspected</div></div>',
                unsafe_allow_html=True,
            )

        if stats["classes"]:
            st.markdown("**Detected Classes (best confidence)**")
            for cls, cf in stats["classes"].items():
                st.progress(cf, text=f"{cls}  —  {cf*100:.1f}%")

        st.download_button(
            "⬇️  Download Annotated Video",
            data=st.session_state.result_video,
            file_name="annotated_result.mp4",
            mime="video/mp4",
            use_container_width=True,
        )
