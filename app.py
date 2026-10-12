"""Sugarcane Leaf Disease Detection and Advisory System - Streamlit UI (Ch.1.5 / 5.7)."""
import os, io, cv2, uuid, pandas as pd, streamlit as st
from PIL import Image
from core import db, model, preprocess as pp
from core.advisory import DISCLAIMER
st.set_page_config(page_title="Sugarcane Disease Detection", page_icon="🌾", layout="wide")
db.init(); os.makedirs("data/uploads", exist_ok=True); os.makedirs("data/heatmaps", exist_ok=True)
S = st.session_state

def login_page():
    st.title("🌾 Sugarcane Leaf Disease Detection using AI & XAI")
    t1, t2 = st.tabs(["Login", "Register"])
    with t1:
        u = st.text_input("Username", key="lu"); p = st.text_input("Password", type="password", key="lp")
        if st.button("Login", type="primary"):
            r = db.login(u, p)
            if r: S.user = dict(r); st.rerun()
            else: st.error("Invalid username or password.")
        st.caption("Demo accounts: admin/admin123, expert/expert123 (change before deployment).")
    with t2:
        u = st.text_input("New username", key="ru"); p = st.text_input("New password (min 6)", type="password", key="rp")
        if st.button("Create farmer account"):
            if len(u) < 3 or len(p) < 6: st.error("Username >= 3 chars and password >= 6 chars.")
            elif db.register(u, p): st.success("Account created. Please log in.")
            else: st.error("Username already exists.")

def analyze_page():
    st.header("Upload Leaf Image & Detect Disease")
    if model.is_demo():
        st.warning("DEMO MODE: no trained model found at models/sugarcane_cnn.keras. Results come from a colour heuristic, not a CNN. Run train.py to enable real predictions.")
    f = st.file_uploader("Sugarcane leaf image (JPG/PNG)", type=["jpg", "jpeg", "png"], key=f"up{S.get('reset',0)}")
    if not f: return
    ok, msg, rgb = pp.validate(f.getvalue())
    if not ok: st.error(f"Invalid image: {msg} Please upload another."); return
    if msg != "OK": st.warning(msg)
    c1, c2 = st.columns(2); c1.image(rgb, caption="Uploaded image", use_container_width=True)
    if st.button("🔍 Analyze Leaf", type="primary"):
        try:
            with st.spinner("Analysing..."):
                x = pp.preprocess(rgb); disease, conf, idx = model.predict(x)
                sev, ratio = model.severity(x); rec = model.recovery_days(disease, sev)
                heat = model.gradcam(x, idx); adv = db.get_advisory(disease)
                tag = uuid.uuid4().hex[:10]; ip, hp = f"data/uploads/{tag}.png", f"data/heatmaps/{tag}.png"
                Image.fromarray(x).save(ip); Image.fromarray(heat).save(hp)
                db.save_prediction(S.user["id"], ip, disease, conf, sev, rec, hp)
            S.result = dict(rgb=heat, disease=disease, conf=conf, sev=sev, ratio=ratio, rec=rec, adv=adv)
        except Exception as e: st.error(f"Processing failed: {e}")
    r = S.get("result")
    if r:
        c2.image(r["rgb"], caption="Grad-CAM heatmap (regions influencing prediction)", use_container_width=True)
        m = st.columns(4); m[0].metric("Disease", r["disease"]); m[1].metric("Confidence", f"{r['conf']*100:.1f}%")
        m[2].metric("Severity", f"{r['sev']} ({r['ratio']*100:.0f}% area)"); m[3].metric("Recovery", "—" if r["rec"] == 0 else f"~{r['rec']} days")
        if r["conf"] < 0.6: st.info("Low confidence: retake a clearer, well-lit photo or consult an expert.")
        a, b = st.columns(2); a.subheader("Treatment"); a.write(r["adv"]["treatment"]); a.caption("Pesticide: " + r["adv"]["pesticides"])
        b.subheader("Preventive measures"); b.write(r["adv"]["prevention"]); st.caption(DISCLAIMER)
        if st.button("Reset / Clear"): S.pop("result", None); S.reset = S.get("reset", 0) + 1; st.rerun()

def history_page():
    st.header("Prediction History")
    rows = db.history(None if S.user["role"] in ("admin", "expert") else S.user["id"])
    if not rows: st.info("No predictions yet."); return
    df = pd.DataFrame(rows); st.dataframe(df.drop(columns=["file_path", "heatmap_path"]), use_container_width=True)
    st.download_button("Download CSV", df.drop(columns=["file_path", "heatmap_path"]).to_csv(index=False), "history.csv")
    pid = st.selectbox("View record", df["id"])
    r = df[df.id == pid].iloc[0]; c = st.columns(2)
    if os.path.exists(r.file_path): c[0].image(r.file_path, caption="Leaf")
    if os.path.exists(r.heatmap_path): c[1].image(r.heatmap_path, caption="Grad-CAM")
    if st.button("Delete record"): db.delete_prediction(int(pid)); st.rerun()

def trends_page():
    st.header("Disease Trends & System Monitoring")
    rows = db.history()
    if rows:
        df = pd.DataFrame(rows); st.bar_chart(df["disease"].value_counts()); st.line_chart(df.groupby(df.predicted_on.str[:10]).size())
        st.metric("Total predictions", len(df)); st.metric("Mean confidence", f"{df.confidence_pct.mean():.1f}%")
    else: st.info("No data yet.")
    if S.user["role"] == "admin": st.subheader("Manage Users"); st.dataframe(pd.DataFrame(db.users()), use_container_width=True)

if "user" not in S: login_page(); st.stop()
role = S.user["role"]; pages = {"Analyze Leaf": analyze_page, "Prediction History": history_page}
if role in ("admin", "expert"): pages["Trends & Monitoring"] = trends_page
with st.sidebar:
    st.title("🌾 Sugarcane AI"); st.write(f"Logged in as **{S.user['username']}** ({role})")
    page = st.radio("Navigate", list(pages)); 
    if st.button("Logout"): S.clear(); st.rerun()
pages[page]()
