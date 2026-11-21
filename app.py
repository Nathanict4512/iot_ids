"""
Real-time Deep Learning Intrusion Detection System for IoT Networks
Streamlit administrator dashboard.  Run locally with:  streamlit run app.py
"""
import os
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import ids_core as core

st.set_page_config(page_title="IoT Intrusion Detection System", page_icon="🛡️", layout="wide")


# ---------------------------------------------------------------- settings
def secret(key, default=""):
    try:
        return st.secrets[key]
    except Exception:  # no secrets file, or key missing
        return os.environ.get(key, default)


CFG = {k: secret(k) for k in ["SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD",
                              "ALERT_TO", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]}
ADMIN_USER = secret("ADMIN_USER", "admin")
ADMIN_PASSWORD = secret("ADMIN_PASSWORD", "admin123")
ALERT_COOLDOWN_S = 60  # minimum gap between external (email/Telegram) alerts

ATTACK_COLOURS = {"Normal": "#2e9d6b", "DDoS": "#d64545", "Port Scan": "#e08a1e",
                  "Botnet": "#7b4fd1", "Brute Force": "#c2377a"}


@st.cache_resource(show_spinner="Loading the trained detection model...")
def get_detector():
    return core.Detector()


core.init_db()
for key, val in {"authed": False, "running": False, "ticks": [], "recent": None,
                 "correct": 0, "seen": 0, "last_ext_alert": 0.0, "tick_no": 0}.items():
    st.session_state.setdefault(key, val)

SHOW_COLS = ["id", "ts", "source", "protocol", "label", "confidence", "pkt_rate",
             "unique_dst_ports", "auth_attempts", "latency_ms", "status", "note"]
COL_CFG = {
    "confidence": st.column_config.ProgressColumn("confidence", min_value=0, max_value=1, format="%.2f"),
    "pkt_rate": st.column_config.NumberColumn("pkt_rate", format="%.1f"),
    "latency_ms": st.column_config.NumberColumn("latency_ms", format="%.2f"),
}


# ------------------------------------------------------------------- login
def login_page():
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.title("🛡️ IoT IDS")
        st.caption("Real-time deep learning intrusion detection for IoT networks")
        with st.form("login"):
            user = st.text_input("Username")
            pw = st.text_input("Password", type="password")
            go_ = st.form_submit_button("Sign in", width="stretch")
        if go_:
            if core.check_login(user, pw, ADMIN_USER, ADMIN_PASSWORD):
                st.session_state.authed = True
                st.rerun()
            else:
                st.error("Incorrect username or password.")
        if ADMIN_PASSWORD == "admin123":
            st.info("Demo login: **admin** / **admin123**. Set ADMIN_USER and ADMIN_PASSWORD "
                    "in the app secrets to change it.")


# --------------------------------------------------------------- dashboard
def page_dashboard():
    st.header("Overview")
    s = core.stats()
    c = st.columns(4)
    c[0].metric("Flows analysed", f"{s['total']:,}")
    c[1].metric("Intrusions detected", f"{s['intrusions']:,}")
    c[2].metric("Intrusion rate", f"{(s['intrusions'] / s['total'] * 100 if s['total'] else 0):.1f}%")
    c[3].metric("Avg. response time", f"{s['avg_latency']:.2f} ms/flow")
    if not s["total"]:
        st.info("No traffic has been analysed yet. Open **Live Monitor** and switch it on, "
                "or use **Analyse Upload**.")
        return
    left, right = st.columns(2)
    with left:
        st.subheader("Traffic by class")
        fig = px.bar(s["by_label"], x="label", y="flows", color="label",
                     color_discrete_map=ATTACK_COLOURS)
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10), xaxis_title=None)
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("Activity per minute")
        tl = s["timeline"]
        fig = go.Figure()
        fig.add_bar(x=tl["minute"], y=tl["normal"], name="Normal", marker_color=ATTACK_COLOURS["Normal"])
        fig.add_bar(x=tl["minute"], y=tl["intrusions"], name="Intrusion", marker_color="#d64545")
        fig.update_layout(barmode="stack", margin=dict(t=10, b=10), legend_orientation="h")
        st.plotly_chart(fig, width="stretch")
    st.subheader("Latest alerts")
    latest = core.fetch(only_alerts=True, limit=8)
    st.dataframe(latest[SHOW_COLS], hide_index=True, width="stretch", column_config=COL_CFG)


# ------------------------------------------------------------ live monitor
def run_tick(detector, n_flows, ratio, send_external):
    batch = core.generate_flows(n_flows, attack_ratio=ratio)
    res, total_ms = detector.predict(batch)
    core.save_detections(res, "Live simulation")
    ss = st.session_state
    ss.tick_no += 1
    ss.seen += len(res)
    ss.correct += int((res["prediction"] == res["true_label"]).sum())
    intr = res[res["is_intrusion"]]
    ss.ticks = (ss.ticks + [{"tick": ss.tick_no, "Normal": int((~res["is_intrusion"]).sum()),
                             "Intrusion": len(intr), "ms": total_ms}])[-60:]
    ss.recent = pd.concat([res, ss.recent]).head(200) if ss.recent is not None else res
    if len(intr):
        st.toast(f"🚨 {len(intr)} intrusion(s): {', '.join(sorted(intr['prediction'].unique()))}")
        if send_external and time.time() - ss.last_ext_alert >= ALERT_COOLDOWN_S:
            core.notify(core.format_alert(intr), CFG)
            ss.last_ext_alert = time.time()


def page_live():
    st.header("Live Monitor")
    st.caption("A simulated IoT network feeds flows to the trained CNN-LSTM model. Each flow is "
               "classified, stored, and an alert is raised for intrusions. (Streamlit Cloud cannot "
               "sniff real packets, so traffic is generated here.)")
    detector = get_detector()
    a, b, c, d = st.columns([1, 1.2, 1, 1])
    st.session_state.running = a.toggle("Run live monitor", value=st.session_state.running)
    ratio = b.slider("Attack share of traffic", 0, 80, 25, step=5, format="%d%%") / 100
    n_flows = c.slider("Flows per second-tick", 5, 60, 20)
    interval = d.slider("Refresh every (s)", 1, 5, 2)
    ext = st.checkbox(f"Also send email/Telegram alerts (max one per {ALERT_COOLDOWN_S}s)",
                      value=bool(core.channels_configured(CFG)),
                      disabled=not core.channels_configured(CFG))

    @st.fragment(run_every=interval if st.session_state.running else None)
    def panel():
        if st.session_state.running:
            run_tick(detector, n_flows, ratio, ext)
        ss = st.session_state
        if ss.recent is None:
            st.info("Switch on **Run live monitor** to start streaming traffic.")
            return
        k = st.columns(4)
        k[0].metric("Flows this session", f"{ss.seen:,}")
        k[1].metric("Agreement with simulated truth", f"{ss.correct / max(ss.seen, 1):.1%}")
        k[2].metric("Last batch time", f"{ss.ticks[-1]['ms']:.1f} ms")
        k[3].metric("Status", "🟢 Monitoring" if ss.running else "⏸️ Paused")
        t = pd.DataFrame(ss.ticks)
        fig = go.Figure()
        fig.add_bar(x=t["tick"], y=t["Normal"], name="Normal", marker_color=ATTACK_COLOURS["Normal"])
        fig.add_bar(x=t["tick"], y=t["Intrusion"], name="Intrusion", marker_color="#d64545")
        fig.update_layout(barmode="stack", height=260, margin=dict(t=10, b=10),
                          xaxis_title="Batch", yaxis_title="Flows", legend_orientation="h")
        st.plotly_chart(fig, width="stretch")
        show = ss.recent.head(15).copy()
        show["alert"] = np.where(show["is_intrusion"], "🚨 " + show["prediction"], "✅ Normal")
        st.dataframe(
            show[["alert", "protocol", "pkt_rate", "src_bytes", "unique_dst_ports", "auth_attempts",
                  "confidence", "true_label"]].rename(columns={"true_label": "simulated_truth"}),
            hide_index=True, width="stretch", column_config=COL_CFG)

    panel()


# ----------------------------------------------------------- upload analyse
def page_upload():
    st.header("Analyse Upload")
    st.caption("Upload a CSV of network flows to classify it in one go.")
    detector = get_detector()
    sample = core.generate_flows(200, np.random.default_rng(7)).rename(columns={"true_label": "label"})
    st.download_button("Download a sample CSV", sample.to_csv(index=False).encode(),
                       "sample_iot_flows.csv", "text/csv")
    with st.expander("Required columns"):
        st.code(", ".join(core.REQUIRED_COLUMNS))
    file = st.file_uploader("Flow CSV", type="csv")
    if not file:
        return
    try:
        df, notes = core.validate_frame(pd.read_csv(file))
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read the file. {e}")
        return
    for n in notes:
        st.warning(n)
    if df.empty:
        st.error("No usable rows were found.")
        return
    res, total_ms = detector.predict(df)
    c = st.columns(4)
    c[0].metric("Flows", f"{len(res):,}")
    c[1].metric("Intrusions", f"{int(res['is_intrusion'].sum()):,}")
    c[2].metric("Total time", f"{total_ms:.0f} ms")
    c[3].metric("Per flow", f"{total_ms / len(res):.3f} ms")
    if "label" in df.columns and df["label"].isin(core.CLASSES).all():
        acc = (res["prediction"] == df["label"]).mean()
        st.success(f"Your file includes labels. Accuracy against them: **{acc:.2%}**")
    st.plotly_chart(px.pie(res, names="prediction", color="prediction", hole=0.45,
                           color_discrete_map=ATTACK_COLOURS), width="stretch")
    out_cols = [c for c in res.columns if c != "true_label"]
    st.dataframe(res[out_cols], hide_index=True, width="stretch")
    left, right = st.columns(2)
    left.download_button("Download results", res[out_cols].to_csv(index=False).encode(),
                         "ids_results.csv", "text/csv")
    if right.button("Save these detections to history"):
        core.save_detections(res, f"Upload: {file.name}")
        st.toast("Saved to history.")


# ------------------------------------------------------------------ alerts
def page_alerts():
    st.header("Intrusion Alerts")
    f1, f2 = st.columns(2)
    labels = f1.multiselect("Attack type", core.CLASSES[1:])
    status = f2.multiselect("Status", ["New", "Acknowledged", "False positive"], default=["New"])
    df = core.fetch(only_alerts=True, labels=labels, statuses=status, limit=500)
    st.caption(f"{len(df)} alert(s) shown. Select rows, then take action.")
    event = st.dataframe(df[SHOW_COLS], hide_index=True, width="stretch", column_config=COL_CFG,
                         on_select="rerun", selection_mode="multi-row")
    rows = event.selection.rows if event else []
    ids = df.iloc[rows]["id"].tolist() if rows else []
    note = st.text_input("Note (optional)")
    a, b, c = st.columns(3)
    if a.button("✅ Acknowledge", disabled=not ids, width="stretch"):
        core.update_status(ids, "Acknowledged", note)
        st.rerun()
    if b.button("🚫 Mark false positive", disabled=not ids, width="stretch"):
        core.update_status(ids, "False positive", note)
        st.rerun()
    if rows:
        with c.popover("Show flow details", width="stretch"):
            st.json(pd.read_json(df.iloc[rows[0]]["features"], typ="series").to_dict())


# ----------------------------------------------------------------- history
def page_history():
    st.header("Detection History")
    f = st.columns(3)
    labels = f[0].multiselect("Class", core.CLASSES)
    only_alerts = f[1].checkbox("Intrusions only")
    limit = f[2].number_input("Rows", 100, 20000, 1000, step=100)
    df = core.fetch(only_alerts=only_alerts, labels=labels, limit=limit)
    st.caption(f"{len(df):,} record(s)")
    st.dataframe(df[SHOW_COLS], hide_index=True, width="stretch", column_config=COL_CFG)
    st.download_button("Download CSV", df.drop(columns=["features"]).to_csv(index=False).encode(),
                       "ids_history.csv", "text/csv")


# -------------------------------------------------------------- evaluation
def page_eval():
    st.header("Model Evaluation")
    m = get_detector().metrics
    st.caption(f"{m['data_note']}  Train {m['n_train']:,} · Validation {m['n_val']:,} · "
               f"Test {m['n_test']:,} flows (test set unseen during training).")
    st.subheader("Intrusion vs normal (binary)")
    b = m["binary"]
    c = st.columns(5)
    for col, (name, key) in zip(c, [("Accuracy", "accuracy"), ("Precision", "precision"),
                                    ("Recall", "recall"), ("F1-score", "f1"), ("ROC-AUC", "roc_auc")]):
        col.metric(name, f"{b[key]:.2%}")
    st.caption(f"False positive rate: {b['false_positive_rate']:.2%}")

    st.subheader("Per attack type (5-class)")
    mc = m["multiclass"]
    rows = [{"Class": k, "Precision": v["precision"], "Recall": v["recall"],
             "F1-score": v["f1-score"], "Support": int(v["support"])} for k, v in mc["per_class"].items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        k: st.column_config.NumberColumn(format="%.3f") for k in ["Precision", "Recall", "F1-score"]})
    st.caption(f"Overall accuracy {mc['accuracy']:.2%} · macro F1 {mc['macro_f1']:.3f}")

    l, r = st.columns(2)
    with l:
        st.subheader("Confusion matrix")
        fig = px.imshow(m["confusion_matrix"], x=m["classes"], y=m["classes"], text_auto=True,
                        color_continuous_scale="Blues", labels=dict(x="Predicted", y="Actual"))
        fig.update_layout(margin=dict(t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
    with r:
        st.subheader("Training history")
        h = m["history"]
        fig = go.Figure()
        for k, name in [("accuracy", "Train acc."), ("val_accuracy", "Validation acc.")]:
            fig.add_scatter(y=h[k], name=name, mode="lines+markers")
        fig.update_layout(margin=dict(t=10, b=10), xaxis_title="Epoch", yaxis_title="Accuracy",
                          legend_orientation="h")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Response speed")
    lat = m["latency_ms"]
    c = st.columns(3)
    c[0].metric("One flow at a time", f"{lat['single_mean']:.1f} ms", f"p95 {lat['single_p95']:.1f} ms",
                delta_color="off")
    c[1].metric("Batch of 32 (per flow)", f"{lat['batch32_per_flow']:.2f} ms")
    if c[2].button("Re-run benchmark on this server"):
        det = get_detector()
        flows = core.generate_flows(100, np.random.default_rng(3))
        times = []
        for i in range(100):
            _, t = det.predict(flows.iloc[[i]])
            times.append(t)
        st.success(f"Measured now: mean {np.mean(times):.1f} ms, p95 {np.percentile(times, 95):.1f} ms "
                   "per flow (single-flow prediction).")


# ---------------------------------------------------------------- settings
def page_settings():
    st.header("Alert Settings")
    ch = core.channels_configured(CFG)
    if ch:
        st.success("Alert channels active: " + ", ".join(ch))
        if st.button("Send a test alert"):
            st.json(core.notify("IoT IDS test alert: notifications are working.", CFG))
    else:
        st.warning("No email or Telegram channel is set up. In-app alerts still work. "
                   "Add the keys from `.streamlit/secrets.toml.example` in the app's Secrets "
                   "(or your local secrets file) to enable them.")
    st.divider()
    st.subheader("Maintenance")
    if st.checkbox("I understand this deletes all stored detections"):
        if st.button("Clear detection history", type="primary"):
            core.clear_history()
            st.session_state.update(ticks=[], recent=None, correct=0, seen=0, tick_no=0)
            st.success("History cleared.")


# -------------------------------------------------------------------- main
PAGES = {"📊 Overview": page_dashboard, "📡 Live Monitor": page_live,
         "📁 Analyse Upload": page_upload, "🚨 Alerts": page_alerts,
         "🗂️ History": page_history, "🎯 Model Evaluation": page_eval,
         "⚙️ Settings": page_settings}

if not st.session_state.authed:
    login_page()
    st.stop()

with st.sidebar:
    st.title("🛡️ IoT IDS")
    choice = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.divider()
    s = core.stats()
    new = len(core.fetch(only_alerts=True, statuses=["New"], limit=10000))
    st.metric("Open alerts", new)
    if st.button("Sign out", width="stretch"):
        st.session_state.authed = False
        st.session_state.running = False
        st.rerun()

PAGES[choice]()
