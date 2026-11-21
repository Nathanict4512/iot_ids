"""
ids_core.py
Core engine for the IoT Intrusion Detection System (IDS).

Contains: flow-data generator, pre-processing, the detector (loads the trained
CNN-LSTM), the SQLite store for detections/alerts, and alert notification
(email / Telegram). It does not depend on Streamlit, so it can be reused or
tested on its own.
"""
from __future__ import annotations

import hmac
import json
import os
import smtplib
import sqlite3
import time
import urllib.parse
import urllib.request
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
MODEL_DIR = BASE / "model"
DB_PATH = BASE / "data" / "ids.db"

CLASSES = ["Normal", "DDoS", "Port Scan", "Botnet", "Brute Force"]
PROTOCOLS = ["TCP", "UDP", "ICMP", "MQTT"]
NUMERIC = [
    "duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts",
    "pkt_rate", "byte_rate", "avg_pkt_size", "iat_mean", "iat_std",
    "syn_count", "ack_count", "rst_count", "fin_count",
    "unique_dst_ports", "unique_dst_ips", "auth_attempts", "payload_entropy",
]
REQUIRED_COLUMNS = ["protocol"] + NUMERIC


# --------------------------------------------------------------------------
# 1. Synthetic IoT flow generator (used for training and for the live demo)
# --------------------------------------------------------------------------
def _frame(proto, duration, ps, pd_, sb, db, syn, ack, rst, fin,
           uports, uips, auth, ent, iat_cv):
    duration = np.maximum(duration, 0.001)
    ps = np.maximum(np.rint(ps), 1).astype(int)
    pd_ = np.maximum(np.rint(pd_), 0).astype(int)
    total = ps + pd_
    sb, db = np.maximum(sb, 20), np.maximum(db, 0)
    byts = sb + db
    iat_mean = duration / total
    ints = lambda a: np.maximum(np.rint(a), 0).astype(int)  # noqa: E731
    return pd.DataFrame({
        "protocol": proto,
        "duration": duration,
        "src_bytes": np.rint(sb).astype(int),
        "dst_bytes": np.rint(db).astype(int),
        "src_pkts": ps,
        "dst_pkts": pd_,
        "pkt_rate": total / duration,
        "byte_rate": byts / duration,
        "avg_pkt_size": byts / total,
        "iat_mean": iat_mean,
        "iat_std": iat_mean * iat_cv,
        "syn_count": ints(syn),
        "ack_count": ints(ack),
        "rst_count": ints(rst),
        "fin_count": ints(fin),
        "unique_dst_ports": np.maximum(ints(uports), 1),
        "unique_dst_ips": np.maximum(ints(uips), 1),
        "auth_attempts": ints(auth),
        "payload_entropy": np.clip(ent, 0, 8),
    })


def _is_tcp(proto):
    return np.isin(proto, ["TCP", "MQTT"]).astype(int)


def _normal(m, rng):
    proto = rng.choice(PROTOCOLS, m, p=[0.45, 0.25, 0.25, 0.05])
    tcp = _is_tcp(proto)
    ps, pd_ = rng.integers(2, 40, m), rng.integers(2, 40, m)
    duration = rng.lognormal(0.5, 0.9, m)
    burst = rng.random(m) < 0.03                       # occasional harmless bursts
    ps = np.where(burst, ps * rng.integers(8, 30, m), ps)
    duration = np.where(burst, duration * 0.3, duration)
    sb = ps * rng.lognormal(4.8, 0.6, m)
    db = pd_ * rng.lognormal(5.2, 0.7, m)
    return _frame(proto, duration, ps, pd_, sb, db,
                  tcp * rng.integers(1, 3, m), tcp * pd_,
                  tcp * (rng.random(m) < 0.03), tcp * (rng.random(m) < 0.8),
                  rng.integers(1, 3, m), rng.integers(1, 3, m),
                  (rng.random(m) < 0.05) * rng.integers(1, 3, m),
                  rng.normal(4.2, 0.8, m), rng.uniform(0.5, 1.5, m))


def _ddos(m, rng):
    proto = rng.choice(PROTOCOLS, m, p=[0.5, 0.3, 0.15, 0.05])
    tcp = _is_tcp(proto)
    ps, pd_ = rng.integers(150, 5000, m), rng.integers(0, 30, m)
    duration = rng.lognormal(-0.8, 0.8, m)
    return _frame(proto, duration, ps, pd_, ps * rng.lognormal(3.9, 0.4, m),
                  pd_ * 60, tcp * ps * rng.uniform(0.5, 1, m), tcp * pd_,
                  tcp * pd_ * 0.5, np.zeros(m), rng.integers(1, 3, m),
                  rng.integers(1, 2, m), np.zeros(m),
                  rng.normal(3.5, 1.0, m), rng.uniform(0.3, 1.2, m))


def _portscan(m, rng):
    proto = rng.choice(PROTOCOLS, m, p=[0.9, 0.1, 0, 0])
    tcp = _is_tcp(proto)
    ps = rng.integers(20, 600, m)
    pd_ = ps * rng.uniform(0.2, 0.9, m)
    return _frame(proto, rng.lognormal(0.5, 0.7, m), ps, pd_,
                  ps * rng.lognormal(4.0, 0.2, m), pd_ * 40,
                  tcp * ps * rng.uniform(0.8, 1, m), np.zeros(m),
                  tcp * pd_ * rng.uniform(0.5, 1, m), np.zeros(m),
                  rng.integers(15, 800, m), rng.integers(1, 4, m), np.zeros(m),
                  rng.normal(2.5, 0.6, m), rng.uniform(0.5, 1.5, m))


def _botnet(m, rng):
    proto = rng.choice(PROTOCOLS, m, p=[0.6, 0.15, 0, 0.25])
    tcp = _is_tcp(proto)
    ps, pd_ = rng.integers(3, 30, m), rng.integers(1, 20, m)
    return _frame(proto, rng.lognormal(1.5, 0.7, m), ps, pd_,
                  ps * rng.lognormal(4.5, 0.4, m), pd_ * rng.lognormal(4.5, 0.4, m),
                  tcp * rng.integers(1, 4, m), tcp * pd_, np.zeros(m),
                  tcp * rng.integers(0, 2, m), rng.integers(1, 4, m),
                  rng.integers(10, 250, m),
                  (rng.random(m) < 0.5) * rng.integers(0, 15, m),
                  rng.normal(7.2, 0.4, m), rng.uniform(0.02, 0.12, m))


def _bruteforce(m, rng):
    proto = rng.choice(PROTOCOLS, m, p=[0.85, 0, 0, 0.15])
    tcp = _is_tcp(proto)
    ps, pd_ = rng.integers(40, 400, m), rng.integers(40, 400, m)
    return _frame(proto, rng.lognormal(2.2, 0.6, m), ps, pd_,
                  ps * rng.lognormal(4.6, 0.4, m), pd_ * rng.lognormal(4.6, 0.4, m),
                  tcp * rng.integers(5, 60, m), tcp * pd_, tcp * rng.integers(0, 10, m),
                  tcp * rng.integers(0, 5, m), rng.integers(1, 3, m), np.ones(m),
                  rng.integers(10, 250, m), rng.normal(5.5, 0.6, m),
                  rng.uniform(0.5, 1.5, m))


_GENERATORS = {"Normal": _normal, "DDoS": _ddos, "Port Scan": _portscan,
               "Botnet": _botnet, "Brute Force": _bruteforce}


def generate_flows(n, rng=None, attack_ratio=0.3, balanced=False, stealth=0.04):
    """Generate n IoT network flows. Returns a DataFrame with a `true_label` column.

    `stealth` is the share of attack flows that look almost normal, which keeps
    the problem realistic (no model should reach 100 %).
    """
    rng = rng if rng is not None else np.random.default_rng()
    attack_ratio = 0.6 if balanced else attack_ratio
    p = [1 - attack_ratio] + [attack_ratio / 4] * 4
    labels = rng.choice(CLASSES, n, p=p)
    parts = []
    for cls in CLASSES:
        idx = np.where(labels == cls)[0]
        if len(idx) == 0:
            continue
        chunk = _GENERATORS[cls](len(idx), rng)
        if cls != "Normal":
            mask = rng.random(len(idx)) < stealth
            if mask.any():
                chunk.loc[mask, :] = _normal(int(mask.sum()), rng).values
        chunk.index = idx
        parts.append(chunk)
    df = pd.concat(parts).sort_index().reset_index(drop=True)
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c])
    df["true_label"] = labels
    return df


# --------------------------------------------------------------------------
# 2. Pre-processing (cleaning, encoding, normalising)
# --------------------------------------------------------------------------
def validate_frame(df: pd.DataFrame):
    """Check an uploaded flow table. Returns (clean_df, list_of_messages)."""
    msgs = []
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(missing))
    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=NUMERIC + ["protocol"]).reset_index(drop=True)
    if len(df) < before:
        msgs.append(f"Removed {before - len(df)} row(s) with missing or invalid values.")
    df["protocol"] = df["protocol"].astype(str).str.upper().str.strip()
    unknown = sorted(set(df["protocol"]) - set(PROTOCOLS))
    if unknown:
        msgs.append("Unknown protocol(s) treated as 'other': " + ", ".join(unknown))
    return df, msgs


def to_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df[NUMERIC].astype(float).clip(lower=0).copy()
    for c in NUMERIC:
        if c != "payload_entropy":
            X[c] = np.log1p(X[c])                      # tame heavy-tailed features
    proto = df["protocol"].astype(str).str.upper()
    for p in PROTOCOLS:
        X["proto_" + p] = (proto == p).astype(float)  # one-hot encoding
    return X


def transform(df: pd.DataFrame, scaler) -> np.ndarray:
    X = scaler.transform(to_matrix(df).values).astype("float32")
    return X[..., None]                                # (n, features, 1) for Conv1D


# --------------------------------------------------------------------------
# 3. Detector
# --------------------------------------------------------------------------
class Detector:
    def __init__(self):
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        from tensorflow import keras  # imported lazily so the app starts quickly
        self.model = keras.models.load_model(MODEL_DIR / "ids_model.keras", compile=False)
        self.scaler = joblib.load(MODEL_DIR / "scaler.joblib")
        with open(MODEL_DIR / "metrics.json") as f:
            self.metrics = json.load(f)
        self.predict(generate_flows(4, np.random.default_rng(0)))  # warm-up

    def predict(self, df: pd.DataFrame):
        """Classify flows. Returns (result_df, total_ms)."""
        t0 = time.perf_counter()
        probs = self.model(transform(df, self.scaler), training=False).numpy()
        total_ms = (time.perf_counter() - t0) * 1000
        idx = probs.argmax(1)
        out = df.copy()
        out["prediction"] = [CLASSES[i] for i in idx]
        out["confidence"] = probs.max(1)
        out["intrusion_prob"] = 1 - probs[:, 0]
        out["is_intrusion"] = idx != 0
        out["latency_ms"] = total_ms / max(len(df), 1)
        return out, total_ms


# --------------------------------------------------------------------------
# 4. Storage (SQLite). Note: on Streamlit Community Cloud the file system is
#    temporary, so history resets whenever the app restarts.
# --------------------------------------------------------------------------
def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH, timeout=10)


def init_db():
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT, source TEXT, protocol TEXT, label TEXT,
            confidence REAL, is_intrusion INTEGER, latency_ms REAL,
            status TEXT, note TEXT DEFAULT '',
            pkt_rate REAL, unique_dst_ports INTEGER, auth_attempts INTEGER,
            features TEXT)""")


def save_detections(res: pd.DataFrame, source: str):
    ts = datetime.now().isoformat(timespec="seconds")
    feats = res[NUMERIC].round(4).to_dict("records")
    rows = [
        (ts, source, r.protocol, r.prediction, float(r.confidence),
         int(r.is_intrusion), float(r.latency_ms),
         "New" if r.is_intrusion else "Normal", "",
         float(r.pkt_rate), int(r.unique_dst_ports), int(r.auth_attempts),
         json.dumps(f))
        for r, f in zip(res.itertuples(index=False), feats)
    ]
    with _conn() as c:
        c.executemany(
            """INSERT INTO detections (ts, source, protocol, label, confidence,
               is_intrusion, latency_ms, status, note, pkt_rate,
               unique_dst_ports, auth_attempts, features)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows)


def fetch(only_alerts=False, labels=None, statuses=None, source=None, limit=1000):
    sql, params = "SELECT * FROM detections WHERE 1=1", []
    if only_alerts:
        sql += " AND is_intrusion = 1"
    if labels:
        sql += f" AND label IN ({','.join('?' * len(labels))})"
        params += list(labels)
    if statuses:
        sql += f" AND status IN ({','.join('?' * len(statuses))})"
        params += list(statuses)
    if source:
        sql += " AND source = ?"
        params.append(source)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(int(limit))
    with _conn() as c:
        return pd.read_sql_query(sql, c, params=params)


def update_status(ids, status, note=""):
    if not len(ids):
        return
    with _conn() as c:
        c.executemany("UPDATE detections SET status=?, note=? WHERE id=?",
                      [(status, note, int(i)) for i in ids])


def stats():
    with _conn() as c:
        t = c.execute("SELECT COUNT(*), COALESCE(SUM(is_intrusion),0), "
                      "COALESCE(AVG(latency_ms),0) FROM detections").fetchone()
        by = pd.read_sql_query(
            "SELECT label, COUNT(*) AS flows FROM detections GROUP BY label", c)
        tl = pd.read_sql_query(
            "SELECT substr(ts,1,16) AS minute, "
            "SUM(is_intrusion) AS intrusions, SUM(1-is_intrusion) AS normal "
            "FROM detections GROUP BY minute ORDER BY minute DESC LIMIT 60", c)
    return {"total": t[0], "intrusions": int(t[1]), "avg_latency": t[2],
            "by_label": by, "timeline": tl.iloc[::-1]}


def clear_history():
    with _conn() as c:
        c.execute("DELETE FROM detections")


# --------------------------------------------------------------------------
# 5. Alerts and login
# --------------------------------------------------------------------------
def check_login(user, password, real_user, real_password):
    ok_u = hmac.compare_digest(str(user).encode(), str(real_user).encode())
    ok_p = hmac.compare_digest(str(password).encode(), str(real_password).encode())
    return ok_u and ok_p


def channels_configured(cfg: dict):
    ch = []
    if cfg.get("SMTP_HOST") and cfg.get("SMTP_USER") and cfg.get("ALERT_TO"):
        ch.append("Email")
    if cfg.get("TELEGRAM_BOT_TOKEN") and cfg.get("TELEGRAM_CHAT_ID"):
        ch.append("Telegram")
    return ch


def format_alert(intr: pd.DataFrame) -> str:
    counts = intr["prediction"].value_counts().to_dict()
    detail = ", ".join(f"{k}: {v}" for k, v in counts.items())
    return (f"IoT IDS ALERT [{datetime.now():%Y-%m-%d %H:%M:%S}]\n"
            f"{len(intr)} intrusive flow(s) detected. {detail}.\n"
            f"Highest confidence: {intr['confidence'].max():.1%}. "
            f"Open the dashboard to review and act.")


def notify(message: str, cfg: dict):
    """Send the alert on every configured channel. Returns {channel: 'sent'|error}."""
    out = {}
    if "Email" in channels_configured(cfg):
        try:
            msg = EmailMessage()
            msg["Subject"] = "IoT IDS alert: intrusion detected"
            msg["From"], msg["To"] = cfg["SMTP_USER"], cfg["ALERT_TO"]
            msg.set_content(message)
            with smtplib.SMTP(cfg["SMTP_HOST"], int(cfg.get("SMTP_PORT") or 587), timeout=10) as s:
                s.starttls()
                s.login(cfg["SMTP_USER"], cfg.get("SMTP_PASSWORD", ""))
                s.send_message(msg)
            out["Email"] = "sent"
        except Exception as e:  # noqa: BLE001
            out["Email"] = f"failed: {e}"
    if "Telegram" in channels_configured(cfg):
        try:
            url = f"https://api.telegram.org/bot{cfg['TELEGRAM_BOT_TOKEN']}/sendMessage"
            data = urllib.parse.urlencode(
                {"chat_id": cfg["TELEGRAM_CHAT_ID"], "text": message}).encode()
            urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=10)
            out["Telegram"] = "sent"
        except Exception as e:  # noqa: BLE001
            out["Telegram"] = f"failed: {e}"
    return out
