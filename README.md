# Real-Time Deep Learning Intrusion Detection System for IoT Networks

A Streamlit administrator dashboard backed by a CNN-LSTM model that classifies IoT network flows as
**Normal**, **DDoS**, **Port Scan**, **Botnet** or **Brute Force**.

## Project files
| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard (login, overview, live monitor, upload, alerts, history, evaluation, settings) |
| `ids_core.py` | Data generator, pre-processing, detector, SQLite storage, email/Telegram alerts |
| `train_model.py` | Trains and evaluates the model, writes `model/` |
| `model/` | Trained model, scaler and evaluation metrics (already included) |
| `requirements.txt` | Pinned versions (the ones the app was tested with) |

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
Demo login: `admin` / `admin123`.

## Deploy on Streamlit Community Cloud
1. Create a GitHub repository and upload all files in this folder (keep the `model/` folder).
2. Go to https://share.streamlit.io, sign in with GitHub and click **Create app**.
3. Choose the repository, branch `main`, and main file `app.py`.
4. Open **Advanced settings**, choose **Python 3.12** (3.11 to 3.13 all work), and paste your secrets (see `.streamlit/secrets.toml.example`). At minimum set `ADMIN_USER` and `ADMIN_PASSWORD`.
5. Click **Deploy**. The first build takes a few minutes because TensorFlow is large.

## Retrain
```
python train_model.py                      # synthetic IoT flows
python train_model.py --csv my_flows.csv   # your own labelled flows
```
The CSV needs the columns listed on the app's **Analyse Upload** page plus a `label` column
(Normal, DDoS, Port Scan, Botnet, Brute Force). Public datasets such as CIC-IoT2023 or TON_IoT use
different column names, so map them to this schema first.

## Things to know
- **Traffic is simulated.** Streamlit Cloud cannot capture raw packets, so Live Monitor generates flows.
  Real capture would need a sensor on the network that exports flows to this model.
- **The model was trained on synthetic data**, so its scores show the pipeline works, not performance on
  real networks. Say this plainly in your project report and retrain with a real dataset if your supervisor expects one.
- **History is temporary on Streamlit Cloud.** The SQLite file resets when the app restarts or is redeployed.
  For permanent records, point `ids_core.py` at an external database such as Supabase or PostgreSQL.
- Apps on the free tier sleep after inactivity and wake on the next visit.
# iot_ids
