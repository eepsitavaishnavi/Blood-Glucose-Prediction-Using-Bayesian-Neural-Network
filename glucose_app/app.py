"""
Bayesian BiGRU Blood Glucose Prediction — Flask Web App
Matches exactly the notebook's model architecture and feature pipeline.
"""

from flask import Flask, request, jsonify, render_template
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import os

app = Flask(__name__)

# ── Config (must match notebook exactly) ──────────────────────
SEQ_LEN  = 24
HIDDEN   = 50
N_LAYERS = 3
DROPOUT  = 0.3
T_MC     = 50     # Monte Carlo passes
DEVICE   = 'cpu'

FEATURE_COLS = [
    'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
    'roll_mean_6', 'roll_std_6', 'roll_mean_12',
    'roll_max_6', 'roll_min_6',
    'diff_1', 'diff_6',
    'hour', 'dow'
]
INPUT_SIZE = len(FEATURE_COLS)  # 14


# ── Exact model architecture from notebook ─────────────────────
class BayesianLinear(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.w_mu  = nn.Parameter(torch.zeros(out_f, in_f))
        self.w_rho = nn.Parameter(torch.full((out_f, in_f), -3.0))
        self.b_mu  = nn.Parameter(torch.zeros(out_f))
        self.b_rho = nn.Parameter(torch.full((out_f,), -3.0))
        nn.init.xavier_uniform_(self.w_mu)

    def forward(self, x):
        w_sigma = F.softplus(self.w_rho)
        b_sigma = F.softplus(self.b_rho)
        w = self.w_mu + w_sigma * torch.randn_like(w_sigma)
        b = self.b_mu + b_sigma * torch.randn_like(b_sigma)
        return F.linear(x, w, b)

    def kl_loss(self):
        def kl(mu, rho):
            sigma = F.softplus(rho)
            return -0.5 * torch.sum(1 + 2*torch.log(sigma + 1e-8) - mu**2 - sigma**2)
        return kl(self.w_mu, self.w_rho) + kl(self.b_mu, self.b_rho)


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.project = nn.Linear(2*hidden_dim, 2*hidden_dim)
        self.context = nn.Linear(2*hidden_dim, 1, bias=False)

    def forward(self, gru_out):
        energy  = torch.tanh(self.project(gru_out))
        weights = torch.softmax(self.context(energy), dim=1)
        context = (weights * gru_out).sum(dim=1)
        return context, weights.squeeze(-1)


class PaperBiGRU(nn.Module):
    def __init__(self, input_size, hidden_dim=HIDDEN,
                 num_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.bigru     = nn.GRU(input_size, hidden_dim, num_layers=num_layers,
                                batch_first=True, bidirectional=True, dropout=dropout)
        self.attention = Attention(hidden_dim)
        self.dropout   = nn.Dropout(dropout)
        self.bay_mean  = BayesianLinear(2*hidden_dim, 1)
        self.bay_lv    = BayesianLinear(2*hidden_dim, 1)

    def forward(self, x):
        gru_out, _  = self.bigru(x)
        ctx, attn_w = self.attention(gru_out)
        ctx         = self.dropout(ctx)
        mean        = self.bay_mean(ctx).squeeze(-1)
        log_var     = self.bay_lv(ctx).squeeze(-1)
        return mean, log_var

    def kl_loss(self):
        return self.bay_mean.kl_loss() + self.bay_lv.kl_loss()

    def enable_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()


# ── Load model and scalers ────────────────────────────────────
model = PaperBiGRU(INPUT_SIZE).to(DEVICE)
MODEL_PATH   = 'best_bigru_model.pt'
SCALERX_PATH = 'scaler_X.pkl'
SCALERY_PATH = 'scaler_y.pkl'

model_loaded   = False
scalers_loaded = False

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    model_loaded = True
    print(f"✅ Model loaded from {MODEL_PATH}")
else:
    print(f"⚠️  Model file not found: {MODEL_PATH}")

scaler_X = scaler_y = None
if os.path.exists(SCALERX_PATH) and os.path.exists(SCALERY_PATH):
    scaler_X = joblib.load(SCALERX_PATH)
    scaler_y = joblib.load(SCALERY_PATH)
    scalers_loaded = True
    print(f"✅ Scalers loaded")
else:
    print(f"⚠️  Scalers not found — save them from notebook first (see README)")


def build_features_from_history(glucose_history: list, hour: int, dow: int) -> np.ndarray:
    """
    Build the 14-feature vector from the last 24 glucose readings.
    glucose_history: list of floats, most recent reading LAST, length >= 24
    Returns shape (24, 14) — one feature vector per time step.
    """
    g = np.array(glucose_history[-24:], dtype=np.float32)  # take last 24 only

    rows = []
    for i in range(len(g)):
        # Lag features (use 0 if not enough history)
        lag_1  = g[i-1]  if i >= 1  else g[0]
        lag_2  = g[i-2]  if i >= 2  else g[0]
        lag_3  = g[i-3]  if i >= 3  else g[0]
        lag_6  = g[i-6]  if i >= 6  else g[0]
        lag_12 = g[i-12] if i >= 12 else g[0]

        # Rolling stats
        window_6  = g[max(0, i-5):i+1]
        window_12 = g[max(0, i-11):i+1]
        roll_mean_6  = window_6.mean()
        roll_std_6   = window_6.std() if len(window_6) > 1 else 0.0
        roll_mean_12 = window_12.mean()
        roll_max_6   = window_6.max()
        roll_min_6   = window_6.min()

        # Differences
        diff_1 = g[i] - g[i-1] if i >= 1 else 0.0
        diff_6 = g[i] - g[i-6] if i >= 6 else 0.0

        rows.append([
            lag_1, lag_2, lag_3, lag_6, lag_12,
            roll_mean_6, roll_std_6, roll_mean_12,
            roll_max_6, roll_min_6,
            diff_1, diff_6,
            float(hour), float(dow)
        ])

    return np.array(rows, dtype=np.float32)  # (24, 14)


def mc_predict_single(sequence_scaled: np.ndarray) -> dict:
    """
    Run T_MC stochastic forward passes on one (1, 24, 14) input.
    Returns mean prediction + epistemic + aleatoric + total uncertainty.
    """
    x = torch.tensor(sequence_scaled[np.newaxis], dtype=torch.float32).to(DEVICE)

    model.eval()
    model.enable_dropout()

    means, variances = [], []
    with torch.no_grad():
        for _ in range(T_MC):
            m, lv = model(x)
            means.append(m.item())
            variances.append(torch.exp(lv).item())

    means      = np.array(means)
    variances  = np.array(variances)

    pred_mean_sc = means.mean()
    epistemic_sc = means.var()
    aleatoric_sc = variances.mean()
    total_var_sc = epistemic_sc + aleatoric_sc

    # Inverse-transform to mg/dL
    sigma_y = scaler_y.scale_[0]
    pred_mg  = scaler_y.inverse_transform([[pred_mean_sc]])[0][0]
    epist_mg = float(np.sqrt(abs(epistemic_sc))) * sigma_y
    aleat_mg = float(np.sqrt(abs(aleatoric_sc))) * sigma_y
    total_mg = float(np.sqrt(abs(total_var_sc)))  * sigma_y

    # 95% credible interval
    ci_lo = float(pred_mg - 1.96 * total_mg)
    ci_hi = float(pred_mg + 1.96 * total_mg)

    # LINEX-adjusted prediction (a=+0.05, penalise overestimation)
    linex_pred = float(pred_mg - (0.05/2) * total_mg**2)

    # Glycaemic zone
    if pred_mg < 70:
        zone = "hypoglycaemia"
    elif pred_mg > 180:
        zone = "hyperglycaemia"
    else:
        zone = "normal"

    return {
        "prediction_mg":   round(float(pred_mg), 1),
        "epistemic_std":   round(epist_mg, 2),
        "aleatoric_std":   round(aleat_mg, 2),
        "total_std":       round(total_mg, 2),
        "ci_lower":        round(ci_lo, 1),
        "ci_upper":        round(ci_hi, 1),
        "linex_adjusted":  round(linex_pred, 1),
        "zone":            zone,
        "mc_passes":       T_MC,
        "mc_means":        [round(float(m), 2) for m in means],
    }


@app.route('/')
def index():
    return render_template('index.html',
                           model_loaded=model_loaded,
                           scalers_loaded=scalers_loaded)


@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded or not scalers_loaded:
        return jsonify({"error": "Model or scalers not loaded. See setup instructions."}), 500

    data = request.get_json()
    glucose_history = data.get('glucose_history', [])
    hour = int(data.get('hour', 12))
    dow  = int(data.get('dow', 0))

    if len(glucose_history) < 24:
        return jsonify({"error": f"Need at least 24 glucose readings. Got {len(glucose_history)}."}), 400

    try:
        # Build feature matrix (24, 14)
        features = build_features_from_history(glucose_history, hour, dow)

        # Scale using the same scaler fitted in notebook
        features_scaled = scaler_X.transform(features)

        # Run MC Dropout inference
        result = mc_predict_single(features_scaled)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health')
def health():
    return jsonify({
        "model_loaded":   model_loaded,
        "scalers_loaded": scalers_loaded,
        "mc_passes":      T_MC,
        "input_features": FEATURE_COLS,
        "seq_len":        SEQ_LEN
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
