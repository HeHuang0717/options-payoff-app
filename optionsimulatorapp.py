# 花里胡哨
import io
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

st.set_page_config(page_title="Gambling Simulator", layout="wide")

# =========================
# Simulation (vectorized)
# =========================
def simulate(
    initial_capital: float,
    odds: float,
    win_prob: float,
    bet_fraction: float,
    n_bets: int,
    n_sims: int,
    seed: None,
):
    """
    Proportional betting:
    - Each round's bet = bet_fraction * current capital (dynamic / compounding)
    - Win: capital += bet * (odds - 1)
      Lose: capital -= bet
    - No negative capital; once capital hits 0, it stays 0 (stop betting)
    Returns:
        all_paths: (n_sims, n_bets+1)
        final_capitals: (n_sims,)
    """
    rng = np.random.default_rng(seed)

    # Pre-allocate
    all_paths = np.empty((n_sims, n_bets + 1), dtype=float)
    all_paths[:, 0] = initial_capital

    capital = np.full(n_sims, initial_capital, dtype=float)
    alive = capital > 0

    for t in range(1, n_bets + 1):
        # Bet is a fraction of CURRENT capital (only for alive sims)
        bet_amt = np.zeros_like(capital)
        bet_amt[alive] = capital[alive] * bet_fraction

        # Random outcomes
        outcomes = rng.random(n_sims) < win_prob  # True=win, False=loss

        # Apply PnL to alive sims
        win_mask  = alive & outcomes
        lose_mask = alive & (~outcomes)

        capital[win_mask]  += bet_amt[win_mask]  * (odds - 1.0)
        capital[lose_mask] -= bet_amt[lose_mask]

        # Clip to 0; once zero, it stays zero (no more betting)
        np.maximum(capital, 0.0, out=capital)
        alive = capital > 0

        all_paths[:, t] = capital

    final_capitals = all_paths[:, -1].copy()
    return all_paths, final_capitals


# =========================
# Plot: all paths (fast)
# =========================
def plot_all_paths(all_paths, initial_capital=100, log_y=False):
    # 1. Show 10 random sample equity curves
    n_simulations = len(all_paths)
    fig = plt.figure(figsize=(10, 6))
    for i in np.random.choice(n_simulations, 100, replace=False):
        plt.plot(all_paths[i], alpha=0.7)
    plt.axhline(initial_capital, color='gray', linestyle='--', label="Initial Capital")

    plt.title("Sample Equity Curves")
    plt.xlabel("Number of Bets")
    plt.ylabel("Capital")
    plt.legend()
    if log_y:
        plt.yscale('log')

    return fig


# =========================
# Heatmap helpers
# =========================
def _heatmap_matrix(values, n_bins=200, normalize="column", log_scale=False):
    """
    values: (n_sims, T) matrix; bin along rows for each time column.
    Returns:
        H (n_bins, T), (vmin, vmax)
    """
    T = values.shape[1]
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    bin_edges = np.linspace(vmin, vmax, n_bins + 1)
    H = np.zeros((n_bins, T), dtype=float)
    for t in range(T):
        counts, _ = np.histogram(values[:, t], bins=bin_edges)
        H[:, t] = counts

    if normalize == "column":
        col_sums = H.sum(axis=0, keepdims=True)
        np.divide(H, col_sums, out=H, where=(col_sums > 0))

    if log_scale:
        H = np.log1p(H)

    return H, (vmin, vmax)


def plot_equity_heatmap(all_paths, n_bins=200, normalize='column', log_scale=False,
                        initial_capital=100, capital_log=False):
    # If log-transform capital: use log1p(capital) to handle zeros gracefully
    if capital_log:
        ys = np.log1p(all_paths)  # transform the variable itself
    else:
        ys = all_paths

    H, (ymin, ymax) = _heatmap_matrix(ys, n_bins=n_bins, normalize=normalize, log_scale=log_scale)
    T = ys.shape[1]
    x = np.arange(T)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    im = ax.imshow(H, origin="lower", aspect="auto", extent=(0, T-1, ymin, ymax))
    cbar = fig.colorbar(im, ax=ax, pad=0.015)
    cbar.set_label("Density" if normalize == "column" else "Counts")

    # Quantiles computed in the same space as the heatmap
    for q, lw in zip((10, 50, 90), (1.2, 1.8, 1.2)):
        ax.plot(x, np.percentile(ys, q, axis=0), linewidth=lw, label=f"{q}th pct")

    # Initial capital line (also transform if needed)
    init_line = np.log1p(initial_capital) if capital_log else initial_capital
    ax.axhline(init_line, linestyle="--", linewidth=1, alpha=0.7, label="Initial capital")

    ax.set_title("Equity Distribution Heatmap" + (" (log1p capital)" if capital_log else ""))
    ax.set_xlabel("Bet index")
    ax.set_ylabel("log1p(Capital)" if capital_log else "Capital")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, alpha=0.2, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig



def plot_final_hist(final_capitals, capital_log=False):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    # final_capitals = np.log1p(final_capitals) if capital_log else final_capitals
    ax.hist(final_capitals, bins=30, edgecolor="black", alpha=0.7, density=True)
    ax.axvline(final_capitals.mean(), linestyle="--", linewidth=1.2,
               label=f"Mean: {final_capitals.mean():.2f}")

    ax.set_title("Final Capital Distribution")
    ax.set_xlabel("Final Capital")
    ax.set_ylabel("Frequency(Normal)")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, alpha=0.2, linewidth=0.6)
    if capital_log:
        ax.set_xscale('log')
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


# =========================
# Sidebar controls
# =========================
st.sidebar.header("Parameters")
initial_capital = st.sidebar.number_input("Initial capital", min_value=1.0, value=1.0, step=1.0)
odds = st.sidebar.number_input("Odds (multiplier)", min_value=1.0, value=2.0, step=0.1,
                               help="Win adds (odds-1)*bet_amount; loss subtracts bet_amount")
win_prob = st.sidebar.number_input("Win probability", min_value=0.0, max_value=1.0, value=0.55, step=0.01)
bet_fraction = st.sidebar.number_input("Bet fraction ", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
n_bets = st.sidebar.number_input("Number of bets per simulation", min_value=1, max_value=100, value=50, step=5)
n_sims = st.sidebar.number_input("Number of simulations", min_value=1, value=2000, step=500)
seed_text = st.sidebar.text_input("Random seed (optional)", value="")
seed = int(seed_text) if seed_text.strip().isdigit() else None

st.sidebar.markdown("---")
n_bins = st.sidebar.slider("Heatmap bins", min_value=20, max_value=300, value=20, step=10)
normalize_opt = st.sidebar.selectbox("Heatmap normalization", options=["column", "none"], index=0)
log_scale = st.sidebar.checkbox("Log-scale heatmap", value=True)

st.sidebar.markdown("---")

log_capital = st.sidebar.checkbox(
    "Log-transform capital (plots)",
    value=True,
    help="If ON: line plot uses log y-scale (zeros clipped to ε); equity heatmap bins on log1p(capital)."
)
max_lines = st.sidebar.slider("Max paths drawn in line plot", min_value=50, max_value=5000, value=300, step=50)
st.sidebar.caption("Drawing too many lines can be slow. Heatmap is better for very large simulations.")


# =========================
# Title & description
# =========================
st.title("🎲 OPTION Simulator")
st.caption("Capital never goes below 0. Once 0, the path stays at 0 (stop betting). "
           "Bet size = bet_fraction × current capital (proportional / compounding).")
# =========================
# Cache & run simulation
# =========================
@st.cache_data(show_spinner=True)
def run_cached(ic, od, wp, bf, nb, ns, sd):
    return simulate(ic, od, wp, bf, nb, ns, sd)

all_paths, final_capitals = run_cached(initial_capital, odds, win_prob, bet_fraction, n_bets, n_sims, seed)

# =========================
# KPIs
# =========================
mean_capital = float(np.mean(final_capitals))
std_capital = float(np.std(final_capitals))
median_capital = float(np.median(final_capitals))
profit_prob = float(np.mean(final_capitals > initial_capital))
loss_prob = float(np.mean(final_capitals < initial_capital))
zero_prob = float(np.mean(final_capitals == 0.0))
min_cap = float(final_capitals.min())
max_cap = float(final_capitals.max())



r1 = st.columns(3)
r2 = st.columns(4)
widgets = [*r1, *r2]  # 共 7 个位置

# k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
widgets[0].metric("Expected final capital", f"{mean_capital:.2f}")
widgets[1].metric("Std dev", f"{std_capital:.2f}")
widgets[2].metric("Median", f"{median_capital:.2f}")
widgets[3].metric("P(Profit)", f"{100*profit_prob:.2f}%")
widgets[4].metric("P(Loss)", f"{100*loss_prob:.2f}%")
widgets[5].metric("P(Bankruptcy=0)", f"{100*zero_prob:.2f}%")
widgets[6].metric("Min/Max", f"{min_cap:.0f} / {max_cap:.0f}")

# =========================
# Plots
# =========================
# 1) All paths (subset for speed)
paths_to_draw = min(max_lines, all_paths.shape[0])
fig_paths = plot_all_paths(all_paths[:paths_to_draw], initial_capital=initial_capital
                           , log_y=log_capital)
st.pyplot(fig_paths, use_container_width=True)

# 2) Heatmap + 3) Final histogram
c1, c2 = st.columns(2)
with c1:
    fig_eq = plot_equity_heatmap(
        all_paths,
        n_bins=n_bins,
        normalize='column' if normalize_opt == "column" else 'none',
        log_scale=log_scale,  # log counts (optional)
        initial_capital=initial_capital,
        capital_log=log_capital  # NEW: log-transform capital itself
    )
    st.pyplot(fig_eq, use_container_width=True)

with c2:
    fig_hist = plot_final_hist(final_capitals, capital_log=log_capital)
    st.pyplot(fig_hist, use_container_width=True)

# =========================
# Downloads
# =========================
st.subheader("Download")
st.caption("Download simulated results.")
buf_paths = io.StringIO()
pd.DataFrame(all_paths).to_csv(buf_paths, index=False)
st.download_button("Download all paths (CSV)", buf_paths.getvalue(), file_name="simulated_paths.csv", mime="text/csv")

buf_final = io.StringIO()
pd.DataFrame({"final_capital": final_capitals}).to_csv(buf_final, index=False)
st.download_button("Download final capitals (CSV)", buf_final.getvalue(), file_name="final_capitals.csv", mime="text/csv")

with st.expander("Modeling assumptions"):
    st.markdown("""
- Fixed bet size per round: **bet_fraction × initial_capital**.
- **No negative capital**; once capital is 0, it stays 0 (no further bets).
- Independent Bernoulli outcomes with probability `win_prob`.
- Win adds `(odds - 1) × bet_amount`; loss subtracts `bet_amount`.
""")
