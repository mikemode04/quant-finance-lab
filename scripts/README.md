# 📚 Quant Finance Lab — Python + SQL Research Stack

End‑to‑end toolkit for **quantitative finance** built around **SQLite + Python**.
It ingests market and macro data (Yahoo, FRED, FX), structures it with SQL schemas/views,
and runs analyses typical for quant & FICC: **betas, correlations, factor models,
macro regressions, regimes, event studies, and simple backtests**.

> Built during my studies at **UiO** (econometrics/time series + SQL) and extended for
> my own macro/FX/equity research. Designed to mirror real quant workflows.

---

## ✨ Highlights
- **Data ingest**: Yahoo Finance (equities/ETFs), FRED (macro), FX (DEX* series)
- **Database**: portable **SQLite** (`data/market.db`) with clean types & indices
- **SQL views**: daily/monthly returns; FRED monthly aggregates
- **Analytics**:
  - Betas (daily/weekly), correlation matrices, top volume
  - Macro regressions (monthly returns ~ Δrates + inflation + Δunemployment + ΔFedFunds)
  - **Regimes** (term spread 10Y–3M; inverted/neutral/steep), quintile buckets
  - (Planned) **Factor models** (Fama‑French/Carhart), **Event studies** (FOMC), **FX links**
- **Reproducibility**: scripts + SQL + optional notebooks

---

## 🧱 Project layout
