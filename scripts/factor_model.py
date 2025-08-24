#!/usr/bin/env python3
"""
Fama-French / Carhart factor regression on monthly returns (SQLite + Python)

Bruk:
  python3 scripts/factor_model.py --db data/market.db --tickers SPY AAPL --carhart --start 2015-01-01

Lagrer:
  - factors_monthly (ym, mkt_rf, smb, hml, rf, [mom])
  - factor_reg_results (per ticker)
"""
import argparse
import sqlite3
import numpy as np
import pandas as pd
import warnings

# Dempe støy fra pandas_datareader (deprecation warnings)
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas_datareader")


def _ym_from_index(idx) -> pd.Index:
    """Robust konvertering av FF-index til 'YYYY-MM' strenger."""
    if isinstance(idx, pd.PeriodIndex):
        return idx.astype(str)
    if isinstance(idx, pd.DatetimeIndex):
        return idx.to_period("M").astype(str)
    return pd.to_datetime(idx, errors="coerce").to_period("M").astype(str)


def load_factors_ff(start_ym="2010-01") -> pd.DataFrame:
    """Hent FF3 (+momentum hvis mulig) via pandas-datareader, normaliser kolonner og ym."""
    from pandas_datareader import data as pdr

    # FF3
    ff3 = pdr.DataReader("F-F_Research_Data_Factors", "famafrench")[0]
    ff3.columns = ff3.columns.astype(str).str.strip()
    ff3["ym"] = _ym_from_index(ff3.index)
    ff3 = (
        ff3.reset_index(drop=True)
        .rename(columns={"Mkt-RF": "mkt_rf", "SMB": "smb", "HML": "hml", "RF": "rf"})
    )

    # Carhart momentum (valgfritt)
    try:
        mom = pdr.DataReader("F-F_Momentum_Factor", "famafrench")[0]
        mom.columns = mom.columns.astype(str).str.strip()  # "Mom   " -> "Mom"
        mom["ym"] = _ym_from_index(mom.index)
        mom = mom.reset_index(drop=True).rename(columns={"Mom": "mom"})
        ff = ff3.merge(mom[["ym", "mom"]], on="ym", how="left")
    except Exception:
        ff = ff3

    # Prosent → desimal (fikset bug: bruk 'col' i filteret)
    for col in [col for col in ff.columns if col != "ym"]:
        ff[col] = pd.to_numeric(ff[col], errors="coerce") / 100.0

    # Filter fra start
    ff = ff.loc[ff["ym"] >= start_ym].dropna(subset=["mkt_rf", "smb", "hml", "rf"])
    return ff


def ensure_factors_in_db(conn: sqlite3.Connection, start_ym="2010-01"):
    """Legg FF-faktorer i SQLite om de ikke finnes fra før."""
    try:
        n = pd.read_sql_query("SELECT COUNT(*) AS n FROM factors_monthly", conn)["n"].iloc[0]
        if n > 0:
            return
    except Exception:
        pass
    ff = load_factors_ff(start_ym)
    ff.to_sql("factors_monthly", conn, if_exists="replace", index=False)
    conn.commit()


def q(conn, sql, params=None):
    return pd.read_sql_query(sql, conn, params=params)


def ols(y, X):
    X1 = np.column_stack([np.ones(len(X)), X])
    coefs, *_ = np.linalg.lstsq(X1, y, rcond=None)
    yhat = X1 @ coefs
    resid = y - yhat
    ss_tot = ((y - y.mean()) ** 2).sum()
    ss_res = (resid ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return coefs, float(r2), len(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/market.db")
    ap.add_argument("--tickers", nargs="+", default=["SPY"])
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--carhart", action="store_true", help="Bruk MOM (4-faktor) hvis tilgjengelig")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)

    # 1) Sikre faktorer i DB
    ensure_factors_in_db(conn, start_ym=args.start[:7])

    # 2) Hent returns + faktorer
    placeholders = ",".join("?" * len(args.tickers))
    rets = q(
        conn,
        f"""
        SELECT ticker, ym, mret
        FROM vw_monthly_returns
        WHERE ym >= ? AND ticker IN ({placeholders})
        ORDER BY ym
        """,
        [args.start[:7]] + args.tickers,
    )
    if rets.empty:
        raise RuntimeError("Fant ingen monthly returns – kjør pipelinen først.")

    fac = q(conn, "SELECT * FROM factors_monthly WHERE ym >= ? ORDER BY ym", [args.start[:7]])
    if fac.empty:
        raise RuntimeError("Fant ingen faktorer i DB.")

    # 3) Resultattabell
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS factor_reg_results (
            ticker TEXT,
            start_ym TEXT,
            end_ym TEXT,
            nobs INTEGER,
            r2 REAL,
            alpha REAL,
            beta_mkt REAL,
            beta_smb REAL,
            beta_hml REAL,
            beta_mom REAL
        )
        """
    )
    conn.commit()

    # 4) OLS per ticker
    for t in sorted(rets.ticker.unique()):
        d = rets[rets.ticker == t].merge(fac, on="ym", how="inner").dropna(subset=["mret", "mkt_rf", "smb", "hml", "rf"])
        if d.empty:
            print(f"[INFO] Skipper {t} (ingen overlapp).")
            continue

        # y = excess return
        y = (d["mret"] - d["rf"]).values

        xcols = ["mkt_rf", "smb", "hml"]
        if args.carhart and "mom" in d.columns:
            xcols.append("mom")

        X = d[xcols].values
        coefs, r2, n = ols(y, X)

        alpha = float(coefs[0])
        beta_mkt = float(coefs[1])
        beta_smb = float(coefs[2])
        beta_hml = float(coefs[3])
        beta_mom = float(coefs[4]) if len(xcols) == 4 else None

        conn.execute(
            """
            INSERT INTO factor_reg_results
            (ticker,start_ym,end_ym,nobs,r2,alpha,beta_mkt,beta_smb,beta_hml,beta_mom)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            [t, d["ym"].min(), d["ym"].max(), n, r2, alpha, beta_mkt, beta_smb, beta_hml, beta_mom],
        )
        conn.commit()

        pretty = f"{t}: alpha={alpha:+.4f}, MKT={beta_mkt:+.2f}, SMB={beta_smb:+.2f}, HML={beta_hml:+.2f}"
        if beta_mom is not None:
            pretty += f", MOM={beta_mom:+.2f}"
        print(pretty)

    conn.close()


if __name__ == "__main__":
    main()
