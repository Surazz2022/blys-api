"""
generate_transactions.py
────────────────────────
Generates data/transactions.csv from data/customer_data.csv.

Perfect sync guarantees
───────────────────────
When you aggregate transactions.csv back per customer, you get EXACTLY:
  • Booking_Frequency  → count of rows           == customer_data.Booking_Frequency
  • Avg_Spending       → mean of Spending         == customer_data.Avg_Spending
  • Preferred_Service  → mode of Service          == customer_data.Preferred_Service
  • Last_Activity      → max of Date              == customer_data.Last_Activity

How each guarantee is enforced
───────────────────────────────
  Booking_Frequency  : generate exactly N rows per customer
  Preferred_Service  : assign floor(N/2)+1 rows to preferred service (always the mode)
  Last_Activity      : last date offset forced to span_days (= Last_Activity)
  Avg_Spending       : generate N-1 spending values freely, compute last value as
                       last = N * Avg_Spending - sum(first N-1), clamped to [40, 3×avg]

Run
───
  python data/generate_transactions.py
Output → data/transactions.csv
"""

import os
import sys
import math
import numpy as np
import pandas as pd
from datetime import timedelta

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INPUT_CSV    = os.path.join(PROJECT_ROOT, "data", "customer_data.csv")
OUTPUT_CSV   = os.path.join(PROJECT_ROOT, "data", "transactions.csv")

# ── Services ──────────────────────────────────────────────────────────────────
SERVICES = ["Massage", "Facial", "Body Scrub", "Hair Spa", "Wellness Package"]

# ── Review text pools ─────────────────────────────────────────────────────────
POSITIVE_REVIEWS = [
    "Amazing session, highly recommend!",
    "Therapist was professional and attentive.",
    "Best massage I've ever had.",
    "Incredibly relaxing, will book again.",
    "Exceeded my expectations.",
    "Wonderful experience from start to finish.",
    "Great value for money.",
    "Very skilled and caring therapist.",
    "Absolutely loved every minute.",
    "Perfect treatment, thank you!",
    "Outstanding service as always.",
    "Felt completely rejuvenated afterwards.",
    "Five stars, no hesitation.",
    "Highly recommend to everyone.",
    "Will definitely return soon.",
    "Brilliant experience, thank you Blys!",
    "Super relaxing and professional.",
    "Loved it, already booked again.",
]

NEUTRAL_REVIEWS = [
    "Good service overall.",
    "Standard treatment, nothing special.",
    "Decent experience.",
    "Fine, met my expectations.",
    "OK session, therapist was on time.",
    "Reasonable quality.",
    "Average but acceptable.",
    "Not bad at all.",
    "Could be better but fine.",
    "Satisfactory service.",
    "Did the job.",
    "Nothing to complain about.",
    "Mediocre but ok.",
    "So-so experience.",
    "Average service, would try again.",
]

NEGATIVE_REVIEWS = [
    "Not satisfied with the service.",
    "Therapist arrived late.",
    "Too expensive for the quality.",
    "Expected much better.",
    "Disappointing experience.",
    "Would not recommend.",
    "Poor service this time.",
    "Not worth the price.",
    "Quite disappointing.",
    "Unprofessional therapist.",
    "Waste of money.",
    "Below average quality.",
    "Major disappointment.",
    "Service quality has declined.",
    "Won't be coming back.",
    "Very unhappy with the visit.",
    "Felt rushed throughout the session.",
]


def _pick_review(sentiment_score: float) -> str:
    if sentiment_score > 0.20:
        return str(np.random.choice(POSITIVE_REVIEWS))
    elif sentiment_score < -0.10:
        return str(np.random.choice(NEGATIVE_REVIEWS))
    else:
        return str(np.random.choice(NEUTRAL_REVIEWS))


def _text_sentiment(text: str) -> float:
    """Keyword heuristic — avoids VADER dependency in generator."""
    t = str(text).lower()
    POSITIVE_KW = {"excellent", "great", "amazing", "love", "best", "wonderful",
                   "highly recommend", "perfect", "outstanding", "brilliant"}
    NEGATIVE_KW = {"not satisfied", "disappointed", "expensive", "poor",
                   "terrible", "bad", "waste", "unprofessional", "would not"}
    if any(kw in t for kw in POSITIVE_KW):
        return  0.5
    if any(kw in t for kw in NEGATIVE_KW):
        return -0.5
    return 0.0


def _build_service_list(preferred: str, freq: int) -> list[str]:
    """
    Build an ordered list of services for this customer's N bookings such that:
      • preferred service appears floor(N/2)+1 times  → always the mode
      • remaining slots filled randomly from other services
    """
    others = [s for s in SERVICES if s != preferred]
    n_preferred = freq // 2 + 1              # minimum count to guarantee mode
    n_preferred = min(n_preferred, freq)     # cannot exceed total bookings
    n_other     = freq - n_preferred

    service_list = [preferred] * n_preferred
    if n_other > 0:
        service_list += list(np.random.choice(others, size=n_other, replace=True))

    np.random.shuffle(service_list)          # randomise order before date assignment
    return service_list


def _build_spending_list(avg_spend: float, freq: int) -> list[float]:
    """
    Generate N spending values whose mean is EXACTLY avg_spend.
      • First N-1 values: avg_spend × Uniform(0.85, 1.15)
      • Last value      : N × avg_spend − sum(first N-1), clamped to [40, 3×avg_spend]
    For freq==1 return [avg_spend] exactly.
    """
    if freq == 1:
        return [round(avg_spend, 2)]

    first = [avg_spend * np.random.uniform(0.85, 1.15) for _ in range(freq - 1)]
    last  = freq * avg_spend - sum(first)
    last  = max(40.0, min(last, avg_spend * 3.0))   # sanity clamp

    spends = first + [last]
    np.random.shuffle(spends)
    return [round(s, 2) for s in spends]


def generate(input_csv: str, output_csv: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    df["Last_Activity"] = pd.to_datetime(df["Last_Activity"])
    df["_sentiment"]    = df["Review_Text"].apply(_text_sentiment)

    rows  = []
    total = len(df)

    for i, customer in df.iterrows():
        if i % 2000 == 0:
            print(f"  Processing customers {i:,}/{total:,} …", flush=True)

        cid       = int(customer["Customer_ID"])
        freq      = max(1, int(customer["Booking_Frequency"]))
        avg_spend = float(customer["Avg_Spending"])
        pref_svc  = str(customer["Preferred_Service"])
        last_date = customer["Last_Activity"]
        sentiment = float(customer["_sentiment"])

        # ── Guaranteed-mode service list ──────────────────────────────────────
        service_list = _build_service_list(pref_svc, freq)

        # ── Exact-mean spending list ──────────────────────────────────────────
        spending_list = _build_spending_list(avg_spend, freq)

        # ── Booking dates: spread back from last_date ─────────────────────────
        span_days  = min(730, max(freq * 30, 60))
        start_date = last_date - timedelta(days=span_days)
        offsets    = sorted(np.random.uniform(0, span_days, freq))
        offsets[-1] = span_days   # last booking == Last_Activity exactly

        for svc, spend, offset in zip(service_list, spending_list, offsets):
            booking_date = start_date + timedelta(days=float(offset))
            rows.append({
                "Customer_ID": cid,
                "Date":        booking_date.strftime("%Y-%m-%d"),
                "Service":     svc,
                "Spending":    spend,
                "Review_Text": _pick_review(sentiment),
            })

    tx = pd.DataFrame(rows)
    tx.sort_values(["Customer_ID", "Date"], inplace=True)
    tx.reset_index(drop=True, inplace=True)
    tx.to_csv(output_csv, index=False)
    return tx


def _verify_sync(input_csv: str, output_csv: str) -> None:
    """Cross-check that aggregated transactions match customer_data.csv exactly."""
    orig = pd.read_csv(input_csv)
    orig["Last_Activity"] = pd.to_datetime(orig["Last_Activity"])
    tx   = pd.read_csv(output_csv)
    tx["Date"] = pd.to_datetime(tx["Date"])

    agg = tx.groupby("Customer_ID").agg(
        freq_check  =("Service",  "count"),
        spend_check =("Spending", "mean"),
        last_check  =("Date",     "max"),
    ).reset_index()
    pref_check = (
        tx.groupby("Customer_ID")["Service"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
        .rename(columns={"Service": "pref_check"})
    )
    agg = agg.merge(pref_check, on="Customer_ID")
    merged = orig.merge(agg, on="Customer_ID")

    freq_ok  = (merged["Booking_Frequency"] == merged["freq_check"]).all()
    spend_ok = (merged["Avg_Spending"] - merged["spend_check"]).abs().max() < 0.02
    pref_ok  = (merged["Preferred_Service"] == merged["pref_check"]).all()
    last_ok  = (
        (merged["Last_Activity"].dt.date - merged["last_check"].dt.date)
        .abs().max().days == 0
    )

    print("\n── Sync verification ─────────────────────────────────────────")
    print(f"  Booking_Frequency  match : {'✅' if freq_ok  else '❌'}")
    print(f"  Avg_Spending       match : {'✅' if spend_ok else '❌'} "
          f"(max drift = ${(merged['Avg_Spending']-merged['spend_check']).abs().max():.4f})")
    print(f"  Preferred_Service  match : {'✅' if pref_ok  else '❌'}")
    print(f"  Last_Activity      match : {'✅' if last_ok  else '❌'}")
    print("──────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    print(f"Reading  : {INPUT_CSV}")
    print(f"Writing  : {OUTPUT_CSV}")
    print()

    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: {INPUT_CSV} not found.", file=sys.stderr)
        sys.exit(1)

    tx = generate(INPUT_CSV, OUTPUT_CSV)

    print()
    print("=" * 55)
    print(f"  Transactions generated : {len(tx):>10,}")
    print(f"  Unique customers       : {tx['Customer_ID'].nunique():>10,}")
    print(f"  Date range             : {tx['Date'].min()} → {tx['Date'].max()}")
    print(f"  Services               : {sorted(tx['Service'].unique())}")
    print(f"  Avg spend / booking    : ${tx['Spending'].mean():.2f}")
    print("=" * 55)

    _verify_sync(INPUT_CSV, OUTPUT_CSV)

    print(f"\nSaved → {OUTPUT_CSV}")
