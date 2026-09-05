FILE_PATH = "sample_chat.txt"      # WhatsApp .txt export OR Telegram result.json
FORMAT = "whatsapp"                 # "whatsapp" or "telegram"
YOUR_NAME = "You"                   # exactly how your name appears in the export
 
# ============================================================
# STEP 1: PARSERS
# ============================================================
import re
import json
import pandas as pd
import numpy as np
from datetime import datetime
 
 
def parse_whatsapp(path):
    """
    Handles the common WhatsApp export line formats:
      12/31/23, 11:59 PM - Alex: message text
      [12/31/23, 11:59:59 PM] Alex: message text
      31/12/23, 23:59 - Alex: message text   (24h / DD-MM-YY locales)
    Multi-line messages (user pressed Enter) are appended to the previous message.
    If your export's date format doesn't match, adjust DATE_FORMATS below.
    """
    line_pattern = re.compile(
        r'^\[?(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?\s?(?:[AaPp][Mm])?)\]?\s*-?\s*([^:]+):\s(.*)$'
    )
    date_formats = [
        "%m/%d/%y %I:%M %p", "%m/%d/%Y %I:%M %p",
        "%d/%m/%y %H:%M", "%d/%m/%Y %H:%M",
        "%m/%d/%y %H:%M", "%d/%m/%y %I:%M %p",
    ]
 
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            m = line_pattern.match(line)
            if m:
                date_str, time_str, sender, message = m.groups()
                dt = None
                for fmt in date_formats:
                    try:
                        dt = datetime.strptime(f"{date_str} {time_str}".replace("\u202f", " "), fmt)
                        break
                    except ValueError:
                        continue
                if dt is None:
                    continue  # couldn't parse timestamp; skip line rather than corrupt the series
                rows.append({"timestamp": dt, "sender": sender.strip(), "message": message.strip()})
            elif rows:
                # continuation of previous multi-line message
                rows[-1]["message"] += " " + line.strip()
 
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(
            "No messages parsed. Your export's date/time format probably doesn't match "
            "the regex — open the .txt file, check one line's format, and adjust "
            "line_pattern / date_formats in parse_whatsapp()."
        )
    return df.sort_values("timestamp").reset_index(drop=True)
 
 
def parse_telegram(path):
    """Handles Telegram Desktop's result.json export."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
 
    rows = []
    for msg in data.get("messages", []):
        if msg.get("type") != "message":
            continue
        text = msg.get("text", "")
        if isinstance(text, list):
            # Telegram splits formatted text into a list of strings/dicts
            text = "".join(t if isinstance(t, str) else t.get("text", "") for t in text)
        if not text:
            continue
        rows.append({
            "timestamp": datetime.fromisoformat(msg["date"]),
            "sender": msg.get("from", "Unknown"),
            "message": text.strip(),
        })
 
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No messages parsed from Telegram export — check the JSON structure.")
    return df.sort_values("timestamp").reset_index(drop=True)
 
 
# ============================================================
# STEP 2: FEATURE EXTRACTION (from real message content)
# ============================================================
EMOJI_PATTERN = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]"
)
 
def compute_features(df):
    df = df.copy()
 
    # response_time_min: minutes since the PREVIOUS message from a DIFFERENT sender.
    # NaN for the very first message and for double-texts (see below).
    prev_sender = df["sender"].shift(1)
    prev_time = df["timestamp"].shift(1)
    is_reply = df["sender"] != prev_sender
    df["response_time_min"] = np.where(
        is_reply,
        (df["timestamp"] - prev_time).dt.total_seconds() / 60,
        np.nan,
    )
 
    # double_texted: this sender also sent the immediately preceding message
    df["double_texted"] = (df["sender"] == prev_sender).astype(int)
 
    df["message_length"] = df["message"].str.len()
    df["Emoji_count"] = df["message"].apply(lambda m: len(EMOJI_PATTERN.findall(m)))
 
    letters = df["message"].str.replace(r"[^A-Za-z]", "", regex=True)
    df["all_caps"] = (
        (letters.str.len() >= 4) & (letters == letters.str.upper())
    ).astype(int)
 
    # NOTE: no 'lefton_read' column — not recoverable from export data.
    return df
 
 
# ============================================================
# STEP 3A: MANUAL LABELING (real supervised learning)
# ============================================================
def label_sample(df, n=40, seed=42):
    """
    Interactively label a random sample of real messages.
    Run this in a real terminal (not silently) — it will prompt you per message.
    Saves to labeled_messages.csv so you only have to do this once.
    """
    sample = df.dropna(subset=["response_time_min"]).sample(n=min(n, len(df)), random_state=seed).copy()
    labels = []
    print(f"Labeling {len(sample)} messages. Enter 1 = red flag, 0 = fine, s = skip.\n")
    for _, row in sample.iterrows():
        print(f"\n[{row['sender']}] ({row['response_time_min']:.0f} min reply gap) \"{row['message'][:120]}\"")
        ans = input("Label (1/0/s): ").strip().lower()
        labels.append(np.nan if ans == "s" else int(ans) if ans in ("0", "1") else np.nan)
    sample["red_flag"] = labels
    sample = sample.dropna(subset=["red_flag"])
    sample.to_csv("labeled_messages.csv", index=False)
    print(f"\nSaved {len(sample)} labeled rows to labeled_messages.csv")
    return sample
 
 
def train_on_labels(labeled_df, feature_cols):
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
 
    X = labeled_df[feature_cols].fillna(0)
    y = labeled_df["red_flag"].astype(int)
 
    if y.nunique() < 2:
        print("Need both classes (0 and 1) represented in your labels to train. Label a few more.")
        return None
 
    clf = make_pipeline(StandardScaler(), LogisticRegression())
    scores = cross_val_score(clf, X, y, cv=min(5, y.value_counts().min()), scoring="f1")
    print(f"F1 across folds: {scores}")
    print(f"Mean F1: {scores.mean():.3f} +/- {scores.std():.3f}")
    clf.fit(X, y)
    return clf
 
 
# ============================================================
# STEP 3B: UNSUPERVISED — no labels needed
# ============================================================
def flag_outliers(df, feature_cols, contamination=0.1):
    """
    Flags statistically unusual messages (long silences, unusual length/caps/emoji
    patterns) relative to the rest of the conversation. Not a moral judgment —
    just "this message is an outlier in this chat's own patterns."
    """
    from sklearn.ensemble import IsolationForest
 
    work = df.dropna(subset=feature_cols).copy()
    iso = IsolationForest(contamination=contamination, random_state=42)
    work["outlier"] = (iso.fit_predict(work[feature_cols]) == -1).astype(int)
    return work.sort_values("response_time_min", ascending=False)
 
 
# ============================================================
# STEP 4: RUN IT
# ============================================================
if __name__ == "__main__":
    if FORMAT == "whatsapp":
        raw = parse_whatsapp(FILE_PATH)
    elif FORMAT == "telegram":
        raw = parse_telegram(FILE_PATH)
    else:
        raise ValueError("FORMAT must be 'whatsapp' or 'telegram'")
 
    print(f"Parsed {len(raw)} messages from {raw['sender'].nunique()} senders.")
    print(raw['sender'].value_counts())
 
    df = compute_features(raw)
    feature_cols = ["response_time_min", "Emoji_count", "message_length", "all_caps", "double_texted"]
 
    print("\n--- Descriptive stats per sender ---")
    print(df.groupby("sender")[["response_time_min", "message_length", "double_texted"]].mean())
 
    print("\n--- Path B: unsupervised outliers (no labels needed) ---")
    outliers = flag_outliers(df, feature_cols)
    print(outliers[outliers["outlier"] == 1][["sender", "message", "response_time_min"]].head(10))
 
    print("\n--- Path A: manual labeling (uncomment to run interactively) ---")
    # labeled = label_sample(df, n=40)
    # model = train_on_labels(labeled, feature_cols)

    
 