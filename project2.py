"""
Real-data version of the "red flag" analysis.

IMPORTANT CHANGE FROM THE SYNTHETIC VERSION:
Real chat exports have no ground-truth "red_flag" label. The synthetic script's
91% accuracy meant nothing more than "the model can recover a formula I wrote
myself." With real messages you have two honest options, both implemented here:

  PATH A - Manual labeling: you label a sample of real messages yourself,
           then train a real supervised model on YOUR judgments.
  PATH B - Unsupervised outlier detection: flags statistically unusual
           messages with zero labels required.

Also: WhatsApp/Telegram exports do NOT contain read receipts, so a real
"left on read" feature cannot be derived from the export file alone.
That feature is dropped rather than faked.

=========================
STEP 0: CONFIGURE THIS
=========================
"""

FILE_PATH = "sample_tele_chat.JSON"      # WhatsApp .txt export OR Telegram result.json
FORMAT = "telegram"                 # "whatsapp" or "telegram"
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

    emoji_lists = df["message"].apply(lambda m: EMOJI_PATTERN.findall(m))
    df["Emoji_count"] = emoji_lists.apply(len)

    # max_emoji_run: longest streak of the SAME emoji repeated back-to-back
    # within one message. "😂😂😂😢" -> 3 (not just "4 emojis total").
    df["max_emoji_run"] = emoji_lists.apply(_longest_run)

    # repeats_prev_emoji: this sender used at least one of the same emoji(s)
    # as their own immediately preceding message — catches people who reuse
    # their "signature" emoji message after message, not just within one.
    prev_emojis = pd.Series([set()] + [set(e) for e in emoji_lists[:-1]], index=df.index)
    prev_sender_match = df["sender"] == prev_sender
    df["repeats_prev_emoji"] = [
        int(bool(set(cur) & prev) and same_sender)
        for cur, prev, same_sender in zip(emoji_lists, prev_emojis, prev_sender_match)
    ]

    letters = df["message"].str.replace(r"[^A-Za-z]", "", regex=True)
    df["all_caps"] = (
        (letters.str.len() >= 4) & (letters == letters.str.upper())
    ).astype(int)

    # NOTE: no 'lefton_read' column — not recoverable from export data.
    return df


def _longest_run(emoji_list):
    """Longest streak of the identical emoji appearing consecutively in a list."""
    if not emoji_list:
        return 0
    longest = current = 1
    for i in range(1, len(emoji_list)):
        if emoji_list[i] == emoji_list[i - 1]:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest


# ============================================================
# STEP 2B: ROAST REPORT — per-sender leaderboard + combo archetypes
# ============================================================
def roast_report(df, feature_cols=None, top_n=1):
    """
    Ranks senders on each behavior and calls out the worst offender.
    Thresholds are relative to THIS chat (top quartile), not fixed numbers,
    so it adapts whether it's a sleepy 40-message chat or a chaotic 4000-message one.
    Also flags combo-archetypes: people who stack multiple behaviors at once.
    """
    stats = df.groupby("sender").agg(
        avg_response_min=("response_time_min", "mean"),
        max_response_min=("response_time_min", "max"),
        avg_emoji=("Emoji_count", "mean"),
        total_emoji=("Emoji_count", "sum"),
        avg_emoji_run=("max_emoji_run", "mean"),
        max_emoji_run=("max_emoji_run", "max"),
        emoji_repeat_rate=("repeats_prev_emoji", "mean"),
        avg_length=("message_length", "mean"),
        double_text_rate=("double_texted", "mean"),
        n_messages=("message", "count"),
    ).round(2)

    def zscore(col):
        # more precise than a flat "median * 1.5" cutoff — accounts for how
        # SPREAD OUT the group's behavior is, not just who's on top.
        s = stats[col]
        std = s.std(ddof=0)
        return (s - s.mean()) / std if std > 0 else s * 0

    def crown(col, label, unit="", min_z=0.5):
        z = zscore(col)
        ranked = stats.assign(_z=z).sort_values("_z", ascending=False)
        winner = ranked.index[0]
        val = ranked.iloc[0][col]
        winner_z = ranked.iloc[0]["_z"]
        flagged = winner_z >= min_z  # only crown if they're meaningfully ahead of the group
        print(f"👑 {label}: {winner} ({val}{unit}, z={winner_z:.2f})" + ("" if flagged else "  [no clear outlier this time]"))
        return winner if flagged else None

    print("=" * 50)
    print("ROAST REPORT")
    print("=" * 50)

    seen_king = crown("avg_response_min", "Leaves-on-seen champion", " min avg reply gap")
    yapper = crown("avg_length", "Certified yapper", " chars/msg avg")
    emoji_king = crown("avg_emoji", "Emoji spammer-in-chief", " emojis/msg avg")
    double_king = crown("double_text_rate", "Double-text repeat offender", " rate")
    broken_record = crown("avg_emoji_run", "Broken Record (same emoji, back to back)", " avg run length")

    print()
    print("--- Individual roast lines (with receipts) ---")
    for sender, row in stats.iterrows():
        lines = []
        if sender == seen_king:
            lines.append(f"takes {row['avg_response_min']:.0f} min to reply on average, cooking something?")
        if sender == yapper:
            lines.append(f"writes {row['avg_length']:.0f}-char essays when a 'lol' would do")
        if sender == emoji_king:
            lines.append(f"drops {row['avg_emoji']:.1f} emojis per message, keyboard's on fire")
        if sender == double_king:
            lines.append(f"double-texts {row['double_text_rate']:.0%} of the time, we get it, you're excited")
        if sender == broken_record and row["max_emoji_run"] >= 3:
            # find the actual worst message as evidence
            worst = df[(df["sender"] == sender) & (df["max_emoji_run"] == row["max_emoji_run"])].iloc[0]
            lines.append(
                f"same emoji {int(row['max_emoji_run'])}x in a row — exhibit A: \"{worst['message'][:60]}\""
            )
        if lines:
            print(f"  {sender}: " + "; ".join(lines))

    print()
    print("--- Combo archetypes (stacking multiple behaviors) ---")
    med_len = stats["avg_length"].median()
    med_dt = stats["double_text_rate"].median()
    med_resp = stats["avg_response_min"].median()
    med_emoji = stats["avg_emoji"].median()

    for sender, row in stats.iterrows():
        combo = []
        if row["avg_length"] > med_len and row["double_text_rate"] > med_dt:
            combo.append("🗣️ Yapper-Texter (writes essays AND can't wait for a reply)")
        if row["avg_response_min"] > med_resp and row["avg_emoji"] > med_emoji:
            combo.append("👻 Mixed Signals (leaves you on seen then floods emojis)")
        if row["double_text_rate"] > med_dt and row["avg_emoji"] > med_emoji:
            combo.append("📱 Notification Terrorist (double-texts AND emoji-spams)")
        if row["avg_emoji_run"] > stats["avg_emoji_run"].median() and row["emoji_repeat_rate"] > stats["emoji_repeat_rate"].median():
            combo.append("♻️ Copy-Paste Emoji Offender (spams the same emoji within AND across messages)")
        if combo:
            print(f"  {sender}: " + ", ".join(combo))

    if not any(
        (row["avg_length"] > med_len and row["double_text_rate"] > med_dt)
        or (row["avg_response_min"] > med_resp and row["avg_emoji"] > med_emoji)
        or (row["double_text_rate"] > med_dt and row["avg_emoji"] > med_emoji)
        for _, row in stats.iterrows()
    ):
        print("  Nobody stacked enough behaviors for a combo title. Suspiciously well-adjusted group.")

    print()
    #print("--- Punishment score (weighted, emoji-repetition penalized hardest) ---")
    # z-scores so different-scale metrics (minutes vs char counts vs rates) combine fairly
    punishment = (
        zscore("avg_response_min").fillna(0) * 1.0
        + zscore("avg_length").fillna(0) * 1.0
        + zscore("double_text_rate").fillna(0) * 1.0
        + zscore("avg_emoji_run").fillna(0) * 2.0        # heaviest weight: same emoji back-to-back
        + zscore("emoji_repeat_rate").fillna(0) * 1.5    # reusing signature emoji across messages
    )
    stats["punishment_score"] = punishment.round(2)
    #print(stats["punishment_score"].sort_values(ascending=False))

    #print()
    print("--- Full leaderboard ---")
    print(stats.sort_values("punishment_score", ascending=False))
    return stats


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

    print()
    roast_report(df)

    print("\n--- Path A: manual labeling (uncomment to run interactively) ---")
    # labeled = label_sample(df, n=40)
    # model = train_on_labels(labeled, feature_cols)