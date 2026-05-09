from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


AUDIT_PATH = Path("outputs/audit/targeted_adequacy_audit_draft.csv")
SCORE_OPTIONS = ["?", "1", "0"]
SCORE_LABELS = {
    "1": "Yes (1)",
    "0": "No (0)",
    "?": "Unclear (?)",
}
GOLD_STATUS_OPTIONS = {
    "ok": "Gold is usable",
    "source_issue": "English/source issue",
    "gold_needs_edit": "I edited verified gold",
    "unclear": "Unclear",
}
AUDIT_DECISION_OPTIONS = {
    "keep": "Keep for adequacy",
    "exclude_from_adequacy": "Exclude from adequacy",
    "data_quality_caveat": "Use as data caveat",
}
MODELS = [
    ("teacher", "teacher_modern", "Teacher"),
    ("full", "full", "Full"),
    ("lora", "lora", "LoRA"),
    ("adapter", "adapter_ia3", "IA3"),
    ("baseline", "baseline_zeroshot", "Zero-shot"),
]


st.set_page_config(page_title="Tamil Audit Review", layout="wide")


def load_audit() -> pd.DataFrame:
    if not AUDIT_PATH.exists():
        st.error(f"Missing {AUDIT_PATH}. Run: python3 scripts/data/create_targeted_adequacy_audit.py")
        st.stop()
    return pd.read_csv(AUDIT_PATH, dtype=str).fillna("")


def save_audit(df: pd.DataFrame) -> None:
    df.to_csv(AUDIT_PATH, index=False)


def score_index(value: str) -> int:
    return SCORE_OPTIONS.index(value) if value in SCORE_OPTIONS else 0


def option_index(options: dict[str, str], value: str) -> int:
    keys = list(options)
    return keys.index(value) if value in keys else 0


def score_buttons(title: str, state_key: str, current_value: str) -> str:
    if state_key not in st.session_state:
        st.session_state[state_key] = current_value if current_value in SCORE_OPTIONS else "?"

    st.markdown(f"**{title}**")
    selected = st.pills(
        title,
        options=["1", "0", "?"],
        format_func=lambda value: SCORE_LABELS[value],
        key=state_key,
        label_visibility="collapsed",
    )
    if selected:
        return selected
    return "?"


def write_current_row(
    df: pd.DataFrame,
    row_pos: int,
    updates: dict[str, str],
    gold_verified: str,
    gold_status: str,
    audit_decision: str,
    gold_notes: str,
    note: str,
) -> None:
    df.at[row_pos, "gold_verified"] = gold_verified
    df.at[row_pos, "gold_status"] = gold_status
    df.at[row_pos, "audit_decision"] = audit_decision
    df.at[row_pos, "gold_notes"] = gold_notes
    for key, value in updates.items():
        df.at[row_pos, key] = value
    df.at[row_pos, "review_note"] = note
    save_audit(df)


df = load_audit()
idx_values = df["idx"].tolist()

if "selected_idx" not in st.session_state or st.session_state["selected_idx"] not in idx_values:
    st.session_state["selected_idx"] = idx_values[0]

st.title("Targeted Adequacy/Register Audit")
st.caption("Buttons write 1 = yes, 0 = no, ? = unclear to the CSV.")

with st.sidebar:
    st.header("Progress")
    reviewed = df["review_note"].astype(str).str.strip().ne("").sum()
    st.metric("Rows with notes", f"{reviewed}/{len(df)}")

    picked_idx = st.selectbox(
        "Example",
        options=idx_values,
        index=idx_values.index(st.session_state["selected_idx"]),
        format_func=lambda idx: f"idx {idx} · {df.loc[df['idx'] == idx, 'bucket'].iloc[0]}",
    )
    st.session_state["selected_idx"] = picked_idx

    current_i = idx_values.index(st.session_state["selected_idx"])
    nav_prev, nav_next = st.columns(2)
    with nav_prev:
        if st.button("Previous", use_container_width=True, disabled=current_i == 0):
            st.session_state["selected_idx"] = idx_values[current_i - 1]
            st.rerun()
    with nav_next:
        if st.button("Next", use_container_width=True, disabled=current_i == len(idx_values) - 1):
            st.session_state["selected_idx"] = idx_values[current_i + 1]
            st.rerun()

    st.divider()
    st.write("Draft buckets")
    st.dataframe(
        df["bucket"].value_counts().rename_axis("bucket").reset_index(name="count"),
        hide_index=True,
        use_container_width=True,
    )

selected_idx = st.session_state["selected_idx"]
row_pos = df.index[df["idx"] == selected_idx][0]
row = df.loc[row_pos]
current_i = idx_values.index(selected_idx)

st.subheader(f"idx {row['idx']} · {row['bucket']}")
st.write(row["selection_reason"])

source_col, ref_col = st.columns([1, 1])
with source_col:
    st.markdown("**English**")
    st.info(row["en"])
with ref_col:
    st.markdown("**Human References**")
    st.write(f"Ref 1: {row['human_colloquial_ref']}")
    st.write(f"Ref 2: {row['human_annotator_2']}")

st.markdown("**Gold / Data Quality Review**")
gold_col, status_col, decision_col = st.columns([4, 1.4, 1.8])
with gold_col:
    gold_verified = st.text_area(
        "Verified gold",
        value=row.get("gold_verified", row["human_colloquial_ref"]),
        help="Do not edit the original human refs. Put a corrected usable gold here if needed.",
        height=70,
    )
with status_col:
    gold_status = st.selectbox(
        "Gold status",
        list(GOLD_STATUS_OPTIONS),
        index=option_index(GOLD_STATUS_OPTIONS, row.get("gold_status", "ok")),
        format_func=lambda value: GOLD_STATUS_OPTIONS[value],
    )
with decision_col:
    audit_decision = st.selectbox(
        "Audit decision",
        list(AUDIT_DECISION_OPTIONS),
        index=option_index(AUDIT_DECISION_OPTIONS, row.get("audit_decision", "keep")),
        format_func=lambda value: AUDIT_DECISION_OPTIONS[value],
    )

gold_notes = st.text_area(
    "Gold/source notes",
    value=row.get("gold_notes", ""),
    placeholder="Example: English source appears mistranslated from the Tamil references; exclude from adequacy claims.",
    height=80,
)

st.divider()

updates: dict[str, str] = {}
for model_key, text_col, label in MODELS:
    st.markdown(f"### {label}")
    sentence_col, adequacy_col, register_col = st.columns([4.2, 2.1, 2.1])

    with sentence_col:
        st.write(row[text_col])

    adequacy_key = f"{model_key}_adequacy"
    register_key = f"{model_key}_register"

    with adequacy_col:
        updates[adequacy_key] = score_buttons(
            "Adequacy",
            f"{row['idx']}_{adequacy_key}",
            row[adequacy_key],
        )

    with register_col:
        updates[register_key] = score_buttons(
            "Register",
            f"{row['idx']}_{register_key}",
            row[register_key],
        )

    st.divider()

note = st.text_area(
    "Review note",
    value=row["review_note"],
    placeholder="Write why this example is useful, or what needs checking.",
    height=100,
)

button_col, next_col, status_col = st.columns([1, 1, 4])
with button_col:
    save = st.button("Save", type="primary", use_container_width=True)
with next_col:
    save_next = st.button(
        "Save & Next",
        use_container_width=True,
        disabled=current_i == len(idx_values) - 1,
    )
with status_col:
    st.caption(f"Draft note: {row['draft_note']}")

if save or save_next:
    write_current_row(
        df,
        row_pos,
        updates,
        gold_verified,
        gold_status,
        audit_decision,
        gold_notes,
        note,
    )
    st.success(f"Saved idx {row['idx']} to {AUDIT_PATH}")
    if save_next and current_i < len(idx_values) - 1:
        st.session_state["selected_idx"] = idx_values[current_i + 1]
        st.rerun()
