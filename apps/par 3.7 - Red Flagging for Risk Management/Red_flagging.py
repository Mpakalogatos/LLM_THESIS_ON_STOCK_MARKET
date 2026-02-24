import streamlit as st
import requests
import json
import re
import html

# --- Page setup ---
st.set_page_config(page_title="Red Flag Detector", page_icon="🚩")
st.title("🚩 Corporate Announcement Red Flag Detector")
st.write("Analyze a company’s statement for vague, overly-positive, or misleading language using a local LLaMA 3 model via Ollama.")

# --- Inputs ---
company = st.text_input("Company name", "Starbucks Corporation")
date = st.text_input("Announcement date", "2024-04-30")

announcement = st.text_area(
    "Paste the corporate announcement text below:",
    """In a highly challenged environment, this quarter’s results do not reflect the power of our brand,
our capabilities or the opportunities ahead. We have a clear plan to execute and the entire organization
is mobilized around it. We are very confident in our long-term and know that our Triple Shot Reinvention
with Two Pumps strategy will deliver on the limitless potential of this brand."""
)

# --- Helper function for color-coding ---
def get_highlight_color(reason: str) -> str:
    reason_lower = reason.lower()
    if any(k in reason_lower for k in ["vague", "unclear", "nonspecific", "ambiguous"]):
        return "#fff3cd"  # yellow
    elif any(k in reason_lower for k in ["positive", "confident", "optimistic", "hyperbolic", "boastful"]):
        return "#f8d7da"  # red/pink
    elif any(k in reason_lower for k in ["misleading", "unrealistic", "exaggerated", "unsupported", "deceptive"]):
        return "#ffe5b4"  # orange
    else:
        return "#e2e3e5"  # gray (other)

# --- Function to highlight phrases with tooltip justification ---
def highlight_phrases(text, flags):
    flags_sorted = sorted(flags, key=lambda x: -len(x["phrase"]))
    for f in flags_sorted:
        phrase = re.escape(f["phrase"])
        color = get_highlight_color(f["reason"])
        tooltip = html.escape(f.get("justification", f["reason"]))
        text = re.sub(
            phrase,
            lambda m: f"<span style='background-color:{color}; color:#000; font-weight:bold;' title='{tooltip}'>{m.group(0)}</span>",
            text,
            flags=re.IGNORECASE
        )
    return text

# --- Main button ---
if st.button("Analyze"):
    if not announcement.strip():
        st.warning("Please enter an announcement to analyze.")
        st.stop()

    API_URL = "http://localhost:11434/api/generate"

    prompt = f"""
You are a financial communications analyst.
Read the following corporate announcement from {company} dated {date}.
Identify any vague, overly-positive, or potentially misleading phrases.
Return *only* a valid JSON list of objects, each with fields:
- "phrase": the flagged text
- "reason": the category (vague, overly positive, misleading)
- "justification": a short explanation of why it was flagged (1-2 sentences)

Return no extra text outside the JSON.

Announcement:
\"\"\"{announcement}\"\"\"
"""

    payload = {"model": "llama3", "prompt": prompt, "stream": True}

    progress_bar = st.progress(0, text="Analyzing with LLaMA 3...")

    # --- Stream response from Ollama ---
    try:
        response = requests.post(API_URL, json=payload, stream=True)
        response.raise_for_status()

        collected_text = ""
        total_chars = 0
        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    data = json.loads(line)
                    chunk = data.get("response", "")
                    collected_text += chunk
                    total_chars += len(chunk)
                    progress = min(total_chars / 1500, 1.0)
                    progress_bar.progress(progress, text="Analyzing with LLaMA 3...")
                except json.JSONDecodeError:
                    continue
        progress_bar.progress(1.0, text="✅ Analysis complete!")
        result_text = collected_text.strip()
    except Exception as e:
        st.error(f"API request failed: {e}")
        st.stop()

    # --- Extract JSON output from model response ---
    match = re.search(r'\[.*?\]', result_text, re.DOTALL)
    if not match:
        st.warning("No valid JSON found in model output.")
        st.text_area("Raw model output", result_text, height=200)
        st.stop()

    try:
        flags = json.loads(match.group(0))
    except json.JSONDecodeError:
        st.warning("Could not parse model output as JSON.")
        st.text_area("Raw model output", result_text, height=200)
        st.stop()

    if not flags:
        st.success("No obvious red flags detected. ✅")
        st.stop()

    # --- Highlighted text with tooltip justification ---
    highlighted_html = highlight_phrases(announcement, flags).replace("\n", "<br>")

    st.subheader("🔍 Highlighted Announcement")
    st.markdown(
        f"<div style='max-height:400px; overflow:auto; line-height:1.5;'>{highlighted_html}</div>",
        unsafe_allow_html=True
    )

    st.markdown("""
**Color Legend:**  
🟡 Vague / unclear  🔴 Overly positive  🟠 Misleading / unrealistic  ⚪ Other
""")

    st.subheader("🚩 Detected Red Flags")
    for f in flags:
        color = get_highlight_color(f["reason"])
        justification = f.get("justification", "No justification provided.")
        with st.expander(f"⚠️ {f['phrase']} ({f['reason'].capitalize()})"):
            st.markdown(
                f"<div style='background-color:{color}; padding:6px; border-radius:4px;'>"
                f"<strong>Reason:</strong> {f['reason'].capitalize()}<br>"
                f"<strong>Justification:</strong> {justification}"
                f"</div>",
                unsafe_allow_html=True
            )

    # --- Download JSON ---
    st.download_button(
        "📥 Download Red Flags",
        json.dumps(flags, indent=2),
        file_name="red_flags.json",
        mime="application/json"
    )
