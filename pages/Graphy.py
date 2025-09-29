import streamlit as st
import google.generativeai as genai
from PIL import Image
import json
import pandas as pd
import io
import re

st.set_page_config(page_title="Graph Chatbot", layout="wide")

# header/logo
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.image("logo.png", width=800)

st.sidebar.image("logo.png", width=400)
st.markdown(""" 
 """)

# 🔑 Configure Gemini
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

# ---------------------------
# Helpers
# ---------------------------
def extract_code_block(text: str):
    if not text:
        return None, None
    m = re.search(r"```(?:\s*(json|csv))?\s*\n([\s\S]*?)```", text, re.IGNORECASE)
    if m:
        lang = (m.group(1) or "").strip().lower()
        block = m.group(2).strip()
        return block, lang
    return None, None

def looks_like_json(s: str) -> bool:
    if not isinstance(s, str):
        return False
    s = s.lstrip()
    return s.startswith("{") or s.startswith("[")

def looks_like_csv(s: str) -> bool:
    if not isinstance(s, str):
        return False
    return ("," in s) and ("\n" in s)

# ---------------------------
# Session state defaults
# ---------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "👋 Hello! How can I assist you with your graph today?"}
    ]

# ensure keys exist (value None is fine)
for key in ["graph_data", "graph_image", "graph_csv", "graph_data_json", "graphy_isolated"]:
    if key not in st.session_state:
        st.session_state[key] = None

if st.session_state.get("graphy_isolated") is None:
    st.session_state["graphy_isolated"] = False

SYSTEM_PROMPT = """You are Graphy, an expert data analyst, chart interpreter, and relational insights generator.

Your goals:

Carefully examine the user’s graph/image and any provided data.

Provide clear, accurate observations with both numerical and contextual interpretation.

Go beyond surface-level trends by also identifying:

Relations and correlations (e.g., "X rises when Y falls").

Possible causations (with uncertainty noted).

Comparisons between categories, groups, or time periods.

Anomalies, outliers, and turning points.

Produce multiple distinct answer variants covering different perspectives, such as:

Concise executive summary.

Detailed numeric/technical analysis (cite exact rows, columns, or data points).

Relational/causal interpretation (explain what variables seem linked and how strongly).

Actionable recommendations (what to do with these insights).

Storytelling/insightful narrative (explain the data as if presenting to a non-technical audience).

When asked for machine-readable output (e.g., "as json", "as csv"), provide only the raw data in the requested format inside triple backticks with the correct language tag — no commentary.

Always state assumptions, uncertainty, and confidence levels in your interpretations.

When doing calculations, show short steps transparently.

Output rules:

Separate answer variants using the exact delimiter line:

===VARIANT===


Label each section as Variant 1, Variant 2, Variant 3, …"""

# ---------------------------
# Isolation toggle UI
# ---------------------------
iso_col1, iso_col2 = st.columns([4, 1])
with iso_col1:
    st.checkbox(
        "Isolate Graphy (don't auto-import Visualizer data)",
        value=st.session_state.get("graphy_isolated", False),
        key="graphy_isolated",
        help="When ON, Graphy will not automatically import the graph data loaded in Visualizer. Use this to keep Graphy independent."
    )

# ---------------------------
# Normalize shared session data
# ---------------------------
gd = None
if not st.session_state.get("graphy_isolated", False):
    gd = st.session_state.get("graph_data")

if isinstance(gd, pd.DataFrame):
    st.session_state["graph_csv"] = gd
    try:
        st.session_state["graph_data_json"] = gd.to_dict(orient="records")
    except Exception:
        st.session_state["graph_data_json"] = None
elif isinstance(gd, (dict, list)):
    st.session_state["graph_data_json"] = gd
else:
    st.session_state["graph_data_json"] = st.session_state.get("graph_data_json", None)

# ---------------------------
# UI - Uploads
# ---------------------------
st.title("📊 Graph Chatbot")

col1, col2 = st.columns(2)

with col1:
    uploaded_img = st.file_uploader("Upload Graph Image", type=["png", "jpg", "jpeg"], key="img_upl")
    if uploaded_img is not None:
        try:
            st.session_state["graph_image"] = Image.open(uploaded_img)
            st.success("✅ Image uploaded!")
        except Exception as e:
            st.error(f"Image load failed: {e}")

with col2:
    uploaded_csv = st.file_uploader("Upload Graph CSV", type="csv", key="csv_upl")
    if uploaded_csv is not None:
        try:
            df_csv = pd.read_csv(uploaded_csv)
            st.session_state["graph_csv"] = df_csv
            st.session_state["graph_data"] = df_csv
            st.session_state["graph_data_json"] = df_csv.to_dict(orient="records")
            st.success("✅ CSV uploaded!")
        except Exception as e:
            st.error(f"CSV load failed: {e}")

# If isolation is ON, show a subtle note
if st.session_state.get("graphy_isolated", False):
    st.info("Graphy is currently isolated. It will not import graph data from Visualizer until you turn isolation off.")

# ---------------------------
# Previews
# ---------------------------
if st.session_state.get("graph_data_json") is not None:
    with st.expander("📂 View JSON Data"):
        st.json(st.session_state.get("graph_data_json"))

if st.session_state.get("graph_image") is not None:
    st.image(st.session_state.get("graph_image"), caption="Uploaded Graph", use_container_width=True)

if st.session_state.get("graph_csv") is not None:
    with st.expander("📂 View CSV Data"):
        st.dataframe(st.session_state.get("graph_csv"))

st.divider()

# Option to choose number of answer variants
n_variants = st.sidebar.slider("Number of answer variants", min_value=1, max_value=5, value=3, help="How many distinct answer variants would you like the model to produce for each prompt?")

# Chat history
for msg in st.session_state.get("messages", []):
    avatar = "🧑" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Chat input
prompt = st.chat_input("Ask about your graph…")
if prompt:
    msgs = st.session_state.get("messages", [])
    msgs.append({"role": "user", "content": prompt})
    st.session_state["messages"] = msgs

    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    variant_instruction = (
        f"Please produce {n_variants} distinct answer variants separated by the exact delimiter line:\n\n===VARIANT===\n\n"
        "Each variant should be labeled (Variant 1, Variant 2, ...). For each variant include:\n"
        "- One-line concise summary\n- Detailed analysis (key numbers/trends, calculations if any)\n- One actionable recommendation\n\n"
        "If the user explicitly asked for machine-readable output ('as json' or 'as csv'), put the raw JSON/CSV in its own variant and wrap it in triple backticks with the proper language tag (```json or ```csv). Otherwise include human-readable text only."
    )

    inputs = [SYSTEM_PROMPT, variant_instruction, prompt]
    lower = prompt.lower()
    if any(k in lower for k in ["as csv", "csv file", "generate csv", "return csv"]):
        inputs.append("IMPORTANT: Output only raw CSV (no explanations, no markdown).")
    if any(k in lower for k in ["as json", "json file", "generate json", "return json"]):
        inputs.append("IMPORTANT: Output only raw JSON (no explanations, no markdown).")

    gjson = st.session_state.get("graph_data_json")
    if gjson is not None:
        try:
            inputs.append("Graph JSON data:\n" + json.dumps(gjson, indent=2))
        except Exception:
            inputs.append("Graph JSON data (stringified):\n" + str(gjson))

    gcsv = st.session_state.get("graph_csv")
    if gcsv is not None:
        try:
            csv_text = gcsv.to_csv(index=False)
            inputs.append("Graph CSV data:\n" + csv_text)
        except Exception:
            pass

    gimg = st.session_state.get("graph_image")
    if gimg is not None:
        inputs.append(gimg)

    try:
        # We keep the same call shape as before but we've structured the inputs so the model receives the system prompt first.
        response = model.generate_content(inputs)
        reply = response.text if response and response.text else "⚠️ No response from model."
    except Exception as e:
        reply = f"⚠️ Model error: {e}"

    # store the raw assistant reply into messages
    msgs = st.session_state.get("messages", [])
    msgs.append({"role": "assistant", "content": reply})
    st.session_state["messages"] = msgs

    # Show the full assistant bubble
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(reply)

    # Try to split into variants using the exact delimiter
    variants = [v.strip() for v in reply.split("===VARIANT===") if v.strip()]

    if not variants:
        # fallback: show whole reply as single variant
        variants = [reply.strip()]

    # Display each variant separately with detection for JSON/CSV code blocks
    for i, var in enumerate(variants):
        with st.expander(f"Variant {i+1}"):
            st.markdown(var)

            # attempt to extract a fenced code block (json/csv)
            extracted, lang = extract_code_block(var)
            candidate = extracted if extracted else var.strip()

            if lang == "json" or looks_like_json(candidate):
                try:
                    parsed_json = json.loads(candidate)
                    st.download_button(
                        f"⬇️ Download Variant {i+1} JSON",
                        data=json.dumps(parsed_json, indent=2),
                        file_name=f"generated_variant_{i+1}.json",
                        mime="application/json",
                    )
                    with st.expander(f"👀 Preview Variant {i+1} JSON"):
                        st.json(parsed_json)
                except Exception:
                    pass

            elif lang == "csv" or looks_like_csv(candidate):
                try:
                    df = pd.read_csv(io.StringIO(candidate))
                    st.download_button(
                        f"⬇️ Download Variant {i+1} CSV",
                        data=candidate,
                        file_name=f"generated_variant_{i+1}.csv",
                        mime="text/csv",
                    )
                    with st.expander(f"👀 Preview Variant {i+1} CSV"):
                        st.dataframe(df)
                except Exception:
                    pass
