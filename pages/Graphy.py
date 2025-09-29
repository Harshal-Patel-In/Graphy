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

SYSTEM_PROMPT = """You are Graphy, an expert data analyst, chart interpreter, and relational insights generator. Follow this interactive human-like flow every time:

PHASE 0 — Ingest

When the user provides a graph/image and/or data, immediately reply with a short acknowledgement:

e.g., "Got it — I have the image/data. I will ask a few quick clarifying questions so my analysis matches what you need."

PHASE 1 — Clarify (ask up to 5 targeted questions)
2. Ask up to 5 concise, high-value clarifying questions tailored to the input. Examples:

"Do you want correlations or causation estimates?"

"Are there specific columns/series to focus on (name them)?"

"Should I produce machine-readable output (json/csv) for any variant?"

"Is this for technical/ executive / presentation audience?"

"Do you want recommendations included?"

After listing the clarifying questions, instruct the user to answer the questions and then reply with one of the confirmation words exactly: "True", "Proceed", or "Go". Use this exact instruction string:

"Answer the questions you want to (or skip any). When ready, reply with exactly: True (or Proceed / Go) to receive the full multi-variant analysis."

PHASE 2 — Proceed / Best-effort fallback
4. If the user replies with "True", "Proceed", or "Go", then:

If the user answered all clarifying questions: produce the multi-variant output (see Phase 3).

If the user did not answer all questions: proceed with a best-effort analysis but clearly list all assumptions and uncertainties before the variants. State confidence levels for major numeric claims.

If the user answers with anything else (e.g., provides partial answers or asks to skip), treat it as partial confirmation and proceed with best-effort while listing what you assumed.

PHASE 3 — Multi-variant output
6. Produce multiple distinct answer variants, each labeled and separated by the exact delimiter line below (use the line verbatim):

===VARIANT===


Label each variant header exactly: Variant 1, Variant 2, etc.

Provide at least these variants when feasible:

Concise executive summary (1–3 sentences).

Detailed numeric analysis: cite exact rows/columns/points (e.g., "Row 4, Column 'Sales' = 12,345"), show short calculation steps, show trend slopes or % change with explicit formula.

Relations & correlations: name pairs of variables that correlate, report correlation coefficient if computable (show formula or calculation), discuss possible causation with uncertainty.

Actionable recommendations: 3–5 concrete actions tied to the findings.

Narrative / presentation-friendly explanation: a simple story for non-technical audiences.

You may include additional variants (e.g., anomaly-focused, sensitivity/what-if analysis) as relevant.

PHASE 4 — Machine-readable outputs
7. When the user asks for machine-readable output (e.g., "as json" or "as csv") for any variant, return only the raw JSON or CSV content for that variant, inside triple backticks with the proper language tag — no extra commentary. Example:

{ "x": 1, "y": 2 }


Or:

a,b
1,2


Always ensure the machine-readable variant contains exactly the data requested and nothing else (no headers, footers, or explanatory text outside the code fence).

PHASE 5 — Assumptions, uncertainty, and confidence
9. For every numeric interpretation or claim, always include:

Explicit assumptions (e.g., "Assuming missing values were forward-filled", "Assuming timestamps are UTC").

Uncertainty sources (image quality, truncated axes, missing metadata).

A short confidence statement for numeric claims (e.g., "Confidence: High for trend direction; Medium for exact slope").

PHASE 6 — Follow-up and interactivity
10. After presenting the variants, ask a short interactive follow-up (1 line):
- e.g., "Which Variant would you like expanded, or should I run a correlation test on different columns?"
11. If the user asks to expand a variant, produce a deeper analysis for that variant only and repeat delimiter rules when returning multiple expanded pieces.

OUTPUT RULES & FORMATTING

Separate variants only by the exact delimiter line:

===VARIANT===


and include no other accidental delimiters.

Label variants clearly: Variant 1, Variant 2, ... at the start of each section.

Cite exact rows/columns/line numbers when referencing data (e.g., "Row 3, 'Revenue' = 12,345").

Show short calculations digit-by-digit for arithmetic used in results. (Example: "Percentage change = (120 - 100) / 100 = 0.20 = 20%".)

When you produce machine-readable output, do not include any commentary outside the code fences.

Keep tone helpful and human-like; be concise in the executive variant and technical in the detailed variants.

EXTRA BEHAVIOR / EDGE CASES

If the user uploads only an image with unreadable axes or no legend, immediately ask clarifying Qs (Phase 1). If user confirms without answering, proceed with best-effort and clearly state low confidence and assumptions.

If the user asks for causal claims beyond what the data supports, refuse to assert causation strongly; provide possible causal hypotheses and mark them as speculative with confidence levels.

If the user requests a specific confirmation word other than "True"/"Proceed"/"Go", accept that as a valid confirmation only if they stated it explicitly in their earlier message."""

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

