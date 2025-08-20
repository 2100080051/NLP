import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
st.set_page_config(page_title="Persona Finder — Natural Language Search", page_icon="🧭", layout="wide")
@st.cache_resource(show_spinner=False)
def load_encoder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)
def norm_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    if isinstance(x, (list, tuple, set)):
        return ", ".join(map(str, x))
    return str(x)
def clean_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t or "").strip()
    return t
def build_persona_text(row: pd.Series) -> str:

    bits = []
    name = norm_text(row.get("name"))
    location = norm_text(row.get("location"))
    role = norm_text(row.get("role"))
    interests = norm_text(row.get("interests"))
    values = norm_text(row.get("values"))
    bio = norm_text(row.get("bio"))
    tags = norm_text(row.get("tags"))
    other = []


    if name:
        bits.append(f"Name: {name}.")
    if role:
        bits.append(f"Role: {role}.")
    if location:
        bits.append(f"Location: {location}.")
    if interests:
        bits.append(f"Interests: {interests}.")
    if values:
        bits.append(f"Values: {values}.")
    if tags:
        bits.append(f"Tags: {tags}.")
    if bio:
        other.append(bio)

    return clean_text(" ".join(bits + other))

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:

    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)

def embed_texts(encoder: SentenceTransformer, texts: List[str]) -> np.ndarray:
    emb = encoder.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    return np.array(emb, dtype=np.float32)

def parse_listish(s: str) -> List[str]:
    if not s:
        return []

    return [w.strip().lower() for w in re.split(r"[;,]", s) if w.strip()]

def keyword_overlap_score(haystack: str, needles: List[str]) -> int:
    hay = haystack.lower()
    score = 0
    for n in needles:

        if n and n in hay:
            score += 1
    return score

def apply_rules_and_score(
    base_sim: float,
    persona_text: str,
    include_keywords: List[str],
    exclude_keywords: List[str],
    hard_exclude_flags: Dict[str, bool],
    row: pd.Series,
) -> Dict[str, Any]:

    persona_lower = persona_text.lower()


    if hard_exclude_flags.get("exclude_smokers", False):

        raw = " ".join([norm_text(row.get("bio")), norm_text(row.get("tags")), norm_text(row.get("values"))]).lower()
        if "smoker" in raw or "smoking" in raw:
            return {"keep": False, "score": 0.0, "penalty": "smoker"}


    include_hits = keyword_overlap_score(persona_lower, include_keywords)
    exclude_hits = keyword_overlap_score(persona_lower, exclude_keywords)


    base_points = max(0.0, min(85.0, base_sim * 85.0))


    bonus = include_hits * 3.0
    malus = exclude_hits * 5.0

    # Final capped 0..100
    score = max(0.0, min(100.0, base_points + bonus - malus))

    return {"keep": True, "score": score, "penalty": None, "include_hits": include_hits, "exclude_hits": exclude_hits}

def build_insights(row: pd.Series, query: str, include_keywords: List[str]) -> Dict[str, str]:
    name = norm_text(row.get("name")) or "This person"
    location = norm_text(row.get("location"))
    interests = norm_text(row.get("interests"))
    values = norm_text(row.get("values"))
    role = norm_text(row.get("role"))

    highlights = []
    if role:
        highlights.append(f"Role: {role}")
    if location:
        highlights.append(f"Location: {location}")
    if interests:

        parts = [p.strip() for p in re.split(r"[;,]", interests) if p.strip()]
        if parts:
            highlights.append("Interests: " + ", ".join(parts[:3]))
    if values:
        parts = [p.strip() for p in re.split(r"[;,]", values) if p.strip()]
        if parts:
            highlights.append("Values: " + ", ".join(parts[:3]))


    opener = f"Hi {name.split()[0] if name!='This person' else ''}, I saw we share interests in {', '.join(include_keywords[:2]) or 'similar areas'}. Would love to connect and chat about {', '.join(include_keywords[:1]) or 'your work'}."
    followups = [
        "Suggest a quick intro call or coffee.",
        "Share one article or resource related to their interests.",
        "Ask a focused question about their current projects."
    ]

    return {
        "highlights": " · ".join(highlights) if highlights else "",
        "opener": opener,
        "next_steps": " • ".join(followups)
    }
st.title("🔎 Persona Finder — Natural-Language Search")
st.caption("Vector search over deep personas (free, local, no paid APIs). Enter plain English like: *“New to NYC, into AI/ML research, cares about climate, no smokers.”*")

with st.sidebar:
    st.subheader("📦 Data")
    data_file = st.file_uploader("Upload personas CSV", type=["csv"])
    st.caption("Expected columns (flexible): name, location, role, interests, values, bio, tags")
    st.divider()
    st.subheader("🧠 Encoder")
    model_name = st.selectbox("Sentence model", ["sentence-transformers/all-MiniLM-L6-v2"], index=0)
    encoder = load_encoder(model_name)
    st.success("Encoder ready")

col_q1, col_q2 = st.columns([1.1, 0.9])
with col_q1:
    st.subheader("Your Query")
    user_query = st.text_area(
        "Describe the kind of people you want to find",
        value="I just moved to NYC. Help me find people interested in AI and physics who love to play tennis.",
        height=120
    )
with col_q2:
    st.subheader("Refinements")
    include_raw = st.text_input("Must-have keywords (comma-separated)", value="AI, physics, tennis")
    exclude_raw = st.text_input("Avoid keywords (comma-separated)", value="smoker")
    top_k = st.slider("How many results?", 1, 20, 5)
    exclude_smokers = st.checkbox("Hard exclude smokers", value=True)

include_keywords = parse_listish(include_raw)
exclude_keywords = parse_listish(exclude_raw)
with st.expander("Optional: Your persona (to tune compatibility)"):
    your_role = st.text_input("Your role/title", value="")
    your_location = st.text_input("Your location", value="")
    your_interests = st.text_input("Your interests", value="")
    your_values = st.text_input("Your values", value="")
user_context_text = clean_text(
    f"User role: {your_role}. User location: {your_location}. User interests: {your_interests}. User values: {your_values}."
)

st.divider()
if not data_file:
    st.info("Upload a personas CSV to begin. (See sample in the repo README.)")
    st.stop()
try:
    df = pd.read_csv(data_file)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

if df.empty:
    st.warning("Your CSV is empty.")
    st.stop()


persona_texts = df.apply(build_persona_text, axis=1).tolist()
query_full = clean_text(
    f"{user_query} Must include: {', '.join(include_keywords)}. Avoid: {', '.join(exclude_keywords)}. {user_context_text}"
)


with st.spinner("Encoding personas and query…"):
    persona_vecs = embed_texts(encoder, persona_texts)
    query_vec = embed_texts(encoder, [query_full])[0]
rows = []
for idx, row in df.iterrows():
    base_sim = cosine_sim(query_vec, persona_vecs[idx])
    rules = apply_rules_and_score(
        base_sim=base_sim,
        persona_text=persona_texts[idx],
        include_keywords=include_keywords,
        exclude_keywords=exclude_keywords,
        hard_exclude_flags={"exclude_smokers": exclude_smokers},
        row=row,
    )
    if not rules["keep"]:
        continue

    rows.append({
        "index": idx,
        "name": norm_text(row.get("name")),
        "location": norm_text(row.get("location")),
        "role": norm_text(row.get("role")),
        "highlights_text": persona_texts[idx],
        "compatibility": round(rules["score"], 1),
        "base_similarity": round(base_sim, 3),
    })

if not rows:
    st.warning("No profiles met your criteria. Try relaxing excludes or adding more includes.")
    st.stop()


rows.sort(key=lambda x: (-x["compatibility"], -x["base_similarity"]))
results = rows[:top_k]
st.subheader("Top Matches")
for r in results:
    p = df.iloc[r["index"]]
    insights = build_insights(p, user_query, include_keywords)
    with st.container(border=True):
        tcol, scol = st.columns([0.7, 0.3])
        with tcol:
            st.markdown(f"### {r['name'] or 'Unknown'}  —  {r['role'] or ''}")
            subbits = [b for b in [p.get('location'), p.get('interests'), p.get('values')] if pd.notna(b) and b]
            if subbits:
                st.caption(" | ".join(map(str, subbits)))
            st.progress(min(1.0, r["compatibility"] / 100.0), text=f"Compatibility: {r['compatibility']}%")
            if insights["highlights"]:
                st.write(f"**Highlights:** {insights['highlights']}")

            excerpt = (r["highlights_text"][:260] + "…") if len(r["highlights_text"]) > 260 else r["highlights_text"]
            st.write(excerpt)
        with scol:
            st.markdown("**Action points**")
            st.write("1) " + insights["opener"])
            st.write("2) " + insights["next_steps"])

st.divider()


export_df = pd.DataFrame([{
    "name": df.iloc[r["index"]].get("name"),
    "role": df.iloc[r["index"]].get("role"),
    "location": df.iloc[r["index"]].get("location"),
    "interests": df.iloc[r["index"]].get("interests"),
    "values": df.iloc[r["index"]].get("values"),
    "compatibility_%": r["compatibility"],
} for r in results])

csv_bytes = export_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download results (CSV)", data=csv_bytes, file_name="persona_matches.csv", mime="text/csv")

st.caption("Tip: add your own flags to the CSV like `smoker`, `remote_only`, `open_to_collab`, and extend the hard-filters easily in code.")
