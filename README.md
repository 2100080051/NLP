# 🔎 Persona Finder — Natural Language Search

A **Streamlit app** that enables natural-language search over deep user personas using **sentence embeddings** (no paid APIs needed).  
This fulfills the requirement of searching people profiles by natural queries, ranking them by compatibility %, and providing actionable insights.  

---

## ✨ Features

- 🧠 **Natural language search**  
  Enter plain-English queries like:  
  > *“I just moved to NYC, help me find people into AI and physics who love tennis, no smokers.”*

- 📦 **Vector DB simulation (local)**  
  Uses [SentenceTransformers](https://www.sbert.net/) to embed personas and queries for semantic search.

- 🎯 **Top matches with compatibility %**  
  Each match shows:
  - Compatibility percentage (0–100)  
  - Highlights (role, location, interests, values)  
  - Suggested opener message  
  - Next-step action points  

- 🛠 **Refinements**  
  - Must-have keywords  
  - Avoid keywords  
  - Hard excludes (e.g. smokers)

- ⬇️ **Export results**  
  Download matched profiles as CSV for further use.

---

## 🚀 Quickstart

### 1. Clone repo
```bash
git clone https://github.com/your-username/persona-finder.git
cd persona-finder
```

### 2. Install requirements
We recommend Python **3.10 or 3.11**.  
```bash
pip install -r requirements.txt
```

### 3. Run Streamlit
```bash
streamlit run app.py
```

### 4. Open in browser
By default → `http://localhost:8501`

---

## 📂 Sample Data

Upload a CSV file containing persona profiles. Example:

```csv
name,location,role,interests,values,bio,tags
Alice,NYC,Researcher,"AI, Physics, Tennis","Vegan, Green","Loves AI and sustainability","tennis, AI"
Sophia,NYC,PhD Student,"Quantum Physics, AI, Tennis","Curiosity, Discovery","Studying quantum physics, enjoys tennis","quantum, tennis, ai"
Ethan,NYC,Data Scientist,"Machine Learning, Physics, Tennis","Open-minded, Collaboration","Works in ML research, plays tennis","ml, physics, tennis"
Charlie,NYC,Professor,"AI, Tennis, Philosophy","Curiosity, Education","Teaches AI and philosophy","professor, tennis"
```

Upload this CSV in the sidebar to test.  

---

## 📊 How It Works

1. **Embed personas** using SentenceTransformers (`all-MiniLM-L6-v2`)  
2. **Embed query + user context**  
3. **Compute cosine similarity** between query and personas  
4. **Apply refinements**:  
   - Boost for must-have keywords  
   - Penalty for avoid keywords  
   - Hard exclude filters  
5. **Rank & return top N** with compatibility % and action points  
6. **Download results** as CSV  

---

## 🛠 Tech Stack

- [Streamlit](https://streamlit.io/) — UI  
- [SentenceTransformers](https://www.sbert.net/) — embeddings  
- [Pandas / NumPy](https://pandas.pydata.org/) — data handling  

---

## ✅ Deliverables Fulfilled

✔ Natural language query over personas  
✔ Returns top matches (5/N)  
✔ Compatibility percentage  
✔ Insights + action points per match  
✔ Refinements (must-have, avoid, exclude)  
✔ CSV export  

---

## 📸 Screenshot

*(Add a screenshot of your Streamlit app results here)*

---

## 📜 License
MIT License
