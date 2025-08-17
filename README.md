# Visual-Search-Open-Source-Visual-Search-Recommendation-Engine
> An open-source project to search and recommend visually similar items from images using CLIP embeddings and FAISS.  
> Upload an image or use natural language to find similar products, people, or objects.

---

## 🚀 Features
- **Image-to-Image Search** – Find visually similar images using embeddings.
- **Text-to-Image Search** – Use natural language to find matching visuals.
- **Hybrid Search** – Combine image + text queries for better recommendations.
- **Interactive Feature Selection** *(Planned)* –  
  Example: If you upload an image of a hero wearing a red shirt, the system will ask:  
  > "Do you want to search for the hero or the red shirt?"  
  and then recommend accordingly.
- **Scalable Indexing** – Powered by FAISS for fast similarity search.
- **Embeddings Storage** – Precomputed embeddings stored for quick retrieval.  
  *Embeddings link – [https://drive.google.com/drive/folders/1En6Ue-S9w1UgtPT5wtNmgqeNjbiB1vTS?usp=sharing]*

---

## 🗺️ Roadmap
1. ✅ Build core search backend using CLIP + FAISS.
2. ✅ Support both image and text queries.
3. 🚧 Build website interface.
4. 🚧 Implement natural language-based recommendations.
5. 🚧 Add interactive feature selection Q&A flow.
6. 🔜 Deploy as a live web app.

---

## 🏗️ Architecture
```plaintext
User Input (Image/Text)
        │
        ▼
Embedding Generator (CLIP)
        │
        ▼
Vector Database (FAISS)
        │
        ▼
Similarity Search
        │
        ▼
Results & Recommendations


