# 🚨 Crisis Intelligence Command Center
**A Multimodal RAG System for National Disaster Response**

![Project Status](https://img.shields.io/badge/Status-Prototype-orange)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Stack](https://img.shields.io/badge/Stack-Streamlit%20%7C%20Qdrant%20%7C%20Gemini-green)

## 📖 Overview
In the chaotic aftermath of natural disasters, critical information is often fragmented across radio logs, text reports, and visual evidence (drone/CCTV footage). The **Crisis Intelligence Command Center** bridges this gap.

This is a **Multimodal Retrieval-Augmented Generation (RAG)** application that allows emergency responders to:
1.  **Ingest** both text logs and images into a shared vector space.
2.  **Query** the database using natural language (e.g., *"Show me flooding in the north"*).
3.  **Retrieve** grounded evidence: The AI fetches the exact text report *and* the matching visual evidence.
4.  **Reason** across modalities to provide actionable situational awareness.

---

## 🧠 System Architecture
The system utilizes a **Dual-Stream Vector Architecture** to handle different data types without losing semantic precision.

### 1. The Ingestion Layer
- **Text Logs:** Processed by `all-MiniLM-L6-v2` (384-dimensional embeddings) -> Stored in `user_episodic_memory`.
- **Images:** Processed by **OpenAI CLIP** (`clip-ViT-B-32`, 512-dimensional embeddings) -> Stored in `disaster_multimodal`.

### 2. The Retrieval Layer (Parallel Search)
When a user asks a question, the system performs two simultaneous vector searches:
- **Semantic Text Search:** Finds relevant chat history and system reports (Threshold: >0.40).
- **Visual Similarity Search:** Finds relevant images using CLIP (Threshold: >0.25).

### 3. The Generation Layer
- **Context Aggregation:** Text logs and Image metadata are combined.
- **Reasoning:** **Google Gemini 1.5 Flash** synthesizes the evidence into a final response.
- **Safety:** Negative constraints (e.g., "Don't show...") function as a final guardrail.

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9+
- A [Qdrant Cloud](https://cloud.qdrant.io/) Account (Free Tier is sufficient)
- A [Google AI Studio](https://aistudio.google.com/) API Key

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/Crisis-Intelligence-AI.git](https://github.com/YOUR_USERNAME/Crisis-Intelligence-AI.git)
cd Crisis-Intelligence-AI

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

```

### 3. Configure Environment

Create a `.env` file in the root directory and add your API keys:

```ini
GEMINI_API_KEY=your_google_api_key_here
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key

```

### 4. Load Base Data (Simulated Scenario)

This script will ingest the 50+ Pan-India disaster logs and images into your vector database.

```bash
python ingest_bulk.py

```

*Note: Ensure your images are in the `data/data_images/` folder and logs are in `data/data_logs.txt`.*

### 5. Run the Application

```bash
streamlit run app.py

```

---

## 💡 Key Features

### 🔄 Stateful Memory with "Safe Reset"

The system maintains a conversation history so you can ask follow-up questions.

* **Smart Wipe:** The "Start New Scenario" button clears *only* the user conversation (`role="user"`) but **preserves** the System Reports (`role="system_report"`).
* **Benefit:** You can run back-to-back demos without re-ingesting data.

### 👁️ Multimodal "Grounding"

Unlike standard chatbots that hallucinate, this system provides **Evidence citations**.

* If it says "There is a flood," it displays the **retrieved image** and the **source text log** that led to that conclusion.

---

## 📂 Project Structure

```text
Crisis-Intelligence-AI/
├── data/
│   ├── data_images/        # Disaster imagery for the demo
│   └── data_logs.txt       # Text logs matching the images
├── documents/              # Project Report & System Design PDF
│   ├── Final_Report.pdf    
│   └── Architecture.png    
├── app.py                  # Main Streamlit Application
├── ingest_bulk.py          # Data Ingestion Script
├── requirements.txt        # Python Dependencies
└── README.md               # Documentation           
```

## ⚠️ Limitations

* **API Dependency:** Requires internet connectivity for Google Gemini and Qdrant Cloud.
* **Visual Thresholds:** Low-confidence image matches (score < 0.25) are suppressed to prevent misinformation.
