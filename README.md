# Ambient Contextual AI

> **Zero-Shot Latent Space Analysis for Real-Time Productivity Intelligence**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CLIP](https://img.shields.io/badge/Model-CLIP--ViT--B--32-orange.svg)](https://openai.com/research/clip)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-purple.svg)](https://ollama.com)

<p align="center">
  <img src="assets/dashboard_preview.png" alt="Dashboard Preview" width="700">
</p>

## 📋 Executive Summary

This project implements an intelligent, automated system for digitizing and analyzing user workflow context in real-time. By leveraging state-of-the-art **Computer Vision (CV)** and **Natural Language Processing (NLP)** techniques, the system transforms raw visual data into high-dimensional vector embeddings.

This allows for **"Zero-Shot" classification and analysis**—meaning the system can quantify focus and generate semantic narratives **without requiring task-specific model training**.

### ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🎯 **Focus Quantification** | Measures productivity using cosine similarity in 512-dim latent space |
| 📖 **Automated Narratives** | Local LLM generates hourly activity summaries |
| 🔍 **Semantic Search** | Query your visual history with natural language |
| 🔒 **100% Offline** | All processing runs locally—zero cloud dependencies |

---

## 🏗️ Technical Architecture

The system operates on a modular architecture designed for local execution, ensuring data privacy and low latency.

```
┌─────────────────────────────────────────────────────────────────┐
│                    AMBIENT CONTEXTUAL AI                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   SENSOR     │───▶│   ANALYSIS   │───▶│  DASHBOARD   │      │
│  │  (sensor.py) │    │ (analysis.py)│    │(dashboard.py)│      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  CLIP ViT    │    │   Ollama     │    │  Streamlit   │      │
│  │  Embeddings  │    │   LLM        │    │  + Plotly    │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                              │                                  │
│                              ▼                                  │
│                    ┌──────────────────┐                        │
│                    │   SQLite + JSON  │                        │
│                    │  (Vector Store)  │                        │
│                    └──────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

### 1. 👁️ Visual Semantic Embedding Engine (Computer Vision)

| Component | Details |
|-----------|--------|
| **Technology** | `sentence-transformers` (CLIP: Contrastive Language-Image Pre-training) |
| **Vector Dimension** | 512-dimensional latent space |
| **Similarity Metric** | Cosine Similarity for focus quantification (0.0 → 1.0) |

The system captures visual data and projects it into a high-dimensional vector space, enabling mathematical comparison between states.

### 2. 🤖 Automated Narrative Generation (Generative AI)

| Component | Details |
|-----------|--------|
| **Technology** | `Ollama` (Llama 3 / Gemma - Local Inference) |
| **Input** | OCR-extracted text from screenshots |
| **Output** | Human-readable hourly activity summaries |

Transforms raw sensor data into semantic context: *"What was the user working on?"*

### 3. 📊 Real-Time Intelligence Dashboard

| Feature | Description |
|---------|-------------|
| **Focus Wave** | Temporal visualization of cognitive load and focus consistency |
| **Semantic Search** | Natural language querying of visual history |
| **Daily Narrative** | LLM-generated summaries of work sessions |

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.10+ |
| **Computer Vision** | CLIP-ViT-B-32, Pillow, OpenCV |
| **Generative AI** | Ollama (Llama 3 / Gemma) |
| **OCR** | Tesseract 5.0 |
| **Database** | SQLite + JSON (Vector Storage) |
| **Frontend** | Streamlit, Plotly Express |

---

## 🚀 Quick Start

### Prerequisites

- [x] Python 3.10+
- [x] [Ollama](https://ollama.com/) (for local LLM inference)
- [x] [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) (for text extraction)

### Installation

```bash
# Clone the repository
git clone https://github.com/CisnerosCodes/Ambient-Contextual-AI.git
cd Ambient-Contextual-AI

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download LLM model
ollama pull llama3
```

### Usage

```bash
# Terminal 1: Start the sensor (runs in background)
python sensor.py

# Terminal 2: Launch the dashboard
streamlit run dashboard.py
```

> 💡 **Tip**: Set your "Anchor" (ideal work state) in the dashboard sidebar to start tracking focus.

---

## 📈 Research & Development Objectives

1. **Zero-Shot Recognition**: Validate CLIP embeddings for unsupervised activity classification
2. **Privacy-First Design**: 100% offline processing with no cloud dependencies
3. **Multimodal Understanding**: Bridge raw pixels → semantic meaning using vision-language models

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <b>Developed by Adrian Cisneros</b><br>
  <i>R&D in Computer Vision and Intelligent Systems</i>
</p>
