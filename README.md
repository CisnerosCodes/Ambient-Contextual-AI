# Ambient Contextual AI: Zero-Shot Latent Space Analysis System

## Executive Summary
This project implements an intelligent, automated system for digitizing and analyzing user workflow context in real-time. By leveraging state-of-the-art Computer Vision (CV) and Natural Language Processing (NLP) techniques, the system transforms raw visual data into high-dimensional vector embeddings. This allows for "Zero-Shot" classification and analysis—meaning the system can quantify focus and generate semantic narratives without requiring task-specific model training.

The solution demonstrates the application of novel Deep Learning architectures (CLIP, Large Language Models) to create a decision support tool that senses, thinks, and acts to provide actionable insights into human-computer interaction.

## Technical Architecture

The system operates on a modular architecture designed for local execution, ensuring data privacy and low latency.

### 1. Visual Semantic Embedding Engine (Computer Vision)
*   **Technology**: `sentence-transformers` (CLIP: Contrastive Language-Image Pre-training).
*   **Methodology**: The system captures high-frequency visual data (screenshots) and projects them into a 512-dimensional vector space.
*   **Mathematical Foundation**: Utilizes **Linear Algebra** and **Cosine Similarity** statistics to measure the angular distance between the current state vector and a reference "Anchor" vector (Ideal Work State). This provides a continuous, quantitative metric for "Focus" ($0.0$ to $1.0$) rather than discrete binary classification.

### 2. Automated Narrative Generation (Generative AI)
*   **Technology**: `Ollama` (running local LLMs like Llama 3 or Gemma).
*   **Methodology**: An Optical Character Recognition (OCR) pipeline extracts textual data from the visual feed. This unstructured text is processed by a local Large Language Model to synthesize concise, human-readable narratives of hourly activity.
*   **Objective**: To transform raw sensor data into semantic context ("What was the user working on?").

### 3. Real-Time Intelligence Dashboard
*   **Technology**: `Streamlit`, `Plotly`, `SQLite`.
*   **Features**:
    *   **Focus Wave**: A temporal visualization of the user's cognitive load and focus consistency.
    *   **Semantic Search**: Enables natural language querying of the visual history (e.g., "Show me when I was coding Python") by mapping text queries into the same latent vector space as the images.

## Tech Stack & Requirements

*   **Core Language**: Python 3.12+
*   **Computer Vision / ML**: `sentence-transformers` (CLIP-ViT-B-32), `Pillow`
*   **Generative AI**: `Ollama` (Local Inference)
*   **OCR Engine**: Tesseract 5.0
*   **Data Storage**: SQLite (Relational + Vector storage)
*   **Visualization**: Streamlit, Plotly Express

## Installation & Usage

### Prerequisites
*   Python 3.10+
*   [Ollama](https://ollama.com/) (for local LLM inference)
*   [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) (for text extraction)

### Setup
1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/CisnerosCodes/Ambient-Contextual-AI.git
    ```
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Initialize AI Models**:
    ```bash
    ollama pull llama3
    ```

### Operation
1.  **Initialize Sensor (Background Process)**:
    Starts the data collection and embedding pipeline.
    ```bash
    python sensor.py
    ```
2.  **Launch Dashboard (Decision Support Interface)**:
    Visualizes the analyzed data.
    ```bash
    streamlit run dashboard.py
    ```

## Research & Development Objectives
*   **Objective 1**: Validate the efficacy of CLIP embeddings for unsupervised activity recognition.
*   **Objective 2**: Develop a privacy-preserving "Ambient Intelligence" that operates entirely offline.
*   **Objective 3**: Bridge the gap between raw pixel data and high-level semantic understanding using multimodal models.

---
*Developed by Adrian Cisneros for R&D in Computer Vision and Intelligent Systems.*
