# eu-funding-topic-matcher
Flask API for semantic matching of project proposals to Horizon Europe funding topics using Sentence-Transformers and FAISS.

## Project Overview & Goal

Finding the right funding opportunity within the vast Horizon Europe program is a complex and time-consuming task for universities, SMEs, and researchers. This project solves that problem by providing an AI-powered system that automatically matches project proposals to the most relevant funding calls.

It leverages semantic search to understand the context of a proposal and deliver accurate, fast recommendations, streamlining the grant discovery process.

## Key Features

* **High-Accuracy Semantic Search**: Uses sentence-transformer embeddings (`all-MiniLM-L6-v2`) to understand the nuance of project proposals beyond simple keywords.
* **High-Performance Retrieval**: Built with `faiss-cpu` for efficient, low-latency similarity search, making it scalable for thousands of topics.
* **Simple REST API**: A clean, lightweight Flask API for easy integration and programmatic access.
* **Proven Results**: The model is backed by a thorough evaluation, demonstrating its effectiveness.

## Results & Full Report

The matching model was evaluated against a test set, achieving excellent performance in retrieving the correct topic.

* **Top-1 Accuracy**: **89.0%**
* **Top-10 Accuracy**: **95.6%**
* **Mean Reciprocal Rank (MRR@10)**: **0.9148**

For a complete overview of the methodology, data scraping, exploratory data analysis, and evaluation, please see the full project report.

---

## Demo

Below is an example of the API in action. A user sends a project description, and the API returns the most relevant Horizon Europe topics, ranked by similarity.

*(**Note**: Consider adding a screenshot of your web application's UI here if you have one!)*

---

## API Usage

The API has one main endpoint: `/`. You can send a POST request with your project data in JSON format.

**Example Request (`curl`):**

```bash
curl -X POST [http://127.0.0.1:5001/](http://127.0.0.1:5001/) \
-H "Content-Type: application/json" \
-d '{
  "project_text": "My project is about developing new battery technologies for electric vehicles using sustainable materials to reduce environmental impact."
}'
```
**Example Response (JSON):**
```json
[
  {
    "topic_id": "HORIZON-CL5-2024-D2-01-01",
    "similarity_score": 0.9148,
    "topic_title": "Next-generation battery technologies...",
    "topic_url": "[https://ec.europa.eu/info/funding-tenders/opportunities/portal/screen/opportunities/topic-details/HORIZON-CL5-2024-D2-01-01](https://ec.europa.eu/info/funding-tenders/opportunities/portal/screen/opportunities/topic-details/HORIZON-CL5-2024-D2-01-01)"
  },
  {
    "topic_id": "HORIZON-CL5-2024-D2-01-03",
    "similarity_score": 0.8992,
    "topic_title": "Advanced materials and chemistries for batteries...",
    "topic_url": "[https://ec.europa.eu/info/funding-tenders/opportunities/portal/screen/opportunities/topic-details/HORIZON-CL5-2024-D2-01-03](https://ec.europa.eu/info/funding-tenders/opportunities/portal/screen/opportunities/topic-details/HORIZON-CL5-2024-D2-01-03)"
  }
]
```
---
## Installation & Setup

### Prerequisites
* **Python 3.8+** (recommended: Python 3.10 or newer)
* **pip**

### Setup (Linux / macOS / WSL)

#### 1. Prepare Virtual Environment

From the project root directory:

```bash
# Create a new virtual environment
python3 -m venv venv

# Activate it (Linux/macOS)
source venv/bin/activate

# Activate it (Windows PowerShell)
# .\venv\Scripts\Activate.ps1
```

You should now see (venv) in your terminal prompt.

** 2. Install Dependencies **
With the environment activated, install the required packages. Using the requirements.txt file is recommended for a one-step installation.

```bash
pip install -r requirements.txt
```
Alternatively, you can install them manually:
```bash
pip install flask numpy pandas faiss-cpu sentence-transformers
```
** 3. Prepare Topic Vectors **
Before starting the server, ensure your topic vector files (the FAISS index and the corresponding .npy file) are present in the project root directory. These files should be generated using the all-MiniLM-L6-v2 model.

** 4. Run the Application **
Start the Flask server:
```bash
python app.py
```
The API service will be running at http://127.0.0.1:5001/.

## Notes
Using the all-MiniLM-L6-v2 model resolves previous memory issues encountered with other models and ensures stable, efficient performance.

It is recommended to keep the FAISS index and .npy files in the project root or configure their path inside app.py.

## License
This project is licensed under the MIT License – feel free to use and adapt it.








