import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify, render_template
import gc 

# --- 1. Configuration ---
# IMPORTANT: Ensure this path is correct for your local environment
DATA_PATH = "data/processed" 
CSV_PATH = "data/processed/cleaned_project_data.csv"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2" # old model was: multi-qa-mpnet-base-dot-v1
TOPIC_ID_COL = "topics"
TOPIC_TEXT_COL = "topic_text"
EVAL_TOP_K = 10 

# Initialize the Flask App
app = Flask(__name__)

# Global variables for resources
model = None
index = None
df_topics = None
year_cols = []
topic_urls = {} 


# --- 2. Resource Loading (Executed once at startup) ---

def load_resources():
    global model, index, df_topics, year_cols, topic_urls

    print("Starting API resource initialization...")

    # Load the Sentence Transformer Model
    model = SentenceTransformer(EMBED_MODEL_NAME)
    print(f"Model '{EMBED_MODEL_NAME}' loaded.")

    # Load data and prepare topic information
    df = pd.read_csv(CSV_PATH)
    df_topics = df.groupby(TOPIC_ID_COL).first().reset_index()
    year_cols = [col for col in df.columns if col.startswith('year_')]
    print(f"Data loaded. Found {len(df_topics)} unique topics.")

    # Integrate URLs for each topic
    base_url = "https://ec.europa.eu/info/funding-tenders/opportunities/portal/screen/opportunities/topic-details/"
    # Using the parameters from your existing logic
    url_parameters = "?isExactMatch=true&status=31094501,31094502,31094503" 
    
    unique_topic_ids = df_topics[TOPIC_ID_COL].unique().tolist()

    topic_urls = {
        topic_id: f"{base_url}{topic_id}{url_parameters}"
        for topic_id in unique_topic_ids if isinstance(topic_id, str) and topic_id.strip()
    }
    print(f"Constructed {len(topic_urls)} URLs for unique topics.")
    
    # Load the 3 components for topic vectors from .npy files
    topic_text_embeddings = np.load(os.path.join(DATA_PATH, 'topic_text_embeddings.npy'))
    topic_keyword_embeddings = np.load(os.path.join(DATA_PATH, 'topic_keyword_embeddings.npy'))
    topic_year_vectors = df_topics[year_cols].values

    # Combine topic vectors
    topic_embeddings_combined = np.hstack([
        topic_text_embeddings,
        topic_keyword_embeddings,
        topic_year_vectors  
    ])
    
    # FIX for FAISS TypeError: Must cast array to float32 before normalization/add
    topic_embeddings_combined = topic_embeddings_combined.astype('float32')

    # Normalize vectors
    faiss.normalize_L2(topic_embeddings_combined)

    # Create and populate the FAISS Index
    index = faiss.IndexFlatIP(topic_embeddings_combined.shape[1])
    # The array is already float32, so no need to cast again
    index.add(topic_embeddings_combined) 
    print(f"FAISS Index created. Dimension: {index.d}, Entries: {index.ntotal}")
    
    # Optional: Free up memory to prevent OOM errors
    gc.collect() 
    
    print("API resources loaded successfully.")

# Load resources when the application starts
load_resources()


# --- 3. Reusable Search Logic (Refactored Core Function) ---

def perform_multimodal_search(project_text, keywords, year):
    """
    Performs vectorization and FAISS search.
    Returns a list of search results.
    """
    if not project_text:
        return []
    
    # Ensure the text is passed as a string
    combined_query_text = project_text
    
    # a) Text Embedding (384 Dim. for MiniLM)
    # IMPORTANT: batch_size=1 reduces RAM consumption
    project_text_embedding = model.encode(
        [combined_query_text], 
        show_progress_bar=False, 
        convert_to_numpy=True,
        batch_size=1 # Optimization for stability
    )

    # b) Keyword Embedding (384 Dim. for MiniLM)
    keywords_str = " ".join(keywords)
    project_keyword_embedding = model.encode(
        [keywords_str], 
        show_progress_bar=False, 
        convert_to_numpy=True,
        batch_size=1 # Optimization for stability
    )

    # c) Year Vector (e.g., 6 Dim.)
    year_vector = np.zeros((1, len(year_cols)), dtype=np.float32)
    if year:
        try:
            year_col = f'year_{int(year)}'
            if year_col in year_cols:
                year_vector[0, year_cols.index(year_col)] = 1
        except ValueError:
            pass # Ignore invalid year

    # d) Combine and normalize vectors
    # Order must match index creation: [Text | Keyword | Year]
    combined_query_vector = np.hstack([
        project_text_embedding,
        project_keyword_embedding,
        year_vector
    ])

    # IMPORTANT: Normalize the vector for Cosine Similarity (Inner Product)
    # The query vector must also be float32 for FAISS
    faiss.normalize_L2(combined_query_vector.astype('float32'))

    # e) FAISS Search
    # D = Distances (Scores), I = Index positions of Top-K Topics
    D, I = index.search(combined_query_vector.astype('float32'), EVAL_TOP_K)
    
    # f) Prepare results
    results = []
    for score, topic_index in zip(D[0], I[0]):
        topic_info = df_topics.iloc[topic_index]
        topic_id = topic_info[TOPIC_ID_COL] # Get the ID

        # Retrieve the URL from the global dictionary
        topic_url = topic_urls.get(topic_id, "#") # Default to '#' if URL is missing

        results.append({
            'topic_id': topic_id,
            'topic_text': topic_info[TOPIC_TEXT_COL],
            'similarity_score': float(score),
            'topic_url': topic_url # Add the URL
        })
        
    return results


# --- 4. Placeholder for LLM Explanation (DEACTIVATED) ---

# def generate_explanation(project_text, topic_text, topic_id):
#     """
#     LLM explanation function: requires external configuration (e.g., Gemini API key).
#     Uncomment the code block in the /search route to use this function.
#     """
#     # NOTE: You would need to import 'genai' client and implement the API call here.
#     return (
#         f"LLM explanation for Topic {topic_id} goes here. "
#         f"The analysis confirms a strong match based on keywords."
#     )


# --- 5. API Endpoints ---

# ENDPOINT 1: JSON API (For scripts, e.g., explain_api.py or curl)
@app.route('/search-topics', methods=['POST'])
def search_topics_json():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON format."}), 400

    project_text = data.get('project_text', '')
    keywords = data.get('keywords', [])
    year = data.get('year')

    if not project_text:
        return jsonify({'error': 'The project text (project_text) is required.'}), 400

    results = perform_multimodal_search(project_text, keywords, year)

    return jsonify(results)


# ENDPOINT 2: HTML Route (For browser testing and visualization)
@app.route('/search', methods=['POST'])
def search_topics_html():
    try:
        # Try to parse JSON (for tools like Postman/Insomnia)
        data = request.get_json(silent=True)
        # If JSON fails, try to parse form data (for a potential future HTML form)
        if data is None:
            data = request.form

    except Exception:
        # This error is only triggered if request.get_json() or request.form itself fails
        return "Error: Invalid input format.", 400
    
    project_text = data.get('project_text', '')
    
    # Convert keywords from string to list (for form submissions)
    keywords_raw = data.get('keywords', '')
    if isinstance(keywords_raw, str):
        keywords = [k.strip() for k in keywords_raw.split(',') if k.strip()]
    else:
        keywords = keywords_raw 
        
    year = data.get('year')

    if not project_text:
        return "Error: Project text is required.", 400

    results = perform_multimodal_search(project_text, keywords, year)
    
    # Initialize LLM variables
    topic_summary = None 
    match_analysis = None

    # --- LLM EXPLANATION BLOCK (DISABLED) ---
    # if results:
    #     top_result = results[0]
    #     # UNCOMMENT THE FOLLOWING LINE to activate the LLM analysis:
    #     # topic_summary, match_analysis = generate_explanation(
    #     #     project_text, top_result['topic_text'], top_result['topic_id']
    #     # )
        
    # Renders the HTML template
    return render_template(
        'results.html',
        results=results,
        project_text=project_text,
        year=year,
        keywords=keywords,
        topic_summary=topic_summary, # Pass for the new HTML structure
        match_analysis=match_analysis # Pass for the new HTML structure
    )

# ENDPOINT 3: Index page with the search form (GET request)
@app.route('/', methods=['GET'])
def show_index(): 
    """Renders the main search form."""
    return render_template('index.html')


# Start the Flask Server
if __name__ == '__main__':
    # Set debug=False for production
    app.run(host='0.0.0.0', port=5001, debug=True)