# import streamlit as st
# import helper
# import pickle
# import os
# import numpy as np
# import pandas as pd
# import logging

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# # Set page configuration
# st.set_page_config(
#     page_title="Quora Question Pairs",
#     page_icon="❓",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for better UI
# st.markdown("""
# <style>
# .main-header {
#     font-size: 2.5rem;
#     color: #1E88E5;
#     text-align: center;
#     margin-bottom: 2rem;
# }
# .sub-header {
#     font-size: 1.5rem;
#     color: #424242;
#     margin-bottom: 1rem;
# }
# .result-duplicate {
#     font-size: 2rem;
#     color: #D32F2F;
#     text-align: center;
#     padding: 1rem;
#     border-radius: 0.5rem;
#     background-color: #FFEBEE;
#     margin: 1rem 0;
# }
# .result-not-duplicate {
#     font-size: 2rem;
#     color: #388E3C;
#     text-align: center;
#     padding: 1rem;
#     border-radius: 0.5rem;
#     background-color: #E8F5E9;
#     margin: 1rem 0;
# }
# .confidence {
#     font-size: 1.2rem;
#     text-align: center;
#     margin-top: 0.5rem;
# }
# .feature-info {
#     font-size: 0.9rem;
#     color: #616161;
#     margin: 0.5rem 0;
# }
# .feature-container {
#     background-color: #F5F5F5;
#     color: #000000;
#     padding: 1rem;
#     border-radius: 0.5rem;
#     margin: 1rem 0;
# }
# .confidence-meter {
#     height: 2rem;
#     border-radius: 1rem;
#     margin: 1rem 0;
# }
# </style>
# """, unsafe_allow_html=True)

# # Function to load model and vectorizer
# @st.cache_resource
# def load_model():
#     try:
#         dir_path = os.path.dirname(os.path.realpath(__file__))
#         model = pickle.load(open(os.path.join(dir_path, 'model.pkl'),'rb'))
#         logger.info("Model loaded successfully")
#         return model
#     except Exception as e:
#         logger.error(f"Error loading model: {e}")
#         st.error(f"Error loading model: {e}")
#         return None

# def main():
#     """Main function to run the Streamlit app"""
#     # Load model
#     model = load_model()
#     if model is None:
#         st.stop()

#     # Ensure session_state keys exist
#     if 'q1' not in st.session_state:
#         st.session_state.q1 = " How can I learn Python programming?"
#     if 'q2' not in st.session_state:
#         st.session_state.q2 = "What is the best way to learn Python programming?"

#     # Sidebar with app information
#     with st.sidebar:
#         st.title("About this app")
#         st.markdown("""
#         This app uses a machine learning model to determine if two questions are duplicates.

#         The model was trained on the Quora Question Pairs dataset and uses various NLP features including:
#         - Text length features
#         - Word count features
#         - Common words analysis
#         - Fuzzy string matching
#         - Bag-of-Words representation

#         **How to use:**
#         1. Enter two questions in the text areas
#         2. Click 'Find' to check if they are duplicates
#         3. View the detailed analysis in the results section
#         """)

#         st.markdown("### Model Information")
#         st.markdown(f"**Model Type:** {type(model).__name__}")
#         if hasattr(model, 'n_estimators'):
#             st.markdown(f"**Number of Trees:** {model.n_estimators}")
#         if hasattr(model, 'max_depth') and model.max_depth is not None:
#             st.markdown(f"**Max Depth:** {model.max_depth}")

#     # Main content
#     st.markdown("<h1 class='main-header'>Duplicate Question Pairs Detector</h1>", unsafe_allow_html=True)
#     st.markdown("Enter two questions to check if they are duplicates of each other.")

#     # Connected input fields
#     st.text_area('Question 1', key='q1', placeholder="e.g., How do I lose weight fast?")
#     st.text_area('Question 2', key='q2', placeholder="e.g., What are the best ways to reduce weight quickly?")

#     # Process when button is clicked
#     if st.button('Find', type="primary"):
#         q1 = st.session_state.q1
#         q2 = st.session_state.q2

#         if not q1 or not q2:
#             st.warning("Please enter both questions to compare.")
#             st.stop()

#         try:
#             # Get prediction
#             query = helper.query_point_creator(q1, q2)
#             result = model.predict(query)[0]

#             # Get probability if available
#             confidence = None
#             if hasattr(model, 'predict_proba'):
#                 proba = model.predict_proba(query)[0]
#                 confidence = proba[1] if result else proba[0]

#             # Display result
#             if result:
#                 st.markdown("<div class='result-duplicate'>Duplicate Questions</div>", unsafe_allow_html=True)
#             else:
#                 st.markdown("<div class='result-not-duplicate'>Not Duplicate Questions</div>", unsafe_allow_html=True)

#             # Display confidence
#             if confidence is not None:
#                 st.markdown("<h2 class='sub-header'>Confidence Level</h2>", unsafe_allow_html=True)
#                 st.progress(float(confidence))
#                 st.markdown(f"The model is **{confidence*100:.2f}%** confident in this prediction.")

#             # Get and display features
#             st.markdown("<h2 class='sub-header'>Feature Analysis</h2>", unsafe_allow_html=True)
#             features = helper.get_basic_features(q1, q2)

#             col1, col2 = st.columns(2)

#             with col1:
#                 st.markdown("<h3 class='sub-header'>Text Statistics</h3>", unsafe_allow_html=True)
#                 stats_df = pd.DataFrame({
#                     'Metric': ['Length (Q1)', 'Length (Q2)', 'Word Count (Q1)', 'Word Count (Q2)'],
#                     'Value': [len(q1), len(q2), len(q1.split()), len(q2.split())]
#                 })
#                 st.dataframe(stats_df, hide_index=True)

#             with col2:
#                 st.markdown("<h3 class='sub-header'>Similarity Metrics</h3>", unsafe_allow_html=True)
#                 similarity_df = pd.DataFrame({
#                     'Metric': ['Common Words', 'Word Share', 'Fuzzy Ratio', 'Token Set Ratio'],
#                     'Value': [
#                         features['common_words'],
#                         f"{features['word_share']*100:.1f}%",
#                         f"{features['fuzzy_ratio']:.1f}%",
#                         f"{features['token_set_ratio']:.1f}%"
#                     ]
#                 })
#                 st.dataframe(similarity_df, hide_index=True)

#             # Display preprocessed text
#             st.markdown("<h3 class='sub-header'>Preprocessed Text</h3>", unsafe_allow_html=True)
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.markdown("**Question 1:**")
#                 st.markdown(f"<div class='feature-container'>{helper.preprocess(q1)}</div>", unsafe_allow_html=True)
#             with col2:
#                 st.markdown("**Question 2:**")
#                 st.markdown(f"<div class='feature-container'>{helper.preprocess(q2)}</div>", unsafe_allow_html=True)

#         except Exception as e:
#             logger.error(f"Error processing questions: {e}")
#             st.error(f"An error occurred: {e}")

# # Run the app
# if __name__ == "__main__":
#     main()
# app.py
# ======================================================================
# Quora Duplicate Question Detector — MINT ULTRA ✨
# Gorgeous UI (glassmorphism + animated gradient), robust fallbacks,
# zero-setup friendliness, exports, presets, and smooth UX.
# ======================================================================

import os
import time
import json
import pickle
import logging
import numpy as np
import pandas as pd
import streamlit as st
from urllib.parse import quote

# Import search functionality
try:
    from search_helper import QuestionSearcher
    SEARCH_AVAILABLE = True
except Exception as e:
    SEARCH_AVAILABLE = False
    logger.warning(f"Search functionality not available: {e}")

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("mint-ultra")

# -----------------------------------------------------------------------------
# Page Config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Quora Question Pairs • MINT ULTRA",
    page_icon="❓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# CSS — animated gradient, glass cards, glow, modern controls
# -----------------------------------------------------------------------------
st.markdown("""
<style>
/* Global background with animated nebula gradient */
.stApp {
  background: radial-gradient(1000px 600px at 12% 10%, rgba(127,255,212,.10), transparent 60%),
              radial-gradient(800px 500px at 85% 20%, rgba(0,170,255,.09), transparent 60%),
              linear-gradient(140deg, #0c0f1b 0%, #0a1226 38%, #060914 100%);
  background-attachment: fixed;
}

/* Hero Title gradient shimmer */
.title-hero {
  font-size: 2.6rem;
  font-weight: 900;
  letter-spacing: .6px;
  line-height: 1.1;
  background: linear-gradient(90deg, #87F6FF, #FEFF9C, #FFD166, #BCA0FF, #87F6FF);
  background-size: 300% 300%;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: flow 12s ease infinite;
}
@keyframes flow {
  0% {background-position: 0% 50%}
  50% {background-position: 100% 50%}
  100% {background-position: 0% 50%}
}

/* Glass card */
# .glass {
#   backdrop-filter: blur(14px);
#   -webkit-backdrop-filter: blur(14px);
#   background: rgba(255,255,255,0.06);
#   border: 1px solid rgba(255,255,255,0.14);
#   border-radius: 18px;
#   padding: 18px 22px;
#   box-shadow: 0 10px 30px rgba(0,0,0,.38), inset 0 0 0 1px rgba(255,255,255,.06);
# }

/* Subtle separator */
.hr {
  height: 1px;
  width: 100%;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,.18), transparent);
  margin: 14px 0 10px;
}

/* Modern buttons */
.stButton>button {
  border-radius: 14px;
  padding: .7rem 1.05rem;
  border: 1px solid rgba(255,255,255,.18);
  background: linear-gradient(180deg, rgba(255,255,255,.10), rgba(255,255,255,.06));
  box-shadow: 0 10px 20px rgba(0,0,0,.35), 0 0 12px rgba(127,255,212,.28);
  transition: transform .08s ease, box-shadow .22s ease;
}
.stButton>button:hover { transform: translateY(-1px); box-shadow: 0 14px 28px rgba(0,0,0,.45), 0 0 16px rgba(127,255,212,.38); }
.stButton>button:active { transform: translateY(0); }

/* Chips */
.chip {
  display:inline-block; padding:5px 12px; margin: 0 8px 8px 0;
  border-radius: 999px; border: 1px solid rgba(255,255,255,.18);
  background: rgba(255,255,255,.08);
  font-size:.9rem;
}

/* Metric glow */
[data-testid="stMetricValue"] { text-shadow: 0 0 12px rgba(122,252,255,.35); }

/* Result banners */
.result-ok {
  font-size: 1.8rem; text-align:center; padding: .9rem 1rem; border-radius: 14px;
  background: rgba(46, 125, 50, .16); border: 1px solid rgba(46,125,50,.38);
  color: #8CE99A; box-shadow: inset 0 0 0 1px rgba(255,255,255,.04);
}
.result-bad {
  font-size: 1.8rem; text-align:center; padding: .9rem 1rem; border-radius: 14px;
  background: rgba(211, 47, 47, .16); border: 1px solid rgba(211,47,47,.38);
  color: #FF8A80; box-shadow: inset 0 0 0 1px rgba(255,255,255,.04);
}

/* Dataframes: compact & clean */
.block-container .dataframe td, .block-container .dataframe th {
  padding: 6px 8px !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Robust helper import with fallbacks (NO external deps required)
# -----------------------------------------------------------------------------
class _SafeHelperFallback:
    """Zero-dependency helper replacement with sane approximations."""
    @staticmethod
    def _tokenize(s: str):
        return [t for t in "".join(ch.lower() if ch.isalnum() else " " for ch in s).split() if t]

    @classmethod
    def preprocess(cls, s: str):
        return " ".join(cls._tokenize(s))

    @classmethod
    def get_basic_features(cls, q1: str, q2: str):
        t1, t2 = cls._tokenize(q1), cls._tokenize(q2)
        s1, s2 = set(t1), set(t2)
        common = s1 & s2
        total = len(s1) + len(s2)
        word_share = (2 * len(common) / total) if total > 0 else 0.0
        # Simple ratios (avoid fuzzy dependencies)
        jaccard = (len(s1 & s2) / len(s1 | s2)) if (s1 | s2) else 0.0
        overlap = (len(common) / min(len(s1), len(s2))) if min(len(s1), len(s2)) > 0 else 0.0
        return {
            "common_words": len(common),
            "word_share": float(word_share),
            "jaccard": float(jaccard),
            "overlap": float(overlap),
            # Mock fuzzy-like signals
            "fuzzy_ratio": float(100 * (0.6 * jaccard + 0.4 * overlap)),
            "token_set_ratio": float(100 * (0.4 * jaccard + 0.6 * overlap)),
        }

    @classmethod
    def query_point_creator(cls, q1: str, q2: str):
        """Create a tiny numeric feature vector compatible with sklearn-like models."""
        feats = cls.get_basic_features(q1, q2)
        # A stable 6-feature vector; models trained with more features will ignore extra or fail.
        # We keep it minimal for robust infer-time fallback.
        vec = np.array([[len(q1), len(q2),
                         len(q1.split()), len(q2.split()),
                         feats["jaccard"], feats["word_share"]]], dtype=float)
        return vec

try:
    import helper as _helper  # your module, if present
    HELPER = _helper
    logger.info("Using provided helper.py")
except Exception as e:
    HELPER = _SafeHelperFallback
    logger.warning(f"helper.py not available or failed to import. Using safe fallback. Details: {e}")

# -----------------------------------------------------------------------------
# Model loader (with dummy model fallback)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model.pkl")
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            logger.info("Model loaded successfully.")
            return model
        except Exception as e:
            logger.error(f"Failed to load model.pkl: {e}")

    logger.warning("model.pkl not found or failed. Using DummyModel fallback.")

    class DummyModel:
        """A tiny heuristic model: predicts duplicate if Jaccard/overlap strong."""
        def predict(self, X):
            # X columns: [len1, len2, wc1, wc2, jaccard, word_share]
            # Heuristic threshold:
            j = float(X[0, 4]) if X.ndim == 2 else float(X[4])
            ws = float(X[0, 5]) if X.ndim == 2 else float(X[5])
            score = 0.65 * j + 0.35 * ws
            return np.array([1 if score >= 0.42 else 0])

        def predict_proba(self, X):
            j = float(X[0, 4]) if X.ndim == 2 else float(X[4])
            ws = float(X[0, 5]) if X.ndim == 2 else float(X[5])
            score = 0.65 * j + 0.35 * ws
            # squash to [0,1]
            conf = max(0.0, min(1.0, (score - 0.2) / 0.6))
            return np.array([[1.0 - conf, conf]])

        @property
        def n_estimators(self): return None
        @property
        def max_depth(self): return None

    return DummyModel()

model = load_model()

# -----------------------------------------------------------------------------
# Safe utilities
# -----------------------------------------------------------------------------
def _normalize_result(y) -> int:
    """Coerce model outputs to {0,1} (0=Not duplicate, 1=Duplicate)."""
    if isinstance(y, (list, tuple, np.ndarray)):
        y = y[0]
    if isinstance(y, (np.bool_, bool)):  # True/False
        return int(bool(y))
    try:
        yi = int(y)
        return 1 if yi != 0 else 0
    except Exception:
        s = str(y).strip().lower()
        if s in {"dup", "duplicate", "true", "yes", "1"}:
            return 1
        return 0

def _proba_positive(p):
    """Return positive-class probability if available, else None."""
    try:
        p = np.asarray(p)
        if p.ndim == 2 and p.shape[1] >= 2:
            return float(p[0, 1])
        # If binary single prob was returned, guess mapping >0.5 as positive
        if p.ndim == 1 and p.size >= 1:
            val = float(p[0])
            return val if 0 <= val <= 1 else None
    except Exception:
        pass
    return None

# -----------------------------------------------------------------------------
# Session defaults BEFORE widgets (prevents Streamlit key mutation errors)
# -----------------------------------------------------------------------------
if "q1" not in st.session_state:
    st.session_state.q1 = "How can I learn Python programming?"
if "q2" not in st.session_state:
    st.session_state.q2 = "What is the best way to learn Python programming?"

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="title-hero">❓ Quora Duplicate Detector</div>', unsafe_allow_html=True)
    st.caption("Polished UI • Robust fallbacks • No surprises")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.subheader("Model Information")
    st.write(f"**Model Type:** `{type(model).__name__}`")
    if hasattr(model, "n_estimators") and getattr(model, "n_estimators") is not None:
        st.write(f"**Trees:** {getattr(model, 'n_estimators')}")
    if hasattr(model, "max_depth") and getattr(model, "max_depth") is not None:
        st.write(f"**Max Depth:** {getattr(model, 'max_depth')}")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.subheader("Quick Presets")
    c1, c2 = st.columns(2)
    if c1.button("Similar 🧩"):
        st.session_state.q1 = "How can I learn Python programming?"
        st.session_state.q2 = "What is the best way to learn Python programming?"
        st.toast("Loaded a similar pair.", icon="🧩")
    if c2.button("Different ✂️"):
        st.session_state.q1 = "What is machine learning?"
        st.session_state.q2 = "How to cook pasta at home?"
        st.toast("Loaded a different pair.", icon="✂️")

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Header (shown on all tabs)
# -----------------------------------------------------------------------------
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.markdown('<div class="title-hero">MINT ULTRA • Quora Question Intelligence</div>', unsafe_allow_html=True)
st.caption("Search similar questions or check if two questions are duplicates. Powered by ML.")
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("")

# -----------------------------------------------------------------------------
# Load Question Searcher (cached) - must be before tabs
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_searcher(_max_questions=None, version=2):
    """Load and initialize the question searcher
    
    Note: _max_questions prefix tells Streamlit to cache based on this parameter
    version: Force cache reload
    """
    if not SEARCH_AVAILABLE:
        return None
    
    try:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        parent_dir = os.path.dirname(dir_path)
        dataset_path = os.path.join(parent_dir, 'train.csv')
        
        searcher = QuestionSearcher(dataset_path=dataset_path, max_questions=_max_questions)
        return searcher
    except Exception as e:
        logger.error(f"Error loading searcher: {e}")
        return None

# -----------------------------------------------------------------------------
# Tabs for extra UX
# -----------------------------------------------------------------------------
tabs = st.tabs(["🔍 Search Questions", "🔎 Duplicate Check", "🧠 Explain", "🧪 Playground", "ℹ️ About"])

# -----------------------------------------------------------------------------
# Tab 0: Question Search
# -----------------------------------------------------------------------------
with tabs[0]:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="title-hero">🔍 Find Similar Questions</div>', unsafe_allow_html=True)
    st.caption("Search through the Quora dataset to find questions similar to yours. Results update as you type!")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("")
    
    # Search configuration
    col_config1, col_config2, col_config3 = st.columns(3)
    with col_config1:
        max_results = st.slider("Max Results", 5, 50, 10, help="Maximum number of similar questions to return")
    with col_config2:
        min_similarity = st.slider("Min Similarity", 0.0, 1.0, 0.3, 0.05, help="Minimum similarity threshold (0-1)")
    with col_config3:
        max_questions_index = st.number_input("Index Size", 1000, 500000, 50000, 1000, 
                                            help="Number of questions to index (more = better results, slower load)")
    
    # Initialize searcher (cache key includes max_questions_index)
    if SEARCH_AVAILABLE:
        searcher = load_searcher(_max_questions=int(max_questions_index))
        
        if searcher is not None:
            stats = searcher.get_stats()
            st.info(f"📊 Indexed {stats.get('total_questions', 0):,} questions | Ready to search!")
            
            # Search input
            st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
            search_query = st.text_input(
                "🔎 Enter your question",
                placeholder="e.g., How can I learn Python programming?",
                key="search_query"
            )
            
            # Auto-search as user types (with debouncing via button)
            search_button = st.button("🔍 Search", type="primary", use_container_width=True)
            
            # Only search when button is clicked (prevents auto-search on every keystroke)
            if search_button:
                if search_query and len(search_query.strip()) > 2:
                    with st.spinner("Searching for similar questions..."):
                        start_time = time.perf_counter()
                        search_output = searcher.search(
                            search_query.strip(),
                            top_k=max_results,
                            min_similarity=min_similarity
                        )
                        search_time = (time.perf_counter() - start_time) * 1000
                        
                        # Handle new return format (dict) or old (list) just in case
                        if isinstance(search_output, dict):
                            results = search_output.get('results', [])
                            viz_data = search_output.get('visualization', None)
                        else:
                            results = search_output
                            viz_data = None
                    
                    if results:
                        st.success(f"Found {len(results)} similar questions in {search_time:.1f}ms")
                        
                        # ---------------------------------------------------------------------
                        # Visualization Section (Research Feature)
                        # ---------------------------------------------------------------------
                        if viz_data:
                            with st.expander("🗺️ Semantic Space Visualization (Research MVP)", expanded=True):
                                st.caption("Visualizing 2D projection of TF-IDF vectors (PCA/SVD). Red = Query, Blue = Results, Grey = Context.")
                                
                                # Prepare data for st.scatter_chart
                                # We need a DataFrame with x, y, color/type
                                points = []
                                
                                # Query Point
                                points.append({
                                    "x": viz_data['query_x'], 
                                    "y": viz_data['query_y'], 
                                    "Type": "🔴 Your Query",
                                    "Prob": 1.0
                                })
                                
                                # Results Points
                                for pt in viz_data['results']:
                                    points.append({
                                        "x": pt['x'], 
                                        "y": pt['y'], 
                                        "Type": "🔵 Result",
                                        "Prob": 0.8
                                    })
                                    
                                # Background Points
                                for pt in viz_data['background']:
                                    points.append({
                                        "x": pt['x'], 
                                        "y": pt['y'], 
                                        "Type": "⚪ Context",
                                        "Prob": 0.2
                                    })
                                
                                df_viz = pd.DataFrame(points)
                                
                                # Use Streamlit's native scatter chart (simplified)
                                st.scatter_chart(
                                    df_viz,
                                    x='x',
                                    y='y',
                                    color='Type',
                                    size='Prob', # Hack to make query/results larger
                                    height=400
                                )
                        
                        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

                        # Display results
                        for idx, result in enumerate(results, 1):
                            with st.container():
                                st.markdown(f"### {idx}. {result.get('score_type', 'Similarity')}: {result['similarity_percent']:.1f}%")
                                
                                # Question text
                                st.markdown(f"**Question:** {result['question']}")
                                
                                # Link and metadata
                                col_link, col_sim = st.columns([3, 1])
                                with col_link:
                                    if result['link']:
                                        st.markdown(f"[🔍 Search on Quora]({result['link']})")
                                    else:
                                        st.caption("Search link not available")
                                with col_sim:
                                    st.metric("Match", f"{result['similarity_percent']:.1f}%")
                                
                                # Similarity bar
                                st.progress(result['similarity'])
                                st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
                        
                        # Fallback option
                        st.markdown("""
                        <div style="text-align: center; margin-top: 20px;">
                            <p style="color: #666; font-size: 0.9em;">Not finding what you're looking for?</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        quora_search_url = f"https://www.quora.com/search?q={quote(search_query)}"
                        st.link_button("🌐 Search YOUR Question on Quora", quora_search_url, use_container_width=True)
                        
                        # Export results
                        export_data = {
                            "query": search_query,
                            "results_count": len(results),
                            "search_time_ms": round(search_time, 2),
                            "results": [
                                {
                                    "question": r['question'],
                                    "qid": r['qid'],
                                    "similarity": r['similarity'],
                                    "link": r['link']
                                }
                                for r in results
                            ]
                        }
                        st.download_button(
                            "⬇️ Download Results (JSON)",
                            data=json.dumps(export_data, indent=2),
                            file_name=f"search_results_{int(time.time())}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    else:
                        st.warning("No similar questions found locally.")
                        st.info("The local dataset might cover older topics. Try searching directly on Quora:")
                        quora_search_url = f"https://www.quora.com/search?q={quote(search_query)}"
                        st.link_button("🌐 Search YOUR Question on Quora", quora_search_url, use_container_width=True)
                else:
                    st.info("👆 Enter a question above and click Search to find similar questions!")
        else:
            st.error("⚠️ Search index not available. Please ensure train.csv exists in the parent directory.")
    else:
        st.warning("⚠️ Search functionality is not available. Please check that search_helper.py is properly configured.")

# -----------------------------------------------------------------------------
# Tab 1: Duplicate Check (existing functionality)
# -----------------------------------------------------------------------------
with tabs[1]:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="title-hero">🔎 Duplicate Question Pairs Detector</div>', unsafe_allow_html=True)
    st.caption("Enter two questions. We'll predict whether they're duplicates, show confidence, and break down similarity signals.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("")
    
    # Layout
    left, right = st.columns([1.2, 1])
    
    with left:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("Input")
        st.caption("Type or paste two questions below. Use presets from the sidebar for quick testing.")

        q1 = st.text_area("Question 1", key="q1", height=100, placeholder="e.g., How do I lose weight fast?")
        q2 = st.text_area("Question 2", key="q2", height=100, placeholder="e.g., What are the best ways to reduce weight quickly?")

        run = st.button("🔎 Find", use_container_width=True)

        if run:
            if not q1.strip() or not q2.strip():
                st.warning("Please enter **both** questions.")
                st.stop()

            start = time.perf_counter()
            with st.spinner("Analyzing…"):
                feats = HELPER.get_basic_features(q1, q2)
                query = HELPER.query_point_creator(q1, q2)

                try:
                    raw_pred = model.predict(query)
                except Exception as e:
                    logger.error(f"model.predict failed: {e}")
                    st.error(f"Model prediction failed: {e}")
                    st.stop()

                result = _normalize_result(raw_pred)
                prob = None
                try:
                    prob = _proba_positive(model.predict_proba(query))
                except Exception:
                    prob = None

                # If no probas, derive a soft confidence from features
                if prob is None:
                    conf_soft = max(0.0, min(1.0, 0.65 * feats["jaccard"] + 0.35 * feats["word_share"]))
                    prob = float(conf_soft)

                latency_ms = (time.perf_counter() - start) * 1000.0

            # Result banner
            if result == 1:
                st.markdown('<div class="result-bad">Duplicate Questions</div>', unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown('<div class="result-ok">Not Duplicate Questions</div>', unsafe_allow_html=True)
                st.snow()

            cA, cB, cC = st.columns(3)
            cA.metric("Confidence", f"{prob*100:.1f}%")
            cB.metric("Decision", "Duplicate 🧩" if result == 1 else "Not Duplicate ✂️")
            cC.metric("Latency", f"{latency_ms:.1f} ms")

            # Export
            export = {
                "q1": q1, "q2": q2,
                "prediction": "duplicate" if result == 1 else "not_duplicate",
                "confidence": round(prob, 4),
                "features": {k: float(v) for k, v in feats.items()},
                "latency_ms": round(latency_ms, 2),
            }
            st.download_button(
                "⬇️ Download result (.json)",
                data=json.dumps(export, indent=2),
                file_name="duplicate_result.json",
                mime="application/json",
                use_container_width=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("Signals & Evidence")
        st.caption("We compute similarity features to explain the prediction.")

        if "q1" in st.session_state and "q2" in st.session_state:
            feats_now = HELPER.get_basic_features(st.session_state.q1, st.session_state.q2)

            # Summary chips
            st.write("**Quick glance**")
            chips = [
                f'<span class="chip">Common Words: <b>{feats_now.get("common_words", 0)}</b></span>',
                f'<span class="chip">Word Share: <b>{feats_now.get("word_share", 0.0):.2f}</b></span>',
                f'<span class="chip">Jaccard: <b>{feats_now.get("jaccard", 0.0):.2f}</b></span>',
                f'<span class="chip">Overlap: <b>{feats_now.get("overlap", 0.0):.2f}</b></span>',
            ]
            st.markdown("".join(chips), unsafe_allow_html=True)

            st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.write("**Text Statistics**")
                df_stats = pd.DataFrame({
                    "Metric": ["Length (Q1)", "Length (Q2)", "Word Count (Q1)", "Word Count (Q2)"],
                    "Value": [len(st.session_state.q1), len(st.session_state.q2),
                              len(st.session_state.q1.split()), len(st.session_state.q2.split())]
                })
                st.dataframe(df_stats, hide_index=True, use_container_width=True)

            with c2:
                st.write("**Similarity Metrics**")
                df_sim = pd.DataFrame({
                    "Metric": ["Common Words", "Word Share", "Jaccard", "Overlap", "Fuzzy-like Ratio", "Token-Set-like"],
                    "Value": [
                        str(feats_now.get("common_words", 0)),
                        f"{feats_now.get('word_share', 0.0):.3f}",
                        f"{feats_now.get('jaccard', 0.0):.3f}",
                        f"{feats_now.get('overlap', 0.0):.3f}",
                        f"{feats_now.get('fuzzy_ratio', 0.0):.1f}%",
                        f"{feats_now.get('token_set_ratio', 0.0):.1f}%"
                    ]
                })
                st.dataframe(df_sim, hide_index=True, use_container_width=True)

            st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
            st.write("**Preprocessed Text (normalized tokens)**")
            c3, c4 = st.columns(2)
            with c3:
                st.caption("Question 1")
                st.code(HELPER.preprocess(st.session_state.q1))
            with c4:
                st.caption("Question 2")
                st.code(HELPER.preprocess(st.session_state.q2))

        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Tab 2: Explain
# -----------------------------------------------------------------------------
with tabs[2]:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("Explanation")
    st.write("""
This app predicts whether two questions are duplicates and helps you find similar questions.

**Duplicate Detection:**
- **Core idea:** similar meaning → high lexical overlap, similar tokens, similar phrasing.
- **Features:** length & word counts, common words, Jaccard similarity, word share, and overlap.
- **Model:** If `model.pkl` is present, we use it. If not, a safe heuristic model kicks in.
- **Confidence:** taken from `predict_proba` if available; otherwise we synthesize a soft score from features.

**Question Search:**
- **How it works:** Uses feature-based similarity matching to find questions similar to your query.
- **Indexing:** Pre-processes and indexes questions from the Quora dataset for fast retrieval.
- **Similarity:** Uses cosine similarity on feature vectors extracted using the same preprocessing as the model.
- **Performance:** Optimized for real-time search with configurable index size.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Tab 3: Playground
# -----------------------------------------------------------------------------
with tabs[3]:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("Playground Presets")
    st.caption("Click to inject sample pairs and then press **Find** in the Duplicate Check tab.")
    a1, a2, a3 = st.columns(3)
    if a1.button("Learning 🔁"):
        st.session_state.q1 = "How do I learn data science quickly?"
        st.session_state.q2 = "What is the fastest way to learn data science?"
        st.toast("Preset loaded.", icon="🔁")
    if a2.button("Math vs Food 🍝"):
        st.session_state.q1 = "What is the derivative of sin(x)?"
        st.session_state.q2 = "How do I make Alfredo pasta sauce?"
        st.toast("Preset loaded.", icon="🍝")
    if a3.button("Same Topic, Different Words 📚"):
        st.session_state.q1 = "How can I prepare for the GRE in two months?"
        st.session_state.q2 = "Best two-month GRE study plan?"
        st.toast("Preset loaded.", icon="📚")
    st.info("Go to the **Duplicate Check** tab to use these presets.")
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Tab 4: About
# -----------------------------------------------------------------------------
with tabs[4]:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("About MINT ULTRA")
    st.write("""
- **Design:** Animated gradients + glassmorphism + micro-interactions.
- **DX:** Works even without `model.pkl` or `helper.py` (sane fallbacks).
- **Performance:** Cached model load; minimal dependencies.
- **Export:** Download the full result JSON after each prediction.
- **Search:** Real-time question search with similarity matching.
    """)
    st.caption("Built with ❤️ using Streamlit.")
    st.markdown("</div>", unsafe_allow_html=True)
# '''
# # app.py — Quora Duplicate Question Detector (MINT ULTRA, presets fixed)

# import os
# import time
# import json
# import pickle
# import logging
# import numpy as np
# import pandas as pd
# import streamlit as st

# # --------------------------- Setup ---------------------------
# logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
# logger = logging.getLogger("mint-ultra")

# st.set_page_config(
#     page_title="Quora Question Pairs • MINT ULTRA",
#     page_icon="❓",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # --------------------------- Styles --------------------------
# st.markdown("""
# <style>
# .stApp {
#   background: radial-gradient(1000px 600px at 12% 10%, rgba(127,255,212,.10), transparent 60%),
#               radial-gradient(800px 500px at 85% 20%, rgba(0,170,255,.09), transparent 60%),
#               linear-gradient(140deg, #0c0f1b 0%, #0a1226 38%, #060914 100%);
#   background-attachment: fixed;
# }
# .title-hero {
#   font-size: 2.6rem; font-weight: 900; letter-spacing: .6px; line-height: 1.1;
#   background: linear-gradient(90deg, #87F6FF, #FEFF9C, #FFD166, #BCA0FF, #87F6FF);
#   background-size: 300% 300%;
#   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
#   animation: flow 12s ease infinite;
# }
# @keyframes flow { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
# .glass {
#   backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
#   background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.14);
#   border-radius: 18px; padding: 18px 22px; box-shadow: 0 10px 30px rgba(0,0,0,.38), inset 0 0 0 1px rgba(255,255,255,.06);
# }
# .hr { height: 1px; width: 100%; background: linear-gradient(90deg, transparent, rgba(255,255,255,.18), transparent); margin: 14px 0 10px; }
# .stButton>button {
#   border-radius: 14px; padding:.7rem 1.05rem; border:1px solid rgba(255,255,255,.18);
#   background: linear-gradient(180deg, rgba(255,255,255,.10), rgba(255,255,255,.06));
#   box-shadow: 0 10px 20px rgba(0,0,0,.35), 0 0 12px rgba(127,255,212,.28);
#   transition: transform .08s ease, box-shadow .22s ease;
# }
# .stButton>button:hover { transform: translateY(-1px); box-shadow: 0 14px 28px rgba(0,0,0,.45), 0 0 16px rgba(127,255,212,.38); }
# .stButton>button:active { transform: translateY(0); }
# .chip { display:inline-block; padding:5px 12px; margin:0 8px 8px 0; border-radius: 999px; border:1px solid rgba(255,255,255,.18); background: rgba(255,255,255,.08); font-size:.9rem; }
# [data-testid="stMetricValue"] { text-shadow: 0 0 12px rgba(122,252,255,.35); }
# .result-ok { font-size:1.8rem; text-align:center; padding:.9rem 1rem; border-radius:14px; background: rgba(46,125,50,.16); border:1px solid rgba(46,125,50,.38); color:#8CE99A; box-shadow: inset 0 0 0 1px rgba(255,255,255,.04); }
# .result-bad { font-size:1.8rem; text-align:center; padding:.9rem 1rem; border-radius:14px; background: rgba(211,47,47,.16); border:1px solid rgba(211,47,47,.38); color:#FF8A80; box-shadow: inset 0 0 0 1px rgba(255,255,255,.04); }
# .block-container .dataframe td, .block-container .dataframe th { padding: 6px 8px !important; }
# </style>
# """, unsafe_allow_html=True)

# # ------------------ Helper (fallback if missing) ---------------
# class _SafeHelperFallback:
#     @staticmethod
#     def _tokenize(s: str):
#         return [t for t in "".join(ch.lower() if ch.isalnum() else " " for ch in s).split() if t]
#     @classmethod
#     def preprocess(cls, s: str): return " ".join(cls._tokenize(s))
#     @classmethod
#     def get_basic_features(cls, q1: str, q2: str):
#         t1, t2 = cls._tokenize(q1), cls._tokenize(q2)
#         s1, s2 = set(t1), set(t2)
#         common = s1 & s2
#         total = len(s1) + len(s2)
#         word_share = (2*len(common)/total) if total>0 else 0.0
#         jaccard = (len(s1&s2)/len(s1|s2)) if (s1|s2) else 0.0
#         overlap = (len(common)/min(len(s1),len(s2))) if min(len(s1),len(s2))>0 else 0.0
#         return {
#             "common_words": len(common),
#             "word_share": float(word_share),
#             "jaccard": float(jaccard),
#             "overlap": float(overlap),
#             "fuzzy_ratio": float(100*(0.6*jaccard+0.4*overlap)),
#             "token_set_ratio": float(100*(0.4*jaccard+0.6*overlap)),
#         }
#     @classmethod
#     def query_point_creator(cls, q1: str, q2: str):
#         f = cls.get_basic_features(q1, q2)
#         return np.array([[len(q1), len(q2), len(q1.split()), len(q2.split()), f["jaccard"], f["word_share"]]], dtype=float)

# try:
#     import helper as _helper
#     HELPER = _helper
#     logger.info("Using provided helper.py")
# except Exception as e:
#     HELPER = _SafeHelperFallback
#     logger.warning(f"helper.py not available; using fallback. {e}")

# # -------------------------- Model ----------------------------
# @st.cache_resource(show_spinner=False)
# def load_model():
#     path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model.pkl")
#     if os.path.exists(path):
#         try:
#             with open(path, "rb") as f: return pickle.load(f)
#         except Exception as e:
#             logger.error(f"Failed to load model.pkl: {e}")
#     class DummyModel:
#         def predict(self, X):
#             j, ws = float(X[0,4]), float(X[0,5])
#             return np.array([1 if 0.65*j + 0.35*ws >= 0.42 else 0])
#         def predict_proba(self, X):
#             j, ws = float(X[0,4]), float(X[0,5])
#             score = 0.65*j + 0.35*ws
#             conf = max(0.0, min(1.0, (score - 0.2) / 0.6))
#             return np.array([[1.0 - conf, conf]])
#         n_estimators, max_depth = None, None
#     return DummyModel()
# model = load_model()

# def _normalize_result(y):
#     if isinstance(y, (list, tuple, np.ndarray)): y = y[0]
#     if isinstance(y, (np.bool_, bool)): return int(bool(y))
#     try: return 1 if int(y) != 0 else 0
#     except Exception: return 1 if str(y).strip().lower() in {"dup","duplicate","true","yes","1"} else 0

# def _proba_positive(p):
#     try:
#         p = np.asarray(p)
#         if p.ndim == 2 and p.shape[1] >= 2: return float(p[0,1])
#         if p.ndim == 1 and p.size >= 1 and 0 <= float(p[0]) <= 1: return float(p[0])
#     except Exception: pass
#     return None

# # --------- Defaults + queued preset application (BEFORE widgets) ----------
# st.session_state.setdefault("q1", "How can I learn Python programming?")
# st.session_state.setdefault("q2", "What is the best way to learn Python programming?")

# # If a preset was queued by a button callback, apply it now (safe).
# if st.session_state.get("_apply_preset", False):
#     st.session_state["q1"] = st.session_state.get("_pending_q1", st.session_state["q1"])
#     st.session_state["q2"] = st.session_state.get("_pending_q2", st.session_state["q2"])
#     # clear flags
#     for k in ("_apply_preset", "_pending_q1", "_pending_q2"):
#         if k in st.session_state: del st.session_state[k]

# # Queue function used by buttons (avoids session_state mutation-after-instantiation error)
# def queue_preset(q1, q2):
#     st.session_state["_pending_q1"] = q1
#     st.session_state["_pending_q2"] = q2
#     st.session_state["_apply_preset"] = True
#     st.rerun()

# # -------------------------- Sidebar -------------------------
# with st.sidebar:
#     st.markdown('<div class="glass">', unsafe_allow_html=True)
#     st.markdown('<div class="title-hero">❓ Quora Duplicate Detector</div>', unsafe_allow_html=True)
#     st.caption("Polished UI • Robust fallbacks • No surprises")
#     st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

#     st.subheader("Model Information")
#     st.write(f"**Model Type:** `{type(model).__name__}`")
#     if getattr(model, "n_estimators", None) is not None: st.write(f"**Trees:** {getattr(model, 'n_estimators')}")
#     if getattr(model, "max_depth", None) is not None: st.write(f"**Max Depth:** {getattr(model, 'max_depth')}")

#     st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
#     st.subheader("Quick Presets")
#     c1, c2 = st.columns(2)
#     c1.button("Similar 🧩", on_click=queue_preset,
#               args=("How can I learn Python programming?", "What is the best way to learn Python programming?"))
#     c2.button("Different ✂️", on_click=queue_preset,
#               args=("What is machine learning?", "How to cook pasta at home?"))
#     st.markdown("</div>", unsafe_allow_html=True)

# # -------------------------- Header --------------------------
# st.markdown('<div class="glass">', unsafe_allow_html=True)
# st.markdown('<div class="title-hero">MINT ULTRA • Duplicate Question Pairs Detector</div>', unsafe_allow_html=True)
# st.caption("Enter two questions. We’ll predict whether they’re duplicates, show confidence, and break down similarity signals.")
# st.markdown("</div>", unsafe_allow_html=True)
# st.markdown("")

# # -------------------------- Layout --------------------------
# left, right = st.columns([1.2, 1])

# with left:
#     st.markdown('<div class="glass">', unsafe_allow_html=True)
#     st.subheader("Input")
#     st.caption("Type or paste two questions below. Use presets from the sidebar or the Playground tab.")

#     q1 = st.text_area("Question 1", key="q1", height=100, placeholder="e.g., How do I lose weight fast?")
#     q2 = st.text_area("Question 2", key="q2", height=100, placeholder="e.g., What are the best ways to reduce weight quickly?")

#     run = st.button("🔎 Find", use_container_width=True)

#     if run:
#         if not q1.strip() or not q2.strip():
#             st.warning("Please enter **both** questions.")
#             st.stop()

#         start = time.perf_counter()
#         with st.spinner("Analyzing…"):
#             feats = HELPER.get_basic_features(q1, q2)
#             query = HELPER.query_point_creator(q1, q2)
#             raw_pred = model.predict(query)
#             result = _normalize_result(raw_pred)
#             prob = _proba_positive(getattr(model, "predict_proba", lambda *_: None)(query))
#             if prob is None:
#                 prob = float(max(0.0, min(1.0, 0.65*feats.get("jaccard",0.0) + 0.35*feats.get("word_share",0.0))))
#             latency_ms = (time.perf_counter() - start) * 1000.0

#         st.markdown('<div class="result-bad">Duplicate Questions</div>' if result == 1
#                     else '<div class="result-ok">Not Duplicate Questions</div>', unsafe_allow_html=True)
#         (st.balloons() if result == 1 else st.snow())

#         cA, cB, cC = st.columns(3)
#         cA.metric("Confidence", f"{prob*100:.1f}%")
#         cB.metric("Decision", "Duplicate 🧩" if result == 1 else "Not Duplicate ✂️")
#         cC.metric("Latency", f"{latency_ms:.1f} ms")

#         export = {
#             "q1": q1, "q2": q2,
#             "prediction": "duplicate" if result == 1 else "not_duplicate",
#             "confidence": round(prob, 4),
#             "features": {k: float(v) for k, v in feats.items()},
#             "latency_ms": round(latency_ms, 2),
#         }
#         st.download_button("⬇️ Download result (.json)", data=json.dumps(export, indent=2),
#                            file_name="duplicate_result.json", mime="application/json", use_container_width=True)

#     st.markdown("</div>", unsafe_allow_html=True)

# with right:
#     st.markdown('<div class="glass">', unsafe_allow_html=True)
#     st.subheader("Signals & Evidence")
#     st.caption("Similarity features that drive the decision.")

#     feats_now = HELPER.get_basic_features(st.session_state.q1, st.session_state.q2)
#     st.write("**Quick glance**")
#     chips = [
#         f'<span class="chip">Common Words: <b>{feats_now.get("common_words", 0)}</b></span>',
#         f'<span class="chip">Word Share: <b>{feats_now.get("word_share", 0.0):.2f}</b></span>',
#         f'<span class="chip">Jaccard: <b>{feats_now.get("jaccard", 0.0):.2f}</b></span>',
#         f'<span class="chip">Overlap: <b>{feats_now.get("overlap", 0.0):.2f}</b></span>',
#     ]
#     st.markdown("".join(chips), unsafe_allow_html=True)

#     st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
#     c1, c2 = st.columns(2)
#     with c1:
#         st.write("**Text Statistics**")
#         st.dataframe(pd.DataFrame({
#             "Metric": ["Length (Q1)", "Length (Q2)", "Word Count (Q1)", "Word Count (Q2)"],
#             "Value": [len(st.session_state.q1), len(st.session_state.q2),
#                       len(st.session_state.q1.split()), len(st.session_state.q2.split())]
#         }), hide_index=True, use_container_width=True)
#     with c2:
#         st.write("**Similarity Metrics**")
#         st.dataframe(pd.DataFrame({
#             "Metric": ["Common Words", "Word Share", "Jaccard", "Overlap", "Fuzzy-like Ratio", "Token-Set-like"],
#             "Value": [
#                 feats_now.get("common_words", 0),
#                 round(feats_now.get("word_share", 0.0), 3),
#                 round(feats_now.get("jaccard", 0.0), 3),
#                 round(feats_now.get("overlap", 0.0), 3),
#                 f"{feats_now.get('fuzzy_ratio', 0.0):.1f}%",
#                 f"{feats_now.get('token_set_ratio', 0.0):.1f}%"
#             ]
#         }), hide_index=True, use_container_width=True)

#     st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
#     st.write("**Preprocessed Text (normalized tokens)**")
#     c3, c4 = st.columns(2)
#     with c3: st.caption("Question 1"); st.code(HELPER.preprocess(st.session_state.q1))
#     with c4: st.caption("Question 2"); st.code(HELPER.preprocess(st.session_state.q2))
#     st.markdown("</div>", unsafe_allow_html=True)

# # -------------------------- Tabs ----------------------------
# tabs = st.tabs(["🧠 Explain", "🧪 Playground", "ℹ️ About"])
# with tabs[0]:
#     st.markdown('<div class="glass">', unsafe_allow_html=True)
#     st.subheader("Explanation")
#     st.write("""
# - Similar meaning → higher lexical overlap and token similarity.
# - Features: length & word counts, common words, Jaccard, word share, overlap.
# - If no `model.pkl`, a safe heuristic model is used; confidence from `predict_proba` or a soft score.
#     """)
#     st.markdown("</div>", unsafe_allow_html=True)

# with tabs[1]:
#     st.markdown('<div class="glass">', unsafe_allow_html=True)
#     st.subheader("Playground Presets")
#     st.caption("Click to set sample pairs, then press **Find**.")
#     a1, a2, a3 = st.columns(3)
#     a1.button("Learning 🔁", on_click=queue_preset,
#               args=("How do I learn data science quickly?", "What is the fastest way to learn data science?"))
#     a2.button("Math vs Food 🍝", on_click=queue_preset,
#               args=("What is the derivative of sin(x)?", "How do I make Alfredo pasta sauce?"))
#     a3.button("Same Topic, Different Words 📚", on_click=queue_preset,
#               args=("How can I prepare for the GRE in two months?", "Best two-month GRE study plan?"))
#     st.info("Presets will populate the inputs above automatically.")
#     st.markdown("</div>", unsafe_allow_html=True)

# with tabs[2]:
#     st.markdown('<div class="glass">', unsafe_allow_html=True)
#     st.subheader("About MINT ULTRA")
#     st.write("""
# - **Design:** Animated gradients + glassmorphism + micro-interactions.
# - **DX:** Works even without `model.pkl`/`helper.py`.
# - **Export:** Download full result JSON after each prediction.
#     """)
#     st.caption("Built with ❤️ using Streamlit.")
#     st.markdown("</div>", unsafe_allow_html=True)

# '''