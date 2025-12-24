"""
Question Search Module - Finds similar questions from the dataset
Uses the existing model's feature extraction for similarity matching
"""
import os
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import logging
import re
from urllib.parse import quote
from sklearn.metrics.pairwise import cosine_similarity
import helper

logger = logging.getLogger(__name__)

class QuestionSearcher:
    """Efficient question search using feature-based similarity"""
    
    def __init__(self, dataset_path: str = None, max_questions: int = None):
        """
        Initialize the question searcher
        
        Args:
            dataset_path: Path to train.csv
            max_questions: Maximum number of questions to index (None = all)
        """
        self.dataset_path = dataset_path
        self.max_questions = max_questions
        self.questions_df = None
        self.question_features = None
        self.cv = None
        self.model = None
        self._load_resources()
        self._build_index()
    
    def _load_resources(self):
        """Load model, vectorizer, and other resources"""
        dir_path = os.path.dirname(os.path.realpath(__file__))
        
        # Load CountVectorizer
        try:
            with open(os.path.join(dir_path, 'cv.pkl'), 'rb') as f:
                self.cv = pickle.load(f)
            logger.info("CountVectorizer loaded successfully")
        except Exception as e:
            logger.error(f"Error loading CountVectorizer: {e}")
            self.cv = None
        
        # Load model (optional, for feature extraction)
        try:
            with open(os.path.join(dir_path, 'model.pkl'), 'rb') as f:
                self.model = pickle.load(f)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.warning(f"Model not loaded: {e}")
            self.model = None
    
    def _build_index(self):
        """Build searchable index of questions from dataset"""
        if self.dataset_path is None:
            # Try to find train.csv in parent directory
            dir_path = os.path.dirname(os.path.realpath(__file__))
            parent_dir = os.path.dirname(dir_path)
            self.dataset_path = os.path.join(parent_dir, 'train.csv')
        
        if not os.path.exists(self.dataset_path):
            logger.warning(f"Dataset not found at {self.dataset_path}. Search will use limited functionality.")
            return
        
        try:
            logger.info(f"Loading dataset from {self.dataset_path}...")
            df = pd.read_csv(self.dataset_path)
            
            # Extract all unique questions
            questions_set = set()
            questions_list = []
            
            # Collect all questions with their IDs
            for _, row in df.iterrows():
                q1 = str(row['question1']).strip()
                q2 = str(row['question2']).strip()
                qid1 = row.get('qid1', None)
                qid2 = row.get('qid2', None)
                
                if q1 and q1 not in questions_set:
                    questions_set.add(q1)
                    questions_list.append({
                        'question': q1,
                        'qid': qid1,
                        'original_index': len(questions_list)
                    })
                
                if q2 and q2 not in questions_set:
                    questions_set.add(q2)
                    questions_list.append({
                        'question': q2,
                        'qid': qid2,
                        'original_index': len(questions_list)
                    })
                
                # Limit if specified
                if self.max_questions and len(questions_list) >= self.max_questions:
                    break
            
            self.questions_df = pd.DataFrame(questions_list)
            logger.info(f"Indexed {len(self.questions_df)} unique questions")
            
            # Pre-compute features for all questions
            logger.info("Pre-computing features for all questions...")
            self._precompute_features()
            
        except Exception as e:
            logger.error(f"Error building index: {e}")
            self.questions_df = None
    
    def _precompute_features(self):
        """Pre-compute features for all indexed questions using TF-IDF"""
        if self.questions_df is None or len(self.questions_df) == 0:
            return
        
        try:
            # Gather all questions text
            questions_text = self.questions_df['question'].apply(lambda x: helper.preprocess(str(x))).tolist()
            
            # Initialize and fit TF-IDF Vectorizer
            # This is better than simple CountVectorizer as it downweights common words
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
            
            logger.info("Fitting TF-IDF vectorizer...")
            self.question_features = self.tfidf.fit_transform(questions_text)
            
            # Initialize SVD for 2D visualization (Semantic Space)
            from sklearn.decomposition import TruncatedSVD
            self.svd = TruncatedSVD(n_components=2, random_state=42)
            logger.info("Fitting SVD for visualization...")
            self.svd_2d = self.svd.fit_transform(self.question_features)
            
            logger.info(f"Pre-computed features shape: {self.question_features.shape}")
            
        except Exception as e:
            logger.error(f"Error pre-computing features: {e}")
            self.question_features = None
    
    def _extract_query_features(self, query: str):
        """Extract features for a query question"""
        try:
            q_processed = helper.preprocess(query)
            if self.tfidf is not None:
                return self.tfidf.transform([q_processed])
            return None
        except Exception as e:
            logger.error(f"Error extracting query features: {e}")
            return None
    
    def _compute_similarity_scores(self, query_features) -> np.ndarray:
        """Compute similarity scores between query and all indexed questions"""
        if self.question_features is None:
            return np.array([])
        
        try:
            # Compute cosine similarity (TF-IDF vectors are usually normalized, but we use sklearn's cosine_similarity which handles it)
            similarities = cosine_similarity(query_features, self.question_features)[0]
            return similarities
        except Exception as e:
            logger.error(f"Error computing similarities: {e}")
            return np.array([])
    
    def get_visualization_data(self, query: str, results_indices: List[int]) -> Dict:
        """
        Get 2D coordinates for the query and results for visualization.
        Returns data for a scatter plot.
        """
        if self.svd is None or self.tfidf is None:
            return None
            
        try:
            # Get 2D point for query
            q_processed = helper.preprocess(query)
            q_vec = self.tfidf.transform([q_processed])
            q_2d = self.svd.transform(q_vec)[0]
            
            # Get 2D points for results
            results_2d = self.svd_2d[results_indices]
            
            # Get 2D points for a random sample of background questions (context)
            # Sample 100 random points to show the "universe"
            import random
            all_indices = list(range(len(self.questions_df)))
            bg_indices = random.sample(all_indices, min(100, len(all_indices)))
            bg_2d = self.svd_2d[bg_indices]
            
            return {
                'query_x': float(q_2d[0]),
                'query_y': float(q_2d[1]),
                'results': [
                    {'x': float(pt[0]), 'y': float(pt[1]), 'type': 'result'} 
                    for pt in results_2d
                ],
                'background': [
                    {'x': float(pt[0]), 'y': float(pt[1]), 'type': 'background'} 
                    for pt in bg_2d
                ]
            }
        except Exception as e:
            logger.error(f"Error generating visualization data: {e}")
            return None

    def _question_to_slug(self, question: str) -> str:
        # [Preserved function]
        pass

    def search(self, query: str, top_k: int = 10, min_similarity: float = 0.0) -> Dict:
        """
        Search for similar questions
        Returns Dict with 'results' list and 'visualization' data
        """
        if self.questions_df is None or len(self.questions_df) == 0:
            return {'results': []}
        
        if not query or len(query.strip()) == 0:
            return {'results': []}
        
        try:
            # Extract query features
            query_features = self._extract_query_features(query)
            if query_features is None:
                return {'results': []}
            
            # Compute similarities
            similarities = self._compute_similarity_scores(query_features)
            if len(similarities) == 0:
                return {'results': []}
            
            # Get top-k results
            # We get 3x k candidates for re-ranking to ensure high quality
            candidate_indices = np.argsort(similarities)[::-1][:min(top_k * 3, len(similarities))]
            
            candidates = []
            valid_indices = [] # Keep track of indices we actually use
            
            for idx in candidate_indices:
                similarity = float(similarities[idx])
                if similarity < min_similarity:
                    continue
                
                try:
                    row = self.questions_df.iloc[idx]
                    candidates.append({
                        'idx': idx,
                        'question': str(row['question']),
                        'qid': row.get('qid', None),
                        'similarity': similarity
                    })
                    valid_indices.append(idx)
                except Exception:
                    continue
            
            # RE-RANKING STEP: Use the actual ML model if available
            if self.model is not None and len(candidates) > 0:
                try:
                    # Prepare batch features
                    for candidate in candidates:
                        q_candidate = candidate['question']
                        try:
                            # Generate full feature vector
                            features = helper.query_point_creator(query, q_candidate)
                            
                            # Predict probability
                            probas = self.model.predict_proba(features)[0]
                            duplicate_prob = probas[1]
                            
                            candidates_score = duplicate_prob
                            candidate['model_score'] = candidates_score
                        except Exception as e:
                            candidate['model_score'] = 0.0
                    
                    # Sort by Model Score
                    candidates.sort(key=lambda x: (x.get('model_score', 0), x['similarity']), reverse=True)
                    
                except Exception as e:
                    logger.error(f"Re-ranking failed: {e}")
            
            # Format final results
            results = []
            final_indices = []
            
            for cand in candidates[:top_k]:
                q_text = cand['question'].strip()
                if q_text:
                    search_q = quote(q_text)
                    link = f"https://www.quora.com/search?q={search_q}"
                else:
                    link = None
                
                if 'model_score' in cand:
                    score = cand['model_score']
                    score_type = "Model Confidence"
                else:
                    score = cand['similarity']
                    score_type = "TF-IDF Similarity"

                results.append({
                    'question': cand['question'],
                    'qid': int(cand['qid']) if cand['qid'] and pd.notna(cand['qid']) else None,
                    'similarity': score,
                    'similarity_percent': round(score * 100, 2),
                    'score_type': score_type,
                    'link': link
                })
                final_indices.append(cand['idx'])
            
            # Generate Visualization Data
            viz_data = self.get_visualization_data(query, final_indices)
            
            return {
                'results': results,
                'visualization': viz_data
            }
            
        except Exception as e:
            logger.error(f"Error in search: {e}")
            return {'results': []}
    
    def get_stats(self) -> Dict:
        """Get statistics about the indexed questions"""
        if self.questions_df is None:
            return {'total_questions': 0, 'indexed': False}
        
        return {
            'total_questions': len(self.questions_df),
            'indexed': True,
            'has_features': self.question_features is not None
        }

