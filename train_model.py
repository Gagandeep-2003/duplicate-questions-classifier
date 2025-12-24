import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from bs4 import BeautifulSoup
import re
from fuzzywuzzy import fuzz
import distance

# Load and sample data
df = pd.read_csv('train.csv')
new_df = df.sample(30000, random_state=2)

def preprocess(q):
    q = str(q).lower().strip()
    
    # Replace special characters
    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')
    q = q.replace('[math]', '')
    
    # Replace numbers
    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)
    
    # Remove HTML tags
    q = BeautifulSoup(q, 'html.parser').get_text()
    
    # Remove punctuations
    pattern = re.compile(r'\W')
    q = re.sub(pattern, ' ', q).strip()
    
    return q

# Preprocess questions
new_df['question1'] = new_df['question1'].apply(preprocess)
new_df['question2'] = new_df['question2'].apply(preprocess)

# Create features
def create_features(df):
    features = []
    
    for index, row in df.iterrows():
        q1 = row['question1']
        q2 = row['question2']
        
        # Basic features
        f = []
        f.append(len(q1))
        f.append(len(q2))
        
        f.append(len(q1.split()))
        f.append(len(q2.split()))
        
        # Common words
        w1 = set(map(lambda word: word.lower().strip(), q1.split()))
        w2 = set(map(lambda word: word.lower().strip(), q2.split()))
        common_words = len(w1 & w2)
        total_words = len(w1) + len(w2)
        
        f.append(common_words)
        f.append(total_words)
        f.append(round(common_words / total_words, 2) if total_words > 0 else 0)
        
        # Fuzzy features
        f.append(fuzz.QRatio(q1, q2))
        f.append(fuzz.partial_ratio(q1, q2))
        f.append(fuzz.token_sort_ratio(q1, q2))
        f.append(fuzz.token_set_ratio(q1, q2))
        
        # Length features
        f.append(abs(len(q1.split()) - len(q2.split())))
        f.append((len(q1.split()) + len(q2.split())) / 2)
        
        strs = list(distance.lcsubstrings(q1, q2))
        f.append(len(strs[0]) / (min(len(q1), len(q2)) + 1) if strs else 0)
        
        features.append(f)
    
    return np.array(features)

# Create features array
print('Creating features...')
features = create_features(new_df)

# Create BOW features
print('Creating BOW features...')
cv = CountVectorizer(max_features=2993)  # Adjusted to match expected features (6006 - 20) / 2
q1_arr = cv.fit_transform(new_df['question1']).toarray()
q2_arr = cv.transform(new_df['question2']).toarray()

# Combine all features
X = np.hstack((features, q1_arr, q2_arr))
y = new_df['is_duplicate'].values

# Train model
print('Training model...')
model = RandomForestClassifier(n_estimators=100, random_state=2)
model.fit(X, y)

# Create streamlit-app directory if it doesn't exist
import os
if not os.path.exists('streamlit-app'):
    os.makedirs('streamlit-app')

# Save model and vectorizer
print('Saving model and vectorizer...')
with open('streamlit-app/model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('streamlit-app/cv.pkl', 'wb') as f:
    pickle.dump(cv, f)

# Create and save stopwords
print('Downloading NLTK data...')
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

with open('streamlit-app/stopwords.pkl', 'wb') as f:
    pickle.dump(stop_words, f)

print('Training completed and files saved successfully!')