import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

# Set NLTK data path
nltk.data.path.append('/home/runner/nltk_data')

def preprocess_text(text):
    """
    Preprocess text for topic modeling
    
    Parameters:
    -----------
    text : str
        The text to preprocess
        
    Returns:
    --------
    str
        Preprocessed text
    """
    if not isinstance(text, str) or pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Simple tokenization to avoid NLTK punkt issues
    tokens = text.split()
    
    # Remove stopwords
    try:
        stop_words = set(stopwords.words('english'))
    except:
        # Fallback stopwords if NLTK data is not available
        stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
                           'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 
                           'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 
                           'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
                           'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 
                           'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 
                           'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
                           'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 
                           'with', 'about', 'against', 'between', 'into', 'through', 'during', 
                           'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
                           'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 
                           'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 
                           'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 
                           'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 
                           'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 
                           'should', 'now'])
    
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    try:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    except:
        # Skip lemmatization if NLTK is not available
        pass
    
    # Join tokens back into a string
    text = ' '.join(tokens)
    
    return text

def perform_topic_modeling(df, num_topics=5, num_words=10, min_df=2, max_df=0.95):
    """
    Perform topic modeling on news articles using Latent Dirichlet Allocation
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing news articles with 'title' and 'description' columns
    num_topics : int
        Number of topics to extract
    num_words : int
        Number of words to include for each topic
    min_df : int or float
        Minimum document frequency for CountVectorizer
    max_df : float
        Maximum document frequency for CountVectorizer
        
    Returns:
    --------
    tuple
        (pandas.DataFrame with topic assignments, list of topic terms)
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_topics = df.copy()
    
    # Prepare the text for topic modeling
    if 'description' in df_topics.columns and 'title' in df_topics.columns:
        # Combine title and description for better topic detection
        df_topics['text_for_topics'] = df_topics['title'].fillna('') + ' ' + df_topics['description'].fillna('')
    elif 'title' in df_topics.columns:
        df_topics['text_for_topics'] = df_topics['title'].fillna('')
    else:
        df_topics['text_for_topics'] = ''
    
    # Preprocess the text
    df_topics['text_for_topics'] = df_topics['text_for_topics'].apply(preprocess_text)
    
    # Filter out empty text
    valid_indices = df_topics['text_for_topics'].str.strip().astype(bool)
    
    if valid_indices.sum() < num_topics:
        # Not enough documents for the requested number of topics
        # Adjust the number of topics
        num_topics = max(2, valid_indices.sum() // 2)
    
    # Extract text for valid documents
    valid_texts = df_topics.loc[valid_indices, 'text_for_topics'].values
    
    if len(valid_texts) == 0:
        # No valid texts for topic modeling, assign default topic
        df_topics['topic'] = 'Topic 1'
        return df_topics, [["no", "valid", "topics", "found"]]
    
    # Create document-term matrix
    vectorizer = CountVectorizer(
        max_df=max_df,
        min_df=min_df,
        stop_words='english'
    )
    
    try:
        dtm = vectorizer.fit_transform(valid_texts)
        
        # Get feature names (words)
        feature_names = vectorizer.get_feature_names_out()
        
        # If the matrix is too small, adjust the number of topics
        if dtm.shape[1] < num_topics or dtm.shape[0] < num_topics:
            num_topics = max(2, min(dtm.shape[0], dtm.shape[1]) // 2)
        
        # Train LDA model
        lda = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            max_iter=15
        )
        
        # Fit the model
        lda.fit(dtm)
        
        # Get topic terms
        topic_terms = []
        for topic_idx, topic in enumerate(lda.components_):
            top_features_idx = topic.argsort()[:-num_words-1:-1]
            top_features = [feature_names[i] for i in top_features_idx]
            topic_terms.append(top_features)
        
        # Transform documents to get topic distributions
        doc_topic_dist = lda.transform(dtm)
        
        # Assign the most prominent topic to each document
        topics = np.argmax(doc_topic_dist, axis=1)
        
        # Initialize all documents with a default topic
        df_topics['topic'] = 'Topic 1'
        
        # Assign topics to valid documents
        valid_indices_list = valid_indices[valid_indices].index.tolist()
        for i, topic_idx in enumerate(topics):
            if i < len(valid_indices_list):
                doc_idx = valid_indices_list[i]
                df_topics.at[doc_idx, 'topic'] = f'Topic {topic_idx+1}'
        
    except Exception as e:
        print(f"Error in topic modeling: {e}")
        # Assign default topic
        df_topics['topic'] = 'Topic 1'
        topic_terms = [["error", "in", "topic", "modeling"]]
    
    # Drop intermediate column
    df_topics = df_topics.drop(columns=['text_for_topics'])
    
    return df_topics, topic_terms
