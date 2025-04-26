import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re

# Download required NLTK resources
nltk.download('vader_lexicon', quiet=True)

# Set NLTK data path
nltk.data.path.append('/home/runner/nltk_data')

def clean_text(text):
    """
    Clean text data by removing URLs, special characters, etc.
    
    Parameters:
    -----------
    text : str
        The text to clean
        
    Returns:
    --------
    str
        Cleaned text
    """
    if not isinstance(text, str) or pd.isna(text):
        return ""
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def categorize_sentiment(score):
    """
    Categorize sentiment score into positive, neutral, or negative
    
    Parameters:
    -----------
    score : float
        Compound sentiment score from VADER
        
    Returns:
    --------
    str
        Sentiment category (Positive, Neutral, or Negative)
    """
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def analyze_sentiment(df):
    """
    Analyze sentiment of news articles in the DataFrame
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing news articles with 'title', 'description', and 'content' columns
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with sentiment analysis results added
    """
    # Initialize the sentiment analyzer
    sid = SentimentIntensityAnalyzer()
    
    # Create a copy of the DataFrame to avoid modifying the original
    df_analyzed = df.copy()
    
    # Clean and combine text for analysis
    df_analyzed['title_clean'] = df_analyzed['title'].apply(clean_text)
    
    if 'description' in df_analyzed.columns:
        df_analyzed['description_clean'] = df_analyzed['description'].apply(clean_text)
    else:
        df_analyzed['description_clean'] = ""
    
    # Create a combined text field for sentiment analysis
    # Use title and description as they are more likely to be available and contain the core sentiment
    df_analyzed['analysis_text'] = df_analyzed['title_clean'] + ". " + df_analyzed['description_clean']
    
    # Calculate sentiment scores
    df_analyzed['sentiment_scores'] = df_analyzed['analysis_text'].apply(
        lambda text: sid.polarity_scores(text) if isinstance(text, str) else {'compound': 0}
    )
    
    # Extract compound score
    df_analyzed['sentiment_score'] = df_analyzed['sentiment_scores'].apply(
        lambda score_dict: score_dict['compound'] if isinstance(score_dict, dict) else 0
    )
    
    # Categorize sentiment
    df_analyzed['sentiment'] = df_analyzed['sentiment_score'].apply(categorize_sentiment)
    
    # Drop intermediate columns
    df_analyzed = df_analyzed.drop(
        columns=['title_clean', 'description_clean', 'analysis_text', 'sentiment_scores'],
        errors='ignore'
    )
    
    return df_analyzed
