import requests
import pandas as pd
from datetime import datetime
import time

def fetch_news(api_key, query, from_date=None, to_date=None, sources="", language="en", sort_by="publishedAt", page_size=100):
    """
    Fetch news articles from NewsAPI based on the provided parameters
    
    Parameters:
    -----------
    api_key : str
        NewsAPI API key
    query : str
        The search query
    from_date : datetime.date
        Start date for articles
    to_date : datetime.date
        End date for articles
    sources : str
        Comma-separated list of news sources
    language : str
        Two-letter ISO-639-1 code of the language
    sort_by : str
        The order to sort the articles (publishedAt, relevancy, popularity)
    page_size : int
        Number of results to return per page (max 100)
        
    Returns:
    --------
    pandas.DataFrame or None
        DataFrame containing news articles or None if the request failed
    """
    try:
        # Format dates for the API
        from_date_str = from_date.strftime('%Y-%m-%d') if from_date else None
        to_date_str = to_date.strftime('%Y-%m-%d') if to_date else None
        
        # Build the request parameters
        params = {
            'apiKey': api_key,
            'q': query,
            'pageSize': page_size,
            'language': language,
            'sortBy': sort_by
        }
        
        # Add optional parameters if provided
        if from_date_str:
            params['from'] = from_date_str
        if to_date_str:
            params['to'] = to_date_str
        if sources:
            params['sources'] = sources
            
        # Make the API request
        response = requests.get('https://newsapi.org/v2/everything', params=params)
        
        # Check if the request was successful
        if response.status_code != 200:
            return None
            
        # Parse the response
        data = response.json()
        
        if data['status'] != 'ok' or data['totalResults'] == 0:
            return None
            
        articles = data['articles']
        
        # Convert to DataFrame
        df = pd.DataFrame(articles)
        
        # Extract source name from the source dictionary
        df['source'] = df['source'].apply(lambda x: x['name'] if isinstance(x, dict) and 'name' in x else 'Unknown')
        
        # Convert publishedAt to datetime
        df['publishedAt'] = pd.to_datetime(df['publishedAt'])
        
        return df
        
    except Exception as e:
        print(f"Error fetching news: {e}")
        return None

def fetch_news_with_retry(api_key, query, max_retries=3, **kwargs):
    """
    Fetch news with retry logic in case of API errors
    
    Parameters:
    -----------
    api_key : str
        NewsAPI API key
    query : str
        The search query
    max_retries : int
        Maximum number of retry attempts
    **kwargs : dict
        Additional parameters to pass to fetch_news
        
    Returns:
    --------
    pandas.DataFrame or None
        DataFrame containing news articles or None if all retries failed
    """
    for attempt in range(max_retries):
        df = fetch_news(api_key, query, **kwargs)
        if df is not None:
            return df
        
        # Wait before retrying (with exponential backoff)
        wait_time = 2 ** attempt
        time.sleep(wait_time)
    
    return None
