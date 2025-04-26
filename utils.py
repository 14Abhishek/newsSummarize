import pandas as pd

def filter_dataframe(df, sentiment_filter, source_filter, topic_filter):
    """
    Filter the dataframe based on selected criteria
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing news articles
    sentiment_filter : list
        List of sentiment categories to include
    source_filter : list
        List of news sources to include
    topic_filter : list
        List of topics to include
        
    Returns:
    --------
    pandas.DataFrame
        Filtered DataFrame
    """
    # Create a copy to avoid modifying the original
    filtered_df = df.copy()
    
    # Apply sentiment filter
    if sentiment_filter and len(sentiment_filter) > 0:
        filtered_df = filtered_df[filtered_df['sentiment'].isin(sentiment_filter)]
        
    # Apply source filter
    if source_filter and len(source_filter) > 0:
        filtered_df = filtered_df[filtered_df['source'].isin(source_filter)]
        
    # Apply topic filter
    if topic_filter and len(topic_filter) > 0:
        filtered_df = filtered_df[filtered_df['topic'].isin(topic_filter)]
        
    return filtered_df

def format_date(date_string):
    """
    Format a date string for display
    
    Parameters:
    -----------
    date_string : str
        Date string to format
        
    Returns:
    --------
    str
        Formatted date string
    """
    try:
        date_obj = pd.to_datetime(date_string)
        return date_obj.strftime('%Y-%m-%d %H:%M')
    except:
        return date_string
    
def truncate_text(text, max_length=100):
    """
    Truncate text to a specified maximum length
    
    Parameters:
    -----------
    text : str
        Text to truncate
    max_length : int
        Maximum length of the truncated text
        
    Returns:
    --------
    str
        Truncated text
    """
    if not isinstance(text, str):
        return ""
        
    if len(text) <= max_length:
        return text
        
    return text[:max_length] + "..."
