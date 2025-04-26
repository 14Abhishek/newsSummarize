import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg for non-interactive use

def plot_sentiment_distribution(df):
    """
    Create a bar chart showing the distribution of sentiments
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing news articles with 'sentiment' column
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object for the sentiment distribution
    """
    # Count sentiments
    sentiment_counts = df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    
    # Define the order and colors
    order = ['Positive', 'Neutral', 'Negative']
    colors = {'Positive': '#4CAF50', 'Neutral': '#2196F3', 'Negative': '#F44336'}
    
    # Create the figure
    fig = px.bar(
        sentiment_counts,
        x='Sentiment',
        y='Count',
        color='Sentiment',
        category_orders={'Sentiment': order},
        color_discrete_map=colors,
        text='Count'
    )
    
    # Update layout
    fig.update_layout(
        title='Sentiment Distribution of News Articles',
        xaxis_title='Sentiment',
        yaxis_title='Number of Articles',
        showlegend=False
    )
    
    # Update traces
    fig.update_traces(textposition='outside')
    
    return fig

def plot_sentiment_over_time(df):
    """
    Create a line chart showing sentiment trends over time
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing news articles with 'publishedAt' and 'sentiment' columns
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object for the sentiment trends
    """
    # Ensure publishedAt is a datetime
    df = df.copy()
    df['publishedAt'] = pd.to_datetime(df['publishedAt'])
    
    # Create a date column for grouping
    df['date'] = df['publishedAt'].dt.date
    
    # Group by date and sentiment, and count
    sentiment_trends = df.groupby(['date', 'sentiment']).size().reset_index(name='count')
    
    # Pivot the data to have sentiments as columns
    pivot_table = sentiment_trends.pivot_table(
        index='date',
        columns='sentiment',
        values='count',
        fill_value=0
    ).reset_index()
    
    # Make sure all sentiment columns exist
    for sentiment in ['Positive', 'Neutral', 'Negative']:
        if sentiment not in pivot_table.columns:
            pivot_table[sentiment] = 0
    
    # Sort by date
    pivot_table = pivot_table.sort_values('date')
    
    # Create the figure
    fig = go.Figure()
    
    # Add traces for each sentiment
    fig.add_trace(go.Scatter(
        x=pivot_table['date'],
        y=pivot_table['Positive'],
        mode='lines+markers',
        name='Positive',
        line=dict(color='#4CAF50', width=2),
        marker=dict(color='#4CAF50', size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=pivot_table['date'],
        y=pivot_table['Neutral'],
        mode='lines+markers',
        name='Neutral',
        line=dict(color='#2196F3', width=2),
        marker=dict(color='#2196F3', size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=pivot_table['date'],
        y=pivot_table['Negative'],
        mode='lines+markers',
        name='Negative',
        line=dict(color='#F44336', width=2),
        marker=dict(color='#F44336', size=6)
    ))
    
    # Update layout
    fig.update_layout(
        title='Sentiment Trends Over Time',
        xaxis_title='Date',
        yaxis_title='Number of Articles',
        legend_title='Sentiment',
        hovermode='x unified'
    )
    
    return fig

def plot_topic_distribution(df):
    """
    Create a pie chart showing the distribution of topics
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing news articles with 'topic' column
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object for the topic distribution
    """
    # Count topics
    topic_counts = df['topic'].value_counts().reset_index()
    topic_counts.columns = ['Topic', 'Count']
    
    # Create a color palette
    colors = px.colors.qualitative.Set3
    
    # Create the figure
    fig = px.pie(
        topic_counts,
        names='Topic',
        values='Count',
        color_discrete_sequence=colors,
        title='Distribution of Topics'
    )
    
    # Update layout
    fig.update_layout(
        showlegend=True,
        legend_title='Topics'
    )
    
    # Update traces
    fig.update_traces(textinfo='percent+label')
    
    return fig

def create_wordcloud_for_topic(topic_words):
    """
    Create a word cloud visualization for a topic
    
    Parameters:
    -----------
    topic_words : list
        List of words associated with the topic
        
    Returns:
    --------
    matplotlib.figure.Figure
        Matplotlib figure object for the word cloud
    """
    # Create a dictionary with word frequencies
    word_freq = {word: i for i, word in enumerate(reversed(topic_words), 1)}
    
    # Create the word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',
        max_words=50,
        max_font_size=100
    ).generate_from_frequencies(word_freq)
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the word cloud
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Topic Key Terms', fontsize=15)
    
    # Tight layout
    plt.tight_layout()
    
    return fig
