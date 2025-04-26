import streamlit as st
import pandas as pd
import datetime
import time
import os

from news_fetcher import fetch_news
from sentiment_analyzer import analyze_sentiment
from topic_modeler import perform_topic_modeling
from data_visualizer import (
    plot_sentiment_distribution,
    plot_sentiment_over_time,
    plot_topic_distribution,
    create_wordcloud_for_topic
)
from utils import filter_dataframe

# Set page configuration
st.set_page_config(
    page_title="Real-Time News Analysis Dashboard",
    page_icon="📰",
    layout="wide"
)

# Initialize session state for storing the news data
if 'news_data' not in st.session_state:
    st.session_state.news_data = None
    
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = None
    
if 'update_frequency' not in st.session_state:
    st.session_state.update_frequency = 60  # Default: update every 60 minutes

def fetch_and_process_news(query, from_date, to_date, sources, language, sort_by, update_freq):
    """Fetch news data and process it with sentiment analysis and topic modeling"""
    with st.spinner('Fetching and analyzing news data...'):
        # Get API key from environment variable
        api_key = os.getenv("NEWS_API_KEY")
        if not api_key:
            st.error("NEWS_API_KEY environment variable is not set. Please set it to use the NewsAPI.")
            return None
            
        # Fetch news articles
        news_data = fetch_news(
            api_key=api_key,
            query=query,
            from_date=from_date,
            to_date=to_date,
            sources=sources,
            language=language,
            sort_by=sort_by
        )
        
        if news_data is None or news_data.empty:
            st.error("No news data found. Try adjusting your search parameters.")
            return None
            
        # Perform sentiment analysis
        news_data = analyze_sentiment(news_data)
        
        # Perform topic modeling
        news_data, topic_terms = perform_topic_modeling(news_data, num_topics=5)
        
        # Update last update time
        st.session_state.last_update_time = datetime.datetime.now()
        st.session_state.update_frequency = update_freq
        
        return news_data, topic_terms

# Sidebar for search parameters and filters
st.sidebar.title("News Search Parameters")

query = st.sidebar.text_input("Search Query", "technology")

# Date range selection
today = datetime.date.today()
month_ago = today - datetime.timedelta(days=30)
from_date = st.sidebar.date_input("From Date", month_ago)
to_date = st.sidebar.date_input("To Date", today)

# Sources and language filter
sources = st.sidebar.text_input("Sources (domain or URL, e.g. bbc.com or https://www.bbc.com)", "")
language_options = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it"
}
language = st.sidebar.selectbox(
    "Language",
    options=list(language_options.keys()),
    index=0
)

# Sort order
sort_options = {
    "Published At": "publishedAt",
    "Relevancy": "relevancy",
    "Popularity": "popularity"
}
sort_by = st.sidebar.selectbox(
    "Sort By",
    options=list(sort_options.keys()),
    index=0
)

# Update frequency
update_freq = st.sidebar.slider(
    "Update Frequency (minutes)",
    min_value=15,
    max_value=120,
    value=st.session_state.update_frequency,
    step=15
)

# Search button
search_button = st.sidebar.button("Search News")

# Filter sidebar section
st.sidebar.title("Filter Results")

# These filters will be applied after the data is loaded
if st.session_state.news_data is not None:
    sentiment_filter = st.sidebar.multiselect(
        "Sentiment",
        options=["Positive", "Neutral", "Negative"],
        default=["Positive", "Neutral", "Negative"]
    )
    
    # Get unique sources from the data
    sources_list = st.session_state.news_data['source'].unique().tolist()
    source_filter = st.sidebar.multiselect(
        "Sources",
        options=sources_list,
        default=sources_list
    )
    
    # Get unique topics from the data
    topic_list = sorted(st.session_state.news_data['topic'].unique().tolist())
    topic_filter = st.sidebar.multiselect(
        "Topics",
        options=topic_list,
        default=topic_list
    )
    
    # Apply filters
    filtered_data = filter_dataframe(
        st.session_state.news_data,
        sentiment_filter,
        source_filter,
        topic_filter
    )
else:
    filtered_data = None

# Main dashboard area
st.title("📰 Real-Time News Analysis Dashboard")

# Display last update time and auto-update
if st.session_state.last_update_time:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"Last updated: {st.session_state.last_update_time.strftime('%Y-%m-%d %H:%M:%S')}")
    with col2:
        auto_update = st.checkbox("Auto-update", value=False)

# Check if we need to fetch new data
current_time = datetime.datetime.now()
if (search_button or 
    (st.session_state.news_data is None) or 
    (auto_update and st.session_state.last_update_time and 
     (current_time - st.session_state.last_update_time).seconds > st.session_state.update_frequency * 60)):
    
    result = fetch_and_process_news(
        query,
        from_date,
        to_date,
        sources,
        language_options[language],
        sort_options[sort_by],
        update_freq
    )
    
    if result:
        st.session_state.news_data, topic_terms = result
        filtered_data = st.session_state.news_data  # On initial load, no filters applied

# Display data and visualizations if data is available
if filtered_data is not None and not filtered_data.empty:
    # Show basic stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Articles", len(filtered_data))
    with col2:
        positive_count = len(filtered_data[filtered_data['sentiment'] == 'Positive'])
        st.metric("Positive Articles", positive_count, f"{positive_count/len(filtered_data):.1%}")
    with col3:
        negative_count = len(filtered_data[filtered_data['sentiment'] == 'Negative'])
        st.metric("Negative Articles", negative_count, f"{negative_count/len(filtered_data):.1%}")
    
    # Visualizations
    st.subheader("Sentiment Analysis")
    tab1, tab2 = st.tabs(["Distribution", "Trend"])
    
    with tab1:
        st.plotly_chart(plot_sentiment_distribution(filtered_data), use_container_width=True)
    
    with tab2:
        st.plotly_chart(plot_sentiment_over_time(filtered_data), use_container_width=True)
    
    # Topic modeling visualizations
    st.subheader("Topic Modeling")
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_topic_distribution(filtered_data), use_container_width=True)
    
    with col2:
        topics = filtered_data['topic'].unique()
        selected_topic = st.selectbox("Select Topic to View", options=topics)
        
        # Get the terms for the selected topic
        topic_idx = int(selected_topic.split()[-1])
        topic_words = topic_terms[topic_idx-1]  # Adjust for 0-based indexing
        
        # Display word cloud for selected topic
        st.pyplot(create_wordcloud_for_topic(topic_words))
    
    # Display news articles table
    st.subheader("News Articles")
    
    # Create a cleaner display table
    display_df = filtered_data[['title', 'source', 'publishedAt', 'sentiment', 'topic']].copy()
    display_df['publishedAt'] = pd.to_datetime(display_df['publishedAt']).dt.strftime('%Y-%m-%d %H:%M')
    
    # Pagination
    items_per_page = 10
    total_pages = (len(display_df) - 1) // items_per_page + 1
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0
    
    def change_page(page_num):
        st.session_state.current_page = page_num
    
    # Page navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        current_page = st.slider("Page", 1, total_pages, st.session_state.current_page + 1)
        change_page(current_page - 1)
    
    # Display the articles for the current page
    start_idx = st.session_state.current_page * items_per_page
    end_idx = min(start_idx + items_per_page, len(display_df))
    
    st.dataframe(display_df.iloc[start_idx:end_idx], use_container_width=True)
    
    # Article details expander
    with st.expander("Article Details"):
        article_idx = st.selectbox(
            "Select Article", 
            options=range(len(filtered_data)),
            format_func=lambda x: filtered_data.iloc[x]['title']
        )
        
        article = filtered_data.iloc[article_idx]
        st.subheader(article['title'])
        st.caption(f"Source: {article['source']} | Published: {pd.to_datetime(article['publishedAt']).strftime('%Y-%m-%d %H:%M')}")
        st.write(f"**Sentiment:** {article['sentiment']} | **Sentiment Score:** {article['sentiment_score']:.2f}")
        st.write(f"**Topic:** {article['topic']}")
        st.write("**Description:**")
        st.write(article['description'])
        
        if pd.notna(article['url']):
            st.markdown(f"[Read Full Article]({article['url']})")
else:
    if search_button:
        st.warning("No news data found. Try adjusting your search parameters.")
    else:
        st.info("Use the sidebar to set your search parameters and click 'Search News' to fetch data.")

# Footer
st.markdown("---")
st.caption("Data source: NewsAPI | Updated in near real-time")
