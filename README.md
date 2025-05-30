
# 📰 Real-Time News Analysis Dashboard

A powerful news analytics platform that fetches, analyzes, and visualizes news articles in real-time using sentiment analysis and topic modeling.

## 🌟 Features

- **Real-time News Fetching**: Retrieve latest news articles from various sources using NewsAPI
- **Sentiment Analysis**: Analyze the emotional tone of news articles (Positive, Neutral, Negative)
- **Topic Modeling**: Automatically categorize articles into relevant topics
- **Interactive Visualizations**: 
  - Sentiment distribution charts
  - Topic distribution pie charts
  - Sentiment trends over time
  - Topic-specific word clouds
- **Customizable Filters**: Filter articles by sentiment, source, and topic
- **Auto-update**: Configurable automatic refresh of news data

## 🚀 Getting Started

1. Set your NewsAPI key as an environment variable:
   ```bash
   export NEWS_API_KEY="your_api_key_here"
   ```

2. Install required packages:
   ```bash
   pip install matplotlib nltk pandas plotly scikit-learn streamlit trafilatura wordcloud
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Natural Language Processing**: NLTK, Scikit-learn
- **Visualization**: Plotly, Matplotlib, WordCloud
- **News Data**: NewsAPI

## 📊 Components

- `app.py`: Main application and UI logic
- `news_fetcher.py`: Handles news article retrieval
- `sentiment_analyzer.py`: Performs sentiment analysis
- `topic_modeler.py`: Implements topic modeling
- `data_visualizer.py`: Creates interactive visualizations
- `utils.py`: Utility functions for data processing

## 🎛️ Configuration

- Customize search parameters in the sidebar
- Adjust update frequency (15-120 minutes)
- Filter results by sentiment, source, and topic
- Configure auto-update functionality

## 📈 Visualizations

- **Sentiment Distribution**: Bar chart showing positive, neutral, and negative article counts
- **Sentiment Trends**: Line chart tracking sentiment changes over time
- **Topic Distribution**: Pie chart displaying article distribution across topics
- **Word Clouds**: Visual representation of key terms in each topic

 