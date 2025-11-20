"""
News Insights Page
AI-powered news analysis and contextual insights
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime, timedelta

sys.path.append(str(Path(__file__).parent.parent))

from utils.visualizations import create_header
from models.rag_pipeline import F1RAGPipeline
from data.news_scraper import F1NewsCollector
from _imports import *

st.set_page_config(
    page_title="News & Insights",
    page_icon="📰",
    layout="wide"
)

st.markdown(create_header(
    "News & Insights",
    "AI-powered analysis of F1 news and developments"
), unsafe_allow_html=True)


def load_news():
    try:
        possible_paths = [
            'data/raw/f1_news.csv',
            f'{BASE_PATH}/data/raw/f1_news.csv'
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return pd.read_csv(path)
        
        return None
    except Exception as e:
        st.error(f"Error loading news: {e}")
        return None


def main():
    # Sidebar
    st.sidebar.header("News Filters")
    
    # Date range
    days_back = st.sidebar.slider("Days Back", 7, 90, 30)
    
    # Category filter
    categories = [
        "All",
        "race_results",
        "qualifying",
        "technical",
        "driver_move",
        "injury",
        "championship",
        "regulation"
    ]
    selected_category = st.sidebar.selectbox("Category", categories)
    
    # Source filter
    news_df = load_news()
    
    if news_df is not None:
        sources = ["All"] + sorted(news_df['source'].unique().tolist())
        selected_source = st.sidebar.selectbox("Source", sources)
    else:
        selected_source = "All"
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh News"):
        with st.spinner("Collecting latest news..."):
            try:
                import os
                api_key = os.getenv('NEWS_API_KEY')
                
                if api_key:
                    collector = F1NewsCollector(news_api_key=api_key)
                    news_df = collector.collect_all_news(days_back=days_back)
                    
                    if not news_df.empty:
                        collector.save_news(news_df, Path('data/raw/f1_news.csv'))
                        st.success(f"Collected {len(news_df)} articles!")
                        st.rerun()
                else:
                    st.warning("NEWS_API_KEY not found. Using cached data.")
            except Exception as e:
                st.error(f"Error collecting news: {e}")
    
    # Main content
    if news_df is None or news_df.empty:
        st.warning("No news data available. Click 'Refresh News' to collect latest articles.")
        
        st.info("""
        ### Setup Required
        
        To enable news collection:
        1. Get a free API key from https://newsapi.org
        2. Add it to your `.env` file: `NEWS_API_KEY=your_key_here`
        3. Click the 'Refresh News' button above
        """)
        return
    
    # Apply filters
    news_df['published_at'] = pd.to_datetime(news_df['published_at'])
    cutoff_date = datetime.now() - timedelta(days=days_back)
    filtered_news = news_df[news_df['published_at'] >= cutoff_date]
    
    if selected_category != "All" and 'auto_category' in filtered_news.columns:
        filtered_news = filtered_news[filtered_news['auto_category'] == selected_category]
    
    if selected_source != "All":
        filtered_news = filtered_news[filtered_news['source'] == selected_source]
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Articles", len(filtered_news))
    
    with col2:
        sources_count = filtered_news['source'].nunique()
        st.metric("Sources", sources_count)
    
    with col3:
        if 'auto_category' in filtered_news.columns:
            categories_count = filtered_news['auto_category'].nunique()
            st.metric("Categories", categories_count)
        else:
            st.metric("Categories", "N/A")
    
    with col4:
        latest = filtered_news['published_at'].max()
        if pd.notna(latest):
            hours_ago = (datetime.now() - latest).total_seconds() / 3600
            st.metric("Latest Article", f"{hours_ago:.0f}h ago")
        else:
            st.metric("Latest Article", "N/A")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "📰 Recent News",
        "🔍 Search & Analysis",
        "📊 Trends"
    ])
    
    with tab1:
        st.subheader("Recent F1 News")
        
        # Display news articles
        for idx, article in filtered_news.head(20).iterrows():
            with st.expander(
                f"📰 {article['title']} - {article['source']}", 
                expanded=idx < 3
            ):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    if pd.notna(article.get('description')):
                        st.write(article['description'])
                    
                    if pd.notna(article.get('content')):
                        st.markdown("---")
                        st.write(article['content'][:500] + "..." if len(str(article['content'])) > 500 else article['content'])
                
                with col2:
                    st.caption(f"**Published:**")
                    st.caption(article['published_at'].strftime('%Y-%m-%d %H:%M'))
                    
                    if 'auto_category' in article:
                        st.caption(f"**Category:**")
                        st.caption(article['auto_category'].replace('_', ' ').title())
                    
                    if pd.notna(article.get('url')):
                        st.link_button("Read Full Article", article['url'])
    
    with tab2:
        st.subheader("🔍 Semantic Search & Analysis")
        
        # Search interface
        search_query = st.text_input(
            "Search news articles",
            placeholder="e.g., 'Max Verstappen championship', 'Ferrari upgrades', etc."
        )
        
        if search_query:
            with st.spinner("Searching and analyzing..."):
                try:
                    # Initialize RAG pipeline
                    rag = F1RAGPipeline(corpus_path=Path('data/news_corpus'))
                    
                    # If corpus doesn't exist, create it
                    if not Path('data/news_corpus/documents.json').exists():
                        st.info("Building news corpus for the first time...")
                        rag.add_news_corpus(filtered_news)
                        rag.save_corpus(Path('data/news_corpus'))
                    
                    # Retrieve relevant articles
                    from models.rag_pipeline import DocumentRetriever
                    retriever = DocumentRetriever()
                    retriever.load_index(Path('data/news_corpus'))
                    
                    results = retriever.retrieve(search_query, top_k=5)
                    
                    if results:
                        st.success(f"Found {len(results)} relevant articles")
                        
                        for doc, score in results:
                            with st.container():
                                st.markdown(f"""
                                <div style='background: #f8f9fa; padding: 1rem; border-radius: 10px; margin: 1rem 0; border-left: 4px solid #E10600;'>
                                    <h4>{doc.title}</h4>
                                    <p>{doc.content[:300]}...</p>
                                    <small>Relevance: {score:.2%} | {doc.source} | {doc.published_at.strftime('%Y-%m-%d')}</small>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.warning("No relevant articles found for your query.")
                
                except Exception as e:
                    st.error(f"Search error: {e}")
                    st.info("The RAG pipeline needs to be initialized with news data.")
        
        else:
            st.info("Enter a search query to find relevant F1 news articles using AI-powered semantic search.")
    
    with tab3:
        st.subheader("📊 News Trends & Analytics")
        
        # Articles over time
        st.markdown("### Articles Published Over Time")
        
        filtered_news['date'] = filtered_news['published_at'].dt.date
        articles_by_date = filtered_news.groupby('date').size().reset_index(name='count')
        
        import plotly.express as px
        
        fig = px.line(
            articles_by_date,
            x='date',
            y='count',
            title="News Volume Over Time"
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of Articles",
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Category distribution
        if 'auto_category' in filtered_news.columns:
            st.markdown("### News by Category")
            
            category_counts = filtered_news['auto_category'].value_counts().reset_index()
            category_counts.columns = ['category', 'count']
            
            fig = px.pie(
                category_counts,
                values='count',
                names='category',
                title="Article Distribution by Category"
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Source distribution
        st.markdown("### Top News Sources")
        
        source_counts = filtered_news['source'].value_counts().head(10).reset_index()
        source_counts.columns = ['source', 'count']
        
        fig = px.bar(
            source_counts,
            x='source',
            y='count',
            title="Top 10 News Sources"
        )
        
        fig.update_layout(
            xaxis_title="Source",
            yaxis_title="Number of Articles",
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Most mentioned terms (simple word frequency)
        st.markdown("### Hot Topics")
        
        if pd.notna(filtered_news['title']).any():
            all_titles = ' '.join(filtered_news['title'].dropna().astype(str))
            words = all_titles.lower().split()
            
            # Filter common words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                         'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'be', 'has', 'have'}
            
            word_freq = pd.Series([w for w in words if len(w) > 3 and w not in stop_words]).value_counts().head(15)
            
            col1, col2, col3 = st.columns(3)
            
            for idx, (word, count) in enumerate(word_freq.items()):
                if idx % 3 == 0:
                    col1.metric(word.title(), count)
                elif idx % 3 == 1:
                    col2.metric(word.title(), count)
                else:
                    col3.metric(word.title(), count)


if __name__ == "__main__":
    main()
