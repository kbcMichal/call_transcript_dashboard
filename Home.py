import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import json
import re
from wordcloud import WordCloud
import seaborn as sns
from io import StringIO
import os
import warnings
import logging

# Filter out the specific deprecation warning
warnings.filterwarnings("ignore", message=".*_get_websocket_headers.*")
# Also suppress StreamlitAPIWarning logs
logging.getLogger("streamlit.runtime.scriptrunner.script_runner").setLevel(logging.ERROR)

# Only use valid deprecation settings
if hasattr(st, 'set_option'):
    try:
        st.set_option('deprecation.showWarningOnDirectUse', False)
    except:
        pass

# Set page config
st.set_page_config(
    page_title="Call Transcript Analysis Dashboard",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    /* Fix for metric boxes */
    div.metric-box {
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        margin-bottom: 20px;
        display: block;
        position: relative;
    }
    div.metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 5px;
    }
    div.metric-label {
        font-size: 1rem;
        color: #616161;
        text-align: center;
    }
    div.info-box {
        border-top: 5px solid #2196F3;
        background-color: #E3F2FD;
    }
    div.success-box {
        border-top: 5px solid #4CAF50;
        background-color: #E8F5E9;
    }
    div.warning-box {
        border-top: 5px solid #FFC107;
        background-color: #FFF8E1;
    }
    div.danger-box {
        border-top: 5px solid #F44336;
        background-color: #FFEBEE;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown("<h1 class='main-header'>📞 Call Transcript Analysis Dashboard</h1>", unsafe_allow_html=True)
st.markdown("""
This dashboard provides insights from analyzed call transcripts to help CRM teams improve customer interactions 
and drive better business outcomes. The analysis includes sentiment tracking, call categorization, compliance monitoring,
and actionable recommendations.
""")

# Load data function
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('/data/in/tables/CALL_TRANSCRIPT_LLM_RESPONSE_EXPANDED.csv')
        
        # Clean and prepare the data
        # Convert date columns
        if 'DATE_TIME' in df.columns:
            df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
            df['call_date'] = df['DATE_TIME'].dt.date
            df['call_hour'] = df['DATE_TIME'].dt.hour
            
        # Process transcript JSON
        def extract_conversation_summary(transcript_json):
            if isinstance(transcript_json, str):
                try:
                    # Parse the JSON transcript
                    transcript = json.loads(transcript_json)
                    
                    # Extract client and agent messages
                    client_messages = [msg['text'] for msg in transcript if msg['role'] == 'client']
                    agent_messages = [msg['text'] for msg in transcript if msg['role'] == 'agent']
                    
                    return {
                        'client_messages': client_messages,
                        'agent_messages': agent_messages,
                        'client_word_count': sum(len(msg.split()) for msg in client_messages),
                        'agent_word_count': sum(len(msg.split()) for msg in agent_messages),
                        'client_message_count': len(client_messages),
                        'agent_message_count': len(agent_messages)
                    }
                except:
                    return {
                        'client_messages': [],
                        'agent_messages': [],
                        'client_word_count': 0,
                        'agent_word_count': 0,
                        'client_message_count': 0,
                        'agent_message_count': 0
                    }
            return {
                'client_messages': [],
                'agent_messages': [],
                'client_word_count': 0,
                'agent_word_count': 0,
                'client_message_count': 0,
                'agent_message_count': 0
            }
        
        # Apply the extraction to the TRANSCRIPT column
        if 'TRANSCRIPT' in df.columns:
            transcript_data = df['TRANSCRIPT'].apply(extract_conversation_summary)
            
            # Add extracted data as new columns
            df['client_word_count'] = transcript_data.apply(lambda x: x['client_word_count'])
            df['agent_word_count'] = transcript_data.apply(lambda x: x['agent_word_count'])
            df['client_message_count'] = transcript_data.apply(lambda x: x['client_message_count'])
            df['agent_message_count'] = transcript_data.apply(lambda x: x['agent_message_count'])
        
        # Map sentiment to numeric values for calculations
        sentiment_map = {'positive': 1, 'neutral': 0.5, 'negative': 0}
        df['sentiment_score'] = df['client_sentiment'].map(sentiment_map)
        
        # Process key topics
        if 'key_topics' in df.columns:
            df['key_topics_list'] = df['key_topics'].str.split(',').apply(lambda x: [item.strip() for item in x] if isinstance(x, list) else [])
        
        # Process keywords
        if 'keywords' in df.columns:
            df['keywords_list'] = df['keywords'].str.split(',').apply(lambda x: [item.strip() for item in x] if isinstance(x, list) else [])
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Return a minimal DataFrame to prevent errors downstream
        return pd.DataFrame({
            'error': ['Data loading failed. Please check the file path and format.']
        })

# Load the data
df = load_data()

# Check if the data was loaded successfully
if 'error' in df.columns:
    st.error(df['error'][0])
    st.stop()

# Sidebar with logo
try:
    from PIL import Image
    
    logo_path = "/data/in/files/kbl.png"
    if os.path.exists(logo_path):
        logo = Image.open(logo_path)
        st.sidebar.image(logo, use_container_width=True)
    else:
        st.sidebar.info("Keboola logo not found at the expected path.")
except Exception as e:
    st.sidebar.info(f"Could not load logo: {e}")


st.sidebar.header("Filters")

# Date range filter
if 'call_date' in df.columns:
    min_date = df['call_date'].min()
    max_date = df['call_date'].max()
    
    date_range = st.sidebar.date_input(
        "Date Range",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df[(df['call_date'] >= start_date) & (df['call_date'] <= end_date)]
    else:
        df_filtered = df
else:
    df_filtered = df

# Call type filter
if 'CALL_TYPE' in df.columns:
    all_call_types = ['All'] + sorted(df['CALL_TYPE'].unique().tolist())
    selected_call_type = st.sidebar.selectbox("Call Type", all_call_types)
    
    if selected_call_type != 'All':
        df_filtered = df_filtered[df_filtered['CALL_TYPE'] == selected_call_type]

# Language filter
if 'LANGUAGE' in df.columns:
    all_languages = ['All'] + sorted(df['LANGUAGE'].unique().tolist())
    selected_language = st.sidebar.selectbox("Language", all_languages)
    
    if selected_language != 'All':
        df_filtered = df_filtered[df_filtered['LANGUAGE'] == selected_language]

# Sentiment filter
if 'client_sentiment' in df.columns:
    all_sentiments = ['All'] + sorted(df['client_sentiment'].unique().tolist())
    selected_sentiment = st.sidebar.selectbox("Client Sentiment", all_sentiments)
    
    if selected_sentiment != 'All':
        df_filtered = df_filtered[df_filtered['client_sentiment'] == selected_sentiment]

# Display the number of filtered calls
st.sidebar.markdown(f"**Showing {len(df_filtered)} calls**")

# Reset filters button
if st.sidebar.button("Reset Filters"):
    df_filtered = df

# Main dashboard content
# Create two columns for KPIs
col1, col2, col3, col4 = st.columns(4)

# KPIs in first row
with col1:
    st.markdown("""
    <div class="metric-box info-box">
        <div class="metric-value">{}</div>
        <div class="metric-label">Total Calls</div>
    </div>
    """.format(len(df_filtered)), unsafe_allow_html=True)

with col2:
    sentiment_avg = df_filtered['sentiment_score'].mean() * 100
    # Determine color based on sentiment
    if sentiment_avg >= 70:
        box_class = "success-box"
    elif sentiment_avg >= 40:
        box_class = "warning-box"
    else:
        box_class = "danger-box"
    
    st.markdown("""
    <div class="metric-box {}">
        <div class="metric-value">{:.1f}%</div>
        <div class="metric-label">Avg Sentiment Score</div>
    </div>
    """.format(box_class, sentiment_avg), unsafe_allow_html=True)

with col3:
    if 'follow_up_required' in df_filtered.columns:
        follow_up_pct = df_filtered['follow_up_required'].mean() * 100
        st.markdown("""
        <div class="metric-box warning-box">
            <div class="metric-value">{:.1f}%</div>
            <div class="metric-label">Follow-up Required</div>
        </div>
        """.format(follow_up_pct), unsafe_allow_html=True)

with col4:
    if 'recording_notice_given' in df_filtered.columns:
        recording_notice_pct = df_filtered['recording_notice_given'].mean() * 100
        box_class = "success-box" if recording_notice_pct >= 90 else "danger-box"
        
        st.markdown("""
        <div class="metric-box {}">
            <div class="metric-value">{:.1f}%</div>
            <div class="metric-label">Recording Notice Given</div>
        </div>
        """.format(box_class, recording_notice_pct), unsafe_allow_html=True)

# Create tabs for different sections of analysis
tab1, tab4, tab2, tab3 = st.tabs([
    "📊 Call Overview", 
    "🔍 Detailed Insights",
    "😊 Sentiment Analysis", 
    "💬 Topic Analysis"
])

# Tab 1: Call Overview
with tab1:
    st.markdown("<h2 class='sub-header'>Call Distribution</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'CALL_TYPE' in df_filtered.columns:
            call_type_counts = df_filtered['CALL_TYPE'].value_counts().reset_index()
            call_type_counts.columns = ['CALL_TYPE', 'Count']
            
            fig = px.bar(
                call_type_counts, 
                x='Count', 
                y='CALL_TYPE',
                orientation='h',
                title='Call Distribution by Type',
                color='Count',
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'client_interest_level' in df_filtered.columns:
            interest_counts = df_filtered['client_interest_level'].value_counts().reset_index()
            interest_counts.columns = ['Interest Level', 'Count']
            
            # Define color map based on interest level
            color_map = {'high': '#4CAF50', 'medium': '#FFC107', 'low': '#F44336'}
            
            fig = px.pie(
                interest_counts,
                values='Count',
                names='Interest Level',
                title='Client Interest Level Distribution',
                color='Interest Level',
                color_discrete_map=color_map
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<h2 class='sub-header'>Call Duration Analysis</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'call_duration_estimate_seconds' in df_filtered.columns:
            duration_by_type = df_filtered.groupby('CALL_TYPE')['call_duration_estimate_seconds'].mean().reset_index()
            duration_by_type.columns = ['Call Type', 'Average Duration (seconds)']
            
            fig = px.bar(
                duration_by_type,
                x='Call Type',
                y='Average Duration (seconds)',
                title='Average Call Duration by Type',
                color='Average Duration (seconds)',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'LANGUAGE' in df_filtered.columns and 'call_duration_estimate_seconds' in df_filtered.columns:
            duration_by_language = df_filtered.groupby('LANGUAGE')['call_duration_estimate_seconds'].mean().reset_index()
            duration_by_language.columns = ['Language', 'Average Duration (seconds)']
            
            fig = px.bar(
                duration_by_language,
                x='Language',
                y='Average Duration (seconds)',
                title='Average Call Duration by Language',
                color='Language',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

# Tab 2: Sentiment Analysis
with tab2:
    st.markdown("<h2 class='sub-header'>Client Sentiment Analysis</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'client_sentiment' in df_filtered.columns:
            sentiment_counts = df_filtered['client_sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']
            
            # Define color map for sentiments
            color_map = {'positive': '#4CAF50', 'neutral': '#FFC107', 'negative': '#F44336'}
            
            fig = px.bar(
                sentiment_counts,
                x='Sentiment',
                y='Count',
                title='Client Sentiment Distribution',
                color='Sentiment',
                color_discrete_map=color_map
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'client_sentiment' in df_filtered.columns and 'CALL_TYPE' in df_filtered.columns:
            # Cross-tabulate sentiment and call type
            sentiment_by_type = pd.crosstab(df_filtered['CALL_TYPE'], df_filtered['client_sentiment'])
            
            # Convert to percentage
            sentiment_by_type_pct = sentiment_by_type.div(sentiment_by_type.sum(axis=1), axis=0) * 100
            
            # Reshape for plotting
            sentiment_by_type_pct = sentiment_by_type_pct.reset_index().melt(
                id_vars='CALL_TYPE', 
                var_name='Sentiment', 
                value_name='Percentage'
            )
            
            fig = px.bar(
                sentiment_by_type_pct,
                x='CALL_TYPE',
                y='Percentage',
                color='Sentiment',
                title='Sentiment Distribution by Call Type',
                color_discrete_map=color_map,
                barmode='group'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    if 'next_best_action' in df_filtered.columns and 'client_sentiment' in df_filtered.columns:
        st.markdown("<h2 class='sub-header'>Next Best Action by Sentiment</h2>", unsafe_allow_html=True)
        
        # Cross-tabulate next best action and sentiment
        action_by_sentiment = pd.crosstab(
            df_filtered['next_best_action'], 
            df_filtered['client_sentiment']
        )
        
        fig = px.bar(
            action_by_sentiment.reset_index().melt(
                id_vars='next_best_action', 
                var_name='Sentiment', 
                value_name='Count'
            ),
            x='next_best_action',
            y='Count',
            color='Sentiment',
            title='Next Best Action by Client Sentiment',
            color_discrete_map=color_map,
            barmode='group'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

# Tab 3: Topic Analysis
with tab3:
    st.markdown("<h2 class='sub-header'>Key Topics & Keywords</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'key_topics' in df_filtered.columns:
            # Flatten the key topics lists
            all_topics = []
            for topics in df_filtered['key_topics_list']:
                if isinstance(topics, list):
                    all_topics.extend(topics)
            
            # Count the frequency of each topic
            topic_counts = pd.Series(all_topics).value_counts().reset_index()
            topic_counts.columns = ['Topic', 'Count']
            topic_counts = topic_counts.head(10)  # Top 10 topics
            
            fig = px.bar(
                topic_counts,
                x='Count',
                y='Topic',
                orientation='h',
                title='Top 10 Key Topics',
                color='Count',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'keywords' in df_filtered.columns:
            # Flatten the keywords lists
            all_keywords = []
            for keywords in df_filtered['keywords_list']:
                if isinstance(keywords, list):
                    all_keywords.extend(keywords)
            
            # Count the frequency of each keyword
            keyword_counts = pd.Series(all_keywords).value_counts().reset_index()
            keyword_counts.columns = ['Keyword', 'Count']
            keyword_counts = keyword_counts.head(10)  # Top 10 keywords
            
            fig = px.bar(
                keyword_counts,
                x='Count',
                y='Keyword',
                orientation='h',
                title='Top 10 Keywords',
                color='Count',
                color_continuous_scale='Turbo'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    # Word cloud for keywords
    if 'keywords' in df_filtered.columns:
        st.markdown("<h2 class='sub-header'>Keyword Word Cloud</h2>", unsafe_allow_html=True)
        
        # Combine all keywords into a single string
        all_keywords_text = ' '.join(all_keywords)
        
        # Generate word cloud
        if all_keywords_text:
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                colormap='viridis',
                max_words=100
            ).generate(all_keywords_text)
            
            # Display the word cloud using matplotlib
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
    
    # Topic distribution by call type
    if 'CALL_TYPE' in df_filtered.columns and 'key_topics_list' in df_filtered.columns:
        st.markdown("<h2 class='sub-header'>Top Topics by Call Type</h2>", unsafe_allow_html=True)
        
        # Create a list of call types
        call_types = df_filtered['CALL_TYPE'].unique()
        
        # For each call type, find the top 5 topics
        topic_by_call_type = []
        
        for call_type in call_types:
            # Filter data for this call type
            call_type_df = df_filtered[df_filtered['CALL_TYPE'] == call_type]
            
            # Flatten the topics for this call type
            call_type_topics = []
            for topics in call_type_df['key_topics_list']:
                if isinstance(topics, list):
                    call_type_topics.extend(topics)
            
            # Count and get top 5 topics
            top_topics = pd.Series(call_type_topics).value_counts().head(5)
            
            # Add to the result
            for topic, count in top_topics.items():
                topic_by_call_type.append({
                    'Call Type': call_type,
                    'Topic': topic,
                    'Count': count
                })
        
        # Convert to DataFrame
        topic_by_call_type_df = pd.DataFrame(topic_by_call_type)
        
        if not topic_by_call_type_df.empty:
            fig = px.bar(
                topic_by_call_type_df,
                x='Call Type',
                y='Count',
                color='Topic',
                title='Top Topics by Call Type',
                barmode='group'
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

# Tab 4: Detailed Insights
with tab4:
    st.markdown("<h2 class='sub-header'>Compliance Metrics</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'recording_notice_given' in df_filtered.columns:
            recording_counts = df_filtered['recording_notice_given'].value_counts()
            recording_pct = recording_counts.get(True, 0) / recording_counts.sum() * 100 if recording_counts.sum() > 0 else 0
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=recording_pct,
                number={'font': {'size': 40}, 'suffix': '%', 'valueformat': '.1f'},
                title={'text': "Recording Notice Compliance", 'font': {'size': 24}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "royalblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 60], 'color': "red"},
                        {'range': [60, 80], 'color': "orange"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor="white",
                font={'color': "darkblue", 'family': "Arial"}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'language_detected' in df_filtered.columns and 'LANGUAGE' in df_filtered.columns:
            # Calculate language detection accuracy - handle edge cases
            if len(df_filtered) > 0:
                valid_rows = df_filtered.dropna(subset=['language_detected', 'LANGUAGE'])
                if len(valid_rows) > 0:
                    # Case-insensitive comparison
                    language_accuracy = (valid_rows['language_detected'].str.lower() == valid_rows['LANGUAGE'].str.lower()).mean() * 100
                else:
                    language_accuracy = 0
            else:
                language_accuracy = 0
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=language_accuracy,
                number={'font': {'size': 40}, 'suffix': '%', 'valueformat': '.1f'},
                title={'text': "Language Detection Accuracy", 'font': {'size': 24}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "royalblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 60], 'color': "red"},
                        {'range': [60, 80], 'color': "orange"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor="white",
                font={'color': "darkblue", 'family': "Arial"}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Add a note about language detection
            if language_accuracy == 0:
                st.info("No language detection data available or languages don't match. Check that 'language_detected' and 'LANGUAGE' columns have comparable values.")
    
    st.markdown("<h2 class='sub-header'>Client & Agent Engagement Metrics</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'client_word_count' in df_filtered.columns and 'agent_word_count' in df_filtered.columns:
            # Calculate average words per message
            df_filtered['client_words_per_message'] = df_filtered['client_word_count'] / df_filtered['client_message_count'].replace(0, 1)
            df_filtered['agent_words_per_message'] = df_filtered['agent_word_count'] / df_filtered['agent_message_count'].replace(0, 1)
            
            # Calculate average
            client_avg = df_filtered['client_words_per_message'].mean()
            agent_avg = df_filtered['agent_words_per_message'].mean()
            
            # Create data for comparison
            words_data = pd.DataFrame({
                'Role': ['Client', 'Agent'],
                'Average Words per Message': [client_avg, agent_avg]
            })
            
            fig = px.bar(
                words_data,
                x='Role',
                y='Average Words per Message',
                title='Average Words per Message',
                color='Role',
                color_discrete_sequence=['#FF6B6B', '#4ECDC4']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'client_message_count' in df_filtered.columns and 'agent_message_count' in df_filtered.columns:
            # Calculate average message count
            client_msg_avg = df_filtered['client_message_count'].mean()
            agent_msg_avg = df_filtered['agent_message_count'].mean()
            
            # Create data for comparison
            msg_data = pd.DataFrame({
                'Role': ['Client', 'Agent'],
                'Average Message Count': [client_msg_avg, agent_msg_avg]
            })
            
            fig = px.bar(
                msg_data,
                x='Role',
                y='Average Message Count',
                title='Average Messages per Call',
                color='Role',
                color_discrete_sequence=['#FF6B6B', '#4ECDC4']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Mobile app referral success
    if 'directed_to_mobile_app' in df_filtered.columns:
        st.markdown("<h2 class='sub-header'>Mobile App Referral Success</h2>", unsafe_allow_html=True)
        
        app_referral_counts = df_filtered['directed_to_mobile_app'].value_counts()
        app_referral_pct = app_referral_counts.get(True, 0) / app_referral_counts.sum() * 100 if app_referral_counts.sum() > 0 else 0
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=app_referral_pct,
            number={'font': {'size': 40}, 'suffix': '%', 'valueformat': '.1f'},
            title={'text': "Mobile App Referral Rate", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "royalblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': "red"},
                    {'range': [50, 80], 'color': "orange"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "green", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="white",
            font={'color': "darkblue", 'family': "Arial"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations section
    st.markdown("<h2 class='sub-header'>🚀 Actionable Recommendations</h2>", unsafe_allow_html=True)
    
    # Generate recommendations based on the data
    recommendations = []
    
    # Compliance recommendations
    if 'recording_notice_given' in df_filtered.columns:
        recording_compliance = df_filtered['recording_notice_given'].mean() * 100
        if recording_compliance < 90:
            recommendations.append({
                'category': 'Compliance',
                'recommendation': "Improve recording notice compliance rate which is currently below 90%",
                'priority': 'High',
                'impact': 'Legal and regulatory'
            })
    
    # Sentiment recommendations
    if 'client_sentiment' in df_filtered.columns:
        negative_sentiment_pct = (df_filtered['client_sentiment'] == 'negative').mean() * 100
        if negative_sentiment_pct > 10:
            recommendations.append({
                'category': 'Customer Experience',
                'recommendation': f"Address high negative sentiment rate ({negative_sentiment_pct:.1f}%) by improving agent training",
                'priority': 'High',
                'impact': 'Customer satisfaction and retention'
            })
    
    # Call duration recommendations
    if 'call_duration_estimate_seconds' in df_filtered.columns and 'CALL_TYPE' in df_filtered.columns:
        long_outbound_calls = df_filtered[
            (df_filtered['CALL_TYPE'].str.contains('outbound')) & 
            (df_filtered['call_duration_estimate_seconds'] > 180)
        ]
        if len(long_outbound_calls) > 0:
            recommendations.append({
                'category': 'Efficiency',
                'recommendation': "Optimize outbound call scripts to reduce call duration",
                'priority': 'Medium',
                'impact': 'Operational efficiency and cost reduction'
            })
    
    # Mobile app referral recommendations
    if 'directed_to_mobile_app' in df_filtered.columns:
        app_referral_rate = df_filtered['directed_to_mobile_app'].mean() * 100
        if app_referral_rate < 90:
            recommendations.append({
                'category': 'Digital Adoption',
                'recommendation': "Increase mobile app referrals during calls to improve digital engagement",
                'priority': 'Medium',
                'impact': 'Digital channel shift and customer self-service'
            })
    
    # Follow-up recommendations
    if 'follow_up_required' in df_filtered.columns:
        follow_up_rate = df_filtered['follow_up_required'].mean() * 100
        if follow_up_rate > 50:
            recommendations.append({
                'category': 'Process Optimization',
                'recommendation': "Review high follow-up requirement rate to identify process improvement opportunities",
                'priority': 'Medium',
                'impact': 'First-call resolution and customer effort reduction'
            })
    
    # Display recommendations as a table
    if recommendations:
        recommendations_df = pd.DataFrame(recommendations)
        
        # Add styling based on priority
        def style_priority(val):
            if val == 'High':
                return 'background-color: #FFEBEE; color: #D32F2F; font-weight: bold'
            elif val == 'Medium':
                return 'background-color: #FFF8E1; color: #FF8F00; font-weight: bold'
            else:
                return 'background-color: #E8F5E9; color: #388E3C; font-weight: bold'
        
        st.dataframe(
            recommendations_df.style.applymap(style_priority, subset=['priority']),
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("No specific recommendations generated based on the current filtered data.")
    
    # Add a transcript viewer section
    st.markdown("<h2 class='sub-header'>📝 Transcript Viewer</h2>", unsafe_allow_html=True)
    
    # Select a client to view their transcript
    if 'CLIENT_NAME' in df_filtered.columns and 'CLIENT_ID' in df_filtered.columns:
        # Create a selection option with Client Name and ID
        client_options = df_filtered.apply(lambda row: f"{row['CLIENT_NAME']} ({row['CLIENT_ID']})", axis=1).unique()
        selected_client = st.selectbox("Select a client to view transcript:", client_options)
        
        if selected_client:
            # Extract the client ID from the selection
            client_id = selected_client.split('(')[1].split(')')[0]
            
            # Get the transcript for this client
            transcript_row = df_filtered[df_filtered['CLIENT_ID'] == client_id].iloc[0]
            
            # Display client information
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Client Name:** {transcript_row['CLIENT_NAME']}")
                st.markdown(f"**Client ID:** {transcript_row['CLIENT_ID']}")
            
            with col2:
                st.markdown(f"**Call Type:** {transcript_row['CALL_TYPE']}")
                st.markdown(f"**Call Date:** {transcript_row['DATE_TIME']}")
            
            with col3:
                st.markdown(f"**Sentiment:** {transcript_row['client_sentiment']}")
                st.markdown(f"**Language:** {transcript_row['LANGUAGE']}")
            
            # Parse and display the transcript
            if isinstance(transcript_row['TRANSCRIPT'], str):
                try:
                    transcript = json.loads(transcript_row['TRANSCRIPT'])
                    
                    # Create a styled transcript view
                    st.markdown("### Conversation:")
                    
                    # Custom CSS for chat bubbles
                    st.markdown("""
                    <style>
                        .chat-container {
                            display: flex;
                            flex-direction: column;
                            gap: 10px;
                            margin-top: 20px;
                        }
                        .message {
                            padding: 10px 15px;
                            border-radius: 18px;
                            max-width: 80%;
                            position: relative;
                            margin: 5px 0;
                        }
                        .agent {
                            background-color: #E3F2FD;
                            border-bottom-left-radius: 5px;
                            align-self: flex-start;
                        }
                        .client {
                            background-color: #F5F5F5;
                            border-bottom-right-radius: 5px;
                            align-self: flex-end;
                            text-align: right;
                        }
                        .role {
                            font-size: 0.8em;
                            color: #757575;
                            margin-bottom: 3px;
                        }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Display messages
                    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
                    
                    for msg in transcript:
                        role = msg.get('role', '')
                        text = msg.get('text', '')
                        
                        bubble_class = "agent" if role == "agent" else "client"
                        role_display = "Agent" if role == "agent" else "Client"
                        
                        st.markdown(f"""
                        <div class="message {bubble_class}">
                            <div class="role">{role_display}</div>
                            {text}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error parsing transcript: {e}")
                    st.text(transcript_row['TRANSCRIPT'])
            else:
                st.warning("No transcript available for this client.")

# Footer with additional information
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>CRM Call Analysis Dashboard | Data based on CALL_TRANSCRIPT_LLM_RESPONSE_EXPANDED table</p>
    <p>❗ For any issues with this dashboard, please contact the Data Science team</p>
</div>
""", unsafe_allow_html=True) 