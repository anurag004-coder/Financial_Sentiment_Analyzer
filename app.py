import streamlit as st
import pandas as pd
import os
from transformers import pipeline

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Financial Sentiment Analyzer",
    page_icon="📈",
    layout="centered"
)

# --- 2. Load the AI Model ---
@st.cache_resource
def load_model():
    # Using a pre-trained financial model optimized for CPU web deployment
    model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    return pipeline("sentiment-analysis", model=model_name)

with st.spinner("Loading AI Model..."):
    nlp_pipeline = load_model()

# --- 3. App Header ---
st.title("📈 Context-Aware Financial Sentiment")
st.write("Analyze business and financial text for hidden market sentiment.")

# --- 4. Single Sentence Analysis ---
st.subheader("1. Single Sentence Analysis")
user_input = st.text_area(
    "Enter a financial sentence:",
    placeholder="e.g., Operating profit fell to EUR 35.4 mn from EUR 68.8 mn in 2007.",
    height=100
)

if st.button("Analyze Single Sentence"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing context..."):
            prediction = nlp_pipeline(user_input)[0]
            label = prediction['label'].upper()
            score = prediction['score']
            
            st.markdown("### Result")
            if label == "POSITIVE":
                st.success(f"**{label}** (Confidence: {score:.2f})")
            elif label == "NEGATIVE":
                st.error(f"**{label}** (Confidence: {score:.2f})")
            else:
                st.info(f"**{label}** (Confidence: {score:.2f})")

st.divider()

# --- 5. Batch Processing (Auto-Loaded CSV with Encoding Fix) ---
st.subheader("2. Batch Processing (Built-in Dataset)")
st.write("Automatically analyzing multiple sentences from the `financial_phrasebank.csv` repository file.")

# Define the file path
file_path = "financial_phrasebank.csv"

# Check if the file exists in the GitHub repository
if os.path.exists(file_path):
    try:
        # Added encoding='ISO-8859-1' to match your working notebook configuration precisely
        df = pd.read_csv(file_path, header=None, names=["True_Sentiment", "Sentence"], encoding='ISO-8859-1')
        
        st.write("Dataset Preview:")
        st.dataframe(df.head())
        
        # Slider to choose how many rows to process
        num_rows = st.slider("Select number of rows to analyze:", min_value=1, max_value=50, value=5)
        
        if st.button("Run Batch Analysis"):
            with st.spinner(f"Analyzing {num_rows} sentences..."):
                # Extract the text list
                sentences = df['Sentence'].head(num_rows).astype(str).tolist()
                
                # Run the NLP model on the batch
                results = nlp_pipeline(sentences)
                
                # Create a new dataframe for the results
                results_df = df.head(num_rows).copy()
                results_df['Predicted_Sentiment'] = [res['label'].upper() for res in results]
                results_df['Confidence'] = [round(res['score'], 2) for res in results]
                
                st.success("Batch analysis complete!")
                
                # Highlight colors based on sentiment
                def color_sentiment(val):
                    if val == 'POSITIVE': return 'background-color: #d4edda'
                    elif val == 'NEGATIVE': return 'background-color: #f8d7da'
                    return 'background-color: #e2e3e5'
                    
                # Display the styled dataframe
                st.dataframe(
                    results_df[['Sentence', 'True_Sentiment', 'Predicted_Sentiment', 'Confidence']]
                    .style.map(color_sentiment, subset=['Predicted_Sentiment'])
                )
    except Exception as e:
        st.error(f"An error occurred while reading the file: {e}")
else:
    st.error(f"Dataset not found! Please ensure `{file_path}` is uploaded to your GitHub repository alongside `app.py`.")
