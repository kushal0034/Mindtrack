import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from deepface import DeepFace
import tempfile
import io

# -----------------------------------------------------------
# SECTION 1: MODEL TRAINING/LOADING (DEMO STUBS)
# -----------------------------------------------------------

def train_or_load_demo_models():
    """
    Returns small demo-trained NLP models for proof-of-concept.
    Swap this for joblib.load() or transformer pipelines in production.
    """
    data = {
        'text': [
            "I feel depressed, empty and have no energy.",
            "Life is good. I'm having a wonderful day!",
            "I'm not sure what I'm trying to say, everything is confusing.",
            "Today was fine. Nothing special, just work.",
            "Can't focus at all. Thoughts are scattered.",
            "I don’t feel like getting out of bed anymore.",
            "Trying to stay strong. Today was okay.",
            "There is no point, I can't concentrate.",
            "Completed a project. I feel accomplished.",
            "I keep losing my train of thought."
        ],
        'label': ['Depressed', 'Control', 'Distracted', 'Control', 'Distracted',
                  'Depressed', 'Control', 'Distracted', 'Control', 'Distracted']
    }
    df = pd.DataFrame(data)
    # Text vectorization
    tfidf = TfidfVectorizer(stop_words='english', max_features=200)
    X = tfidf.fit_transform(df['text'])
    # Depression/Anxiety classifier: (Depressed vs Control)
    y_main = df['label'].map({'Depressed': 1, 'Control': 0, 'Distracted': 0})
    text_clf = RandomForestClassifier(n_estimators=30, random_state=42).fit(X, y_main)
    # Cognitive Alignment classifier: (Distracted/Disorganized vs Aligned)
    y_cog = df['label'].map({'Depressed': 0, 'Control': 0, 'Distracted': 1})
    cog_clf = LogisticRegression(max_iter=300).fit(X, y_cog)
    return tfidf, text_clf, cog_clf

tfidf, text_clf, cog_clf = train_or_load_demo_models()

# -----------------------------------------------------------
# SECTION 2: TEXT ANALYSIS HELPERS
# -----------------------------------------------------------

def predict_post(text):
    """
    Predicts depression/anxiety and cognitive alignment status from a single text post.
    Returns both predictions as labels.
    """
    X = tfidf.transform([text])
    status_pred = text_clf.predict(X)[0]
    status_prob = text_clf.predict_proba(X)[0][1]
    status_label = "Depressed/Anxious" if status_pred else "Control"
    cog_pred = cog_clf.predict(X)[0]
    cog_prob = cog_clf.predict_proba(X)[0][1]
    cog_label = "Distracted/Disorganized" if cog_pred else "Cognitively Aligned"
    return status_label, status_prob, cog_label, cog_prob

def batch_predict_texts(df):
    """
    Runs prediction on every row of DataFrame (with column 'text').
    Adds columns for predicted depression/anxiety and cognitive alignment.
    """
    X = tfidf.transform(df['text'])
    df['Predicted_Status'] = np.where(text_clf.predict(X), "Depressed/Anxious", "Control")
    df['Status_Prob'] = text_clf.predict_proba(X)[:,1]
    df['Cognitive_Alignment'] = np.where(cog_clf.predict(X), "Distracted/Disorganized", "Cognitively Aligned")
    df['CogAlign_Prob'] = cog_clf.predict_proba(X)[:,1]
    return df

def create_wordcloud(text_list, max_words=120):
    """
    Generates a word cloud from a list of strings (posts).
    """
    all_text = " ".join(text_list)
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=max_words).generate(all_text)
    return wordcloud

# -----------------------------------------------------------
# SECTION 3: IMAGE/EMOTION ANALYSIS HELPERS
# -----------------------------------------------------------

def analyze_face_image(image_file):
    """
    Uses DeepFace to analyze a face image and returns dominant emotion and probabilities.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(image_file.read())
        tmp_path = tmp_file.name
    result = DeepFace.analyze(img_path=tmp_path, actions=['emotion'], enforce_detection=False)
    return result['dominant_emotion'], result['emotion']

# -----------------------------------------------------------
# SECTION 4: STREAMLIT UI LOGIC
# -----------------------------------------------------------

st.set_page_config(page_title="MindTrack: Mental Health Early Detection", layout="wide")
st.title("🧠 MindTrack – Social Media Early Detection of Depression & Anxiety")
st.markdown("""
MindTrack is a **research prototype** designed to help spot potential signs of depression, anxiety, and cognitive distraction using social media posts and images.
- **Text Analysis:** Classifies posts as 'Depressed/Anxious' or 'Control' and checks for signs of cognitive distraction.
- **Image Analysis:** Detects facial emotions from selfies/profile images.
- **Bulk Upload:** Analyze trends for a user or population using CSV files.

---
""")

with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    - Choose your analysis mode (Single post, Bulk CSV, or Image).
    - For text, results are for **support/awareness only** – not for diagnosis.
    - For image, make sure the face is visible and clear for emotion detection.
    - Download your analyzed results for further exploration.
    ---
    """)
    st.info("Replace the demo models with your production models for clinical-grade results.")

choice = st.sidebar.radio("Choose Analysis Mode", ["Single Post Analysis", "Bulk Posts (CSV)", "Image (Facial Emotion)"])

# ---- SINGLE POST UI ----
if choice == "Single Post Analysis":
    st.header("🔎 Analyze a Single Social Media Post")
    st.markdown("Enter or paste a post below (supports English only):")
    example_post = "I don't feel like doing anything today. Everything is so overwhelming."
    text = st.text_area("Social Media Post", value=example_post, height=120)

    if st.button("Analyze Text"):
        if not text.strip():
            st.warning("Please enter a post to analyze.")
        else:
            status_label, status_prob, cog_label, cog_prob = predict_post(text)
            st.subheader("🔹 Prediction Results")
            st.write(f"**Depression/Anxiety Prediction:** {status_label} (Probability: {status_prob:.2f})")
            st.write(f"**Cognitive Alignment:** {cog_label} (Probability: {cog_prob:.2f})")

            st.markdown("**What does this mean?**")
            if status_label == "Depressed/Anxious":
                st.info("This post contains language similar to people experiencing depression or anxiety. Consider reaching out for support or talking to someone you trust.")
            else:
                st.success("No strong signs of depression/anxiety detected. If you feel otherwise, consider sharing your feelings with someone.")

            if cog_label == "Distracted/Disorganized":
                st.warning("This post shows signs of cognitive distraction. The user may be writing while distracted or under cognitive strain.")
            else:
                st.info("The post appears cognitively aligned (clear and focused).")

            st.markdown("**Word Cloud from your Post:**")
            wordcloud = create_wordcloud([text])
            fig, ax = plt.subplots(figsize=(7,3))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

# ---- BULK CSV UI ----
elif choice == "Bulk Posts (CSV)":
    st.header("📊 Analyze Multiple Posts from a CSV File")
    st.markdown("""
    - Upload a `.csv` file with at least one column: `text`  
    - Optionally include `post_id`, `user_id`, and `label`
    - [Download a sample CSV to get started](https://gist.githubusercontent.com/sambernke/16dbcd0c55363877fffd9652b67704e8/raw/ce02a0417236279947e6c581cb7afc18a08b3a27/mindtrack_sample.csv)
    """)
    csv_file = st.file_uploader("Upload CSV", type=['csv'])

    if csv_file:
        try:
            df = pd.read_csv(csv_file)
            if "text" not in df.columns:
                st.error("CSV must have a 'text' column. Please check your file.")
            else:
                st.write("First few rows of your file:", df.head())
                if st.button("Run Bulk Analysis"):
                    df = batch_predict_texts(df)
                    st.success("Bulk analysis complete! See summary and download your results below.")
                    st.dataframe(df[["post_id", "user_id", "text", "Predicted_Status", "Cognitive_Alignment"]].head(10))

                    # Trends - Depression/Anxiety
                    st.markdown("### 📈 Emotional Trend")
                    fig1, ax1 = plt.subplots()
                    df['Predicted_Status'].value_counts().plot(kind='bar', ax=ax1, color=['dodgerblue', 'purple'])
                    ax1.set_ylabel('Count')
                    ax1.set_title('Posts classified as Depressed/Anxious vs Control')
                    st.pyplot(fig1)

                    # Trends - Cognitive Alignment
                    st.markdown("### 🧩 Cognitive Alignment Trend")
                    fig2, ax2 = plt.subplots()
                    df['Cognitive_Alignment'].value_counts().plot(kind='bar', ax=ax2, color=['orangered', 'forestgreen'])
                    ax2.set_ylabel('Count')
                    ax2.set_title('Posts classified as Distracted/Aligned')
                    st.pyplot(fig2)

                    # Combined word cloud
                    st.markdown("### ☁️ Word Cloud from All Posts")
                    wc = create_wordcloud(df['text'].tolist())
                    st.image(wc.to_array(), use_column_width=True)

                    # Download
                    csv_download = df.to_csv(index=False)
                    st.download_button("Download Results as CSV", csv_download, file_name="mindtrack_results.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Error processing file: {e}")

# ---- IMAGE UI ----
elif choice == "Image (Facial Emotion)":
    st.header("🖼️ Analyze Emotions from a Face Image")
    st.markdown("""
    - Upload a clear photo of a face (selfie or profile pic)
    - The app uses DeepFace for emotion detection
    - [Learn about DeepFace](https://github.com/serengil/deepface)
    """)
    img_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if img_file is not None:
        st.image(img_file, caption="Uploaded Image", width=320)
        if st.button("Analyze Image"):
            try:
                dom_emotion, all_emotions = analyze_face_image(img_file)
                st.markdown(f"**Detected Dominant Emotion:** :violet[{dom_emotion.capitalize()}]")
                st.markdown("**Emotion Probability Distribution:**")
                st.json(all_emotions)
                # Plot
                fig3, ax3 = plt.subplots(figsize=(5,2.5))
                labels = list(all_emotions.keys())
                values = list(all_emotions.values())
                ax3.barh(labels, values)
                ax3.set_xlabel("Probability (%)")
                ax3.set_title("Detected Emotion Probabilities")
                st.pyplot(fig3)
            except Exception as e:
                st.error(f"Could not analyze image: {e}")

# ---- END OF UI ----

st.markdown("""
---
:warning: **Disclaimer:**  
MindTrack is a research tool, not a substitute for mental health advice. If you are concerned for yourself or others, please reach out to a qualified mental health professional.

---
**References & Libraries:**
- CLPsych Dataset - UCI Machine Learning Repository
- Reddit Depression Dataset - Kaggle
- DeepFace (Serengil, 2023)
- BERT, RoBERTa: HuggingFace Transformers
- Python libraries: pandas, numpy, scikit-learn, matplotlib, wordcloud, streamlit

Developed as a template for social media-based mental health signal detection projects.
""")
