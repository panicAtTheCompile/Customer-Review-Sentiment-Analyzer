import streamlit as st
import joblib
st.set_page_config(
    page_title="Customer Review Sentiment Analyzer",
    page_icon="🛒",
    layout="wide"
)
# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
st.sidebar.title("📊 Model Information")

st.sidebar.markdown("""
### Model
- Logistic Regression

### Vectorizer
- TF-IDF

### Vocabulary
- 5,000 Features

### Classes
- 😊 Positive
- 😐 Neutral
- 😡 Negative
""")
st.title("🛒 Customer Review Sentiment Analyzer")

st.write(
    "Enter a customer review and predict whether it is Positive, Neutral, or Negative."
)
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Model Performance")

st.sidebar.metric("Accuracy","91.6%")

examples = {
    "Write my own": "",
    "😊 Positive Example": "Excellent quality and fast delivery. Highly recommended!",
    "😐 Neutral Example": "The product works as expected but shipping was delayed.",
    "😡 Negative Example": "Very poor quality. Completely disappointed with the purchase."
}

selected = st.selectbox(
    "Choose an example or write your own",
    examples.keys()
)

review = st.text_area(
    "Customer Review",
    value=examples[selected],
    height=180
)

if st.button("Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:

        review_vec = vectorizer.transform([review])

        prediction = model.predict(review_vec)[0]

        probabilities = model.predict_proba(review_vec)[0]

        confidence = max(probabilities) * 100

        st.subheader("Prediction")

        if prediction == "Positive":
            st.success(f"😊 {prediction}")

        elif prediction == "Neutral":
            st.info(f"😐 {prediction}")

        else:
            st.error(f"😡 {prediction}")

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.subheader("Prediction Probabilities")

        labels = model.classes_

        for label, prob in zip(labels, probabilities):

            st.write(f"{label}: {prob*100:.2f}%")

            st.progress(float(prob))

st.markdown("---")

st.subheader("⚙️ How It Works")

st.markdown("""
1. User enters a customer review.
2. The review is transformed using **TF-IDF vectorization**.
3. A trained **Logistic Regression** model predicts the sentiment.
4. The application displays the predicted class and confidence score.
""")

st.markdown("---")
st.caption("Built with Streamlit • Scikit-learn • TF-IDF • Logistic Regression")