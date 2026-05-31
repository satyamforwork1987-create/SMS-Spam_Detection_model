import pickle
from pathlib import Path

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# -------------------------------------------------------------------------
# 1. TRAINING — Always retrain to avoid stale / unfitted pkl files
# -------------------------------------------------------------------------
vectorizer_path = Path("vectorizer.pkl")
model_path = Path("model.pkl")


def train_and_save_model():
    """Train on sample data and persist fresh, fitted pkl files."""
    training_messages = [
        # --- SPAM ---
        "Free entry! Claim your reward now by clicking here.",
        "Congratulations! You won a $30000 cash prize. Sign up today.",
        "URGENT: Your account has been suspended. Reply YES to reactivate.",
        "Win a brand new iPhone! Click the link to claim your prize.",
        "You have been selected for a special offer. Call now to redeem.",
        "Get a free vacation package. Limited time offer. Act now!",
        "Dear winner, you have won a lottery of $1,000,000. Claim now.",
        "FREE ringtones! Reply to get yours today.",
        "Exclusive deal! Buy now and get 80% off. Only today!",
        "You've been pre-approved for a $5000 loan. No credit check!",
        # --- HAM ---
        "Hi, your appointment with Dr. Smith is confirmed for tomorrow.",
        "Hey, are we still meeting up for lunch later today?",
        "Can you send me the report by this evening? Thanks.",
        "Mom says dinner is at 7. Don't be late!",
        "Reminder: team standup at 10 AM tomorrow.",
        "I'll be home around 6. Can you pick up some groceries?",
        "Your package has been shipped and will arrive by Friday.",
        "Let's catch up soon. It's been a while!",
        "Please review the document I sent and share your feedback.",
        "Your subscription has been renewed successfully.",
    ]
    labels = (["spam"] * 10) + (["ham"] * 10)

    # Fit vectorizer
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(training_messages)

    # Fit classifier
    model = MultinomialNB()
    model.fit(X_train, labels)

    # Persist
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return vectorizer, model


# Always train fresh — guarantees fitted objects on every cold start
# (fast enough: ~5 ms on 20 samples)
_vectorizer, _model = train_and_save_model()


# -------------------------------------------------------------------------
# 2. STREAMLIT UI
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGE_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
body {
    background: linear-gradient(135deg, #f2f7ff 0%, #f9fbff 100%);
}
.stButton > button {
    background-color: #0f62fe;
    color: white;
    border-radius: 12px;
    height: 48px;
    font-weight: 600;
    border: none;
    width: 100%;
}
.stButton > button:hover {
    background-color: #0353e9;
}
.stTextArea > div > div > textarea {
    border-radius: 16px;
}
.result-box {
    padding: 1.5rem;
    border-radius: 16px;
    margin-top: 1rem;
}
</style>
"""

st.markdown(PAGE_STYLE, unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("📖 How to use")
    st.write(
        "- Paste any SMS or email text in the box.\n"
        "- Click **Classify** to analyse it.\n"
        "- Review the label, confidence score, and details."
    )
    st.divider()
    st.subheader("Example messages")
    st.write("**Spam:** Free entry! Claim your reward now.")
    st.write("**Not spam:** Hi, your appointment with Dr. Smith is confirmed for tomorrow.")
    st.divider()
    st.subheader("Model details")
    st.write(f"- Vectorizer: `{_vectorizer.__class__.__name__}`")
    st.write(f"- Classifier: `{_model.__class__.__name__}`")
    st.write(f"- Training samples: 20 (10 spam / 10 ham)")
    st.divider()
    st.caption("Built with Streamlit · TF-IDF + Naïve Bayes")

# --- Main ---
st.markdown("## 📩 SMS / Email Spam Classifier")
st.markdown(
    "Paste a message below and click **Classify** to detect whether it is spam."
)

with st.expander("Why this app?"):
    st.write(
        "Spam messages often contain promotional language, suspicious links, or "
        "urgent call-to-action text. This app uses a TF-IDF vectorizer and a "
        "Multinomial Naïve Bayes classifier trained on labelled examples."
    )

# --- Input form ---
with st.form("message_form"):
    message = st.text_area(
        "Enter your message here",
        placeholder="e.g. Congratulations! You have won a free ticket. Reply YES to claim.",
        height=200,
    )
    submitted = st.form_submit_button("🔍 Classify message")

# --- Prediction ---
if submitted:
    if not message.strip():
        st.warning("⚠️ Please enter a message before classifying.")
    else:
        # Transform using the in-memory (always fitted) objects
        transformed = _vectorizer.transform([message])
        prediction = _model.predict(transformed)[0]

        # Confidence
        probabilities = _model.predict_proba(transformed)[0]
        pred_index = list(_model.classes_).index(prediction)
        confidence = probabilities[pred_index]

        is_spam = prediction.lower() == "spam"

        st.markdown("---")
        st.subheader("Prediction Result")

        col1, col2 = st.columns([2, 1])
        with col1:
            if is_spam:
                st.error(f"🚫 **Spam detected**")
            else:
                st.success(f"✅ **Not spam (Ham)**")

        with col2:
            st.metric("Confidence", f"{confidence * 100:.1f}%")

        with st.expander("View message & details"):
            st.write("**Message text:**")
            st.info(message)
            st.write(f"**Predicted label:** `{prediction}`")
            st.write(f"**Spam probability:** `{probabilities[list(_model.classes_).index('spam')] * 100:.1f}%`")
            st.write(f"**Ham probability:** `{probabilities[list(_model.classes_).index('ham')] * 100:.1f}%`")

        if is_spam:
            st.warning("⚠️ Be cautious with this message. It shows characteristics common in spam.")
        else:
            st.balloons()

st.markdown("---")
st.caption("SMS / Email Spam Classifier · TF-IDF + Multinomial Naïve Bayes · Built with Streamlit")