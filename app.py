import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time


# === Load model and tokenizer ===
@st.cache_resource
def load_assets():
    model = load_model("next_word_model_v1.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_assets()

MAX_SEQ_LEN = 20  # Replace with your actual training value


# === Prediction Function ===
def predict_next_words(seed_text, num_words, animate=False):
    seed_text = seed_text.strip()
    generated_text = seed_text
    new_words = []

    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([generated_text])[0]
        token_list = pad_sequences([token_list], maxlen=MAX_SEQ_LEN - 1, padding='pre')

        predicted_probs = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted_probs)
        predicted_word = tokenizer.index_word.get(predicted_index, "")

        if not predicted_word:
            break

        generated_text += " " + predicted_word
        new_words.append(predicted_word)

        if animate:
            st.markdown(f"📝 **Current:** `{generated_text}`")
            time.sleep(0.5)

    return seed_text, new_words


# === Streamlit UI ===
st.set_page_config(page_title="Next Word Predictor", layout="centered")
st.title("🧠 Next Word Predictor")
st.markdown("Predict the next few words given a starting sentence using a trained LSTM model.")

# === Input Form ===
with st.form("prediction_form"):
    seed_text = st.text_input("Enter seed text:", value="How are you")
    num_words = st.slider("Number of words to predict:", 1, 20, 3)
    animate = st.checkbox("Show word-by-word prediction (slow)", value=False)
    submitted = st.form_submit_button("🔮 Generate Text")

    if submitted:
        if not seed_text.strip():
            st.warning("Please enter some seed text to begin prediction.")
        else:
            with st.spinner("Generating..."):
                result = predict_next_words(seed_text, num_words, animate=animate)
            if result:
                st.markdown("### 📄 Predicted Text")
                seed_part, generated_words = result
                highlighted = seed_part + " " + " ".join([f"**:blue[{word}]**" for word in generated_words])
                st.markdown(highlighted)

