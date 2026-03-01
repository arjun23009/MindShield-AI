import streamlit as st
from transformers import pipeline
import pandas as pd

# Page setup
st.set_page_config(page_title="MindShield AI", layout="wide")

st.title("MindShield AI - Mental Health Monitor")
st.markdown("### AI-powered early mental health detection system")

# Load emotion model (cached so it doesn't reload every time)
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base"
    )

emotion = load_model()

# User input
user_input = st.text_area("How are you feeling today?")

# Analyze button
if st.button("Analyze"):
    if user_input:

        # Emotion detection
        result = emotion(user_input)
        label = result[0]['label']
        score = result[0]['score']

        st.write(f"Detected Emotion: {label}")
        st.write(f"Confidence: {score:.2f}")

        # Stress score logic
        stress_map = {
            "sadness": 80,
            "anger": 85,
            "fear": 75,
            "neutral": 40,
            "joy": 10,
            "love": 15
        }

        stress_score = stress_map.get(label, 50)

        st.subheader(f"Stress Score: {stress_score}/100")

        # Risk display
        if stress_score > 70:
            st.error("⚠️ High stress detected. Consider taking a break or talking to someone.")
        elif stress_score > 40:
            st.warning("Moderate stress level. Monitor your emotional health.")
        else:
            st.success("You are doing well.")

        # Personalized recommendation
        st.subheader("Personalized Recommendation")

        if stress_score > 70:
            st.write("• Take a short break and reduce workload.")
            st.write("• Try breathing or meditation.")
            st.write("• Talk to a trusted friend or mentor.")
        elif stress_score > 40:
            st.write("• Maintain proper sleep and routine.")
            st.write("• Limit screen and social media time.")
        else:
            st.write("• Continue your current healthy habits.")

        # Save data
        df = pd.DataFrame({
            "text": [user_input],
            "emotion": [label],
            "stress": [stress_score]
        })

        df.to_csv("data.csv", mode="a", header=False, index=False)

        # Stress trend graph
        try:
            data = pd.read_csv("data.csv", names=["text", "emotion", "stress"])
            st.subheader("Stress Trend Over Time")
            st.line_chart(data["stress"])
        except:
            pass