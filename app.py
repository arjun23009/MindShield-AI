import streamlit as st
from transformers import pipeline
import pandas as pd

# Page setup
st.set_page_config(page_title="MindShield AI", layout="wide")

st.title("MindShield AI - Mental Health Monitor")
st.markdown("### AI-powered early mental health detection system")

# Load model (cached)
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base"
    )

emotion = load_model()

# Input
user_input = st.text_area("How are you feeling today?")

# Analyze
if st.button("Analyze"):
    if user_input:

        # Emotion detection
        result = emotion(user_input)
        label = result[0]['label']
        score = result[0]['score']

        st.write(f"Detected Emotion: {label}")
        st.write(f"Confidence: {score:.2f}")

        # Stress score
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

        # Risk alert
        if stress_score > 70:
            st.error("⚠️ High stress detected. Consider taking a break or talking to someone.")
        elif stress_score > 40:
            st.warning("Moderate stress level.")
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

        # -------- Advanced emotional detection --------
        text_lower = user_input.lower()

        heartbreak_keywords = [
            "lonely", "miss", "heartbreak", "alone", "isolated",
            "breakup", "missing someone", "lost someone"
        ]

        overthinking_keywords = [
            "overthink", "overthinking", "imagining",
            "scenario", "future fear", "anxious about future"
        ]

        # Heartbreak support
        if any(keyword in text_lower for keyword in heartbreak_keywords):
            st.subheader("Emotional Support: Loneliness & Heartbreak")
            st.write("It is normal to miss someone deeply. Emotional attachment takes time to heal.")
            st.write("Focus on your growth and identity. Over time, emotional pain reduces.")
            st.write("Physical activity and social interaction can speed emotional recovery.")

        # Overthinking detection
        if any(keyword in text_lower for keyword in overthinking_keywords):
            st.subheader("Mental Coaching: Overthinking Detected")
            st.write("Your mind is stuck in repetitive thinking. Try grounding techniques.")
            st.write("Write your thoughts in a journal to release mental pressure.")
            st.write("Focus on action instead of imagined scenarios.")

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