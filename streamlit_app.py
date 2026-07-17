import streamlit as st
from transformers import pipeline

# Configure the page settings
st.set_page_config(page_title="🚗 AI Traffic Advisor", page_icon="🚦")
st.title("🚗 AI Traffic Advisor")
st.write("Get effective driving advice using GPT-2")

# Cache the model pipeline so it only loads once
@st.cache_resource
def load_model():
    try:
        return pipeline("text-generation", model="gpt-2")
    except (OSError, Exception) as e:
        st.warning("⚠️ AI model unavailable. Using heuristic-based advice instead.")
        return None

generator = load_model()

# 1. User Inputs for Factors & Preferences
st.subheader("📋 Trip Details")
col1, col2 = st.columns(2)

with col1:
    route = st.text_input("Departure & Destination:", placeholder="e.g., Downtown to Airport")
    weather = st.selectbox("Current Weather:", ["Clear Sky", "Heavy Rain", "Snow/Ice", "Foggy"])
    departure_time = st.time_input("Planned Departure Time:")

with col2:
    priority = st.selectbox("User Preference:", ["Fastest Route", "Avoid Tolls", "Eco-Friendly / Transit"])
    is_peak = st.checkbox("Is this during local rush hour? (e.g., 7-9 AM, 4-6 PM)")

# 2. Rule-Based Heuristics (Estimating Delays)
base_delay = 0
alternatives = []

if is_peak:
    base_delay += 25
    alternatives.append("Departing 1 hour later to skip the rush hour peak.")

if weather == "Heavy Rain":
    base_delay += 15
    alternatives.append("Using the highway route instead of local backroads prone to flooding.")
elif weather == "Snow/Ice":
    base_delay += 35
    alternatives.append("Taking the local Subway/Train network to ensure maximum safety.")
elif weather == "Foggy":
    base_delay += 10

# 3. Generate Smart Advisor Response
if st.button("Analyze Best Travel Time"):
    if route:
        with st.spinner("Calculating optimal travel window..."):
            # Construct a structured prompt for GPT-2
            prompt = (
                f"Route: {route}. Weather: {weather}. Preference: {priority}. "
                f"Estimated Traffic Delay: {base_delay} minutes. "
                f"Therefore, the best driving advice for this trip is:"
            )
            
            results = generator(prompt, max_length=150, num_return_sequences=1, truncation=True) if generator else None
            ai_advice = results[0]["generated_text"][len(prompt):].strip() if results else None
            
            # Display Results
            st.markdown("---")
            st.subheader("🚦 Advisor Assessment")
            
            # Metric Callouts
            st.metric(label="Estimated Added Delay", value=f"{base_delay} mins")
            
            # Alternatives Section
            if alternatives:
                st.markdown("**💡 Recommended Alternatives:**")
                for alt in alternatives:
                    st.write(f"- {alt}")
            
            # AI Synthesis
            st.markdown("**🤖 AI Travel Synthesis:**")
            if ai_advice:
                st.write(ai_advice)
            else:
                fallback_advice = "Based on current conditions: "
                if is_peak:
                    fallback_advice += "Expect significant delays during rush hour. "
                if weather != "Clear Sky":
                    fallback_advice += f"Exercise caution in {weather.lower()} conditions. "
                fallback_advice += "Drive safely and maintain extra stopping distance."
                st.write(fallback_advice)
    else:
        st.warning("Please enter your departure and destination route.")