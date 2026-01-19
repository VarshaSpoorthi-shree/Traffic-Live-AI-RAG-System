import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

JUNCTION_COORDS = {
    1: (12.9716, 77.5946),  # Bangalore
    2: (12.9352, 77.6245),
    3: (12.9982, 77.5920),
    4: (12.9141, 77.6446),
    5: (12.9763, 77.6033),
    6: (12.9900, 77.6100),
    7: (12.9600, 77.5800),
    8: (12.9500, 77.6200),
    9: (12.9800, 77.6500),
    10: (13.0000, 77.5700)
}

# ===============================
# MOCK EMBEDDING SETUP
# ===============================
EMBEDDING_DIM = 384
np.random.seed(42)

def mock_embedding():
    return np.random.rand(EMBEDDING_DIM)

# ===============================
# LIVE DATASET (CSV-based)
# ===============================
DATA_FILE = "traffic_original.csv"
LIVE_DATA_FILE = "traffic_live.csv"

if not os.path.exists(DATA_FILE):
    pd.DataFrame({"text": []}).to_csv(DATA_FILE, index=False)
if not os.path.exists(LIVE_DATA_FILE):
    pd.DataFrame({"text": []}).to_csv(LIVE_DATA_FILE, index=False)


def load_data():
    # Load original structured data
    orig_df = pd.read_csv(DATA_FILE)

    # Convert structured data to text
    orig_df["text"] = (
        "Time " + orig_df.iloc[:, 0].astype(str) +
        " | Traffic Level " + orig_df.iloc[:, 1].astype(str)
    )

    if os.path.getsize(LIVE_DATA_FILE) == 0:
        live_df = pd.DataFrame({"text": []})
    else:
        live_df = pd.read_csv(LIVE_DATA_FILE)

    # Combine
    df = pd.concat([orig_df[["text"]], live_df], ignore_index=True)

    df = df.dropna(subset=["text"])
    df["embedding"] = df["text"].apply(lambda x: mock_embedding())

    return df




# ===============================
# RETRIEVAL FUNCTION
# ===============================
def retrieve_similar_cases(query, df, k=3):
    query_emb = mock_embedding()
    similarities = cosine_similarity(
        [query_emb],
        list(df["embedding"])
    )[0]
    df = df.copy()
    df["similarity"] = similarities
    return df.sort_values("similarity", ascending=False).head(k)

def get_color(severity):
    if severity >= 4:
        return [255, 0, 0]      # Red
    elif severity == 3:
        return [255, 165, 0]    # Orange
    else:
        return [0, 200, 0]      # Green

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(page_title="Traffic Live RAG AI", layout="wide")

st.title("🚦 Traffic Live AI – RAG System")
st.markdown("**Real-time traffic incident analysis using Live RAG architecture**")

st.sidebar.header("📡 Live Incident Input")

junction = st.sidebar.number_input("🚦 Junction ID", min_value=1, max_value=10, value=1)
incident = st.sidebar.text_area(
    "📝 Incident Description",
    placeholder="Describe the live traffic issue here..."
)
severity = st.sidebar.slider("⚠️ Severity Level", 1, 5, 3)
vehicles = st.sidebar.number_input("🚗 Live Vehicle Count", value=50)


if st.sidebar.button("🚨 Analyze Traffic"):
        # Save live incident to dataset
    new_entry = pd.DataFrame({
        "text": [f"Junction {junction}: {incident} (Severity {severity}, Vehicles {vehicles})"]
    })
    lat, lon = JUNCTION_COORDS[junction]

    map_df = pd.DataFrame({
        "lat": [lat],
        "lon": [lon],
        "severity": [severity],
        "color": [get_color(severity)]
    })



    if not os.path.exists(LIVE_DATA_FILE) or os.path.getsize(LIVE_DATA_FILE) == 0:
        live_df = pd.DataFrame({"text": []})
    else:
        live_df = pd.read_csv(LIVE_DATA_FILE)

    live_df = pd.concat([live_df, new_entry], ignore_index=True)
    live_df.to_csv(LIVE_DATA_FILE, index=False)

    hist_df = load_data()

    st.subheader(f"📍 Junction {junction} – Live Alert")

    st.markdown(f"""
    **Incident:** {incident}
    **Severity:** {severity}
    **Live Vehicles:** {vehicles}
    """)

    similar_cases = retrieve_similar_cases(incident, hist_df)

    st.subheader("📚 Retrieved Historical Traffic Cases")
    for _, row in similar_cases.iterrows():
        st.markdown(f"- {row['text']}")

    st.subheader("🤖 AI Traffic Analysis")

    if severity >= 4:
        st.error("High congestion expected. Immediate rerouting recommended.")
        action = "Deploy traffic police and reroute vehicles"
        confidence = "High"
    else:
        st.warning("Moderate congestion. Monitor traffic flow.")
        action = "Monitor and optimize signal timings"
        confidence = "Medium"

    st.markdown(f"""
    **Recommended Action:** {action}
    **Confidence Level:** {confidence}
    """)
    st.subheader("🗺️ Incident Location")

    import pydeck as pdk

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position='[lon, lat]',
        get_color='color',
        get_radius=300,
        pickable=True
    )

    view_state = pdk.ViewState(
        latitude=lat,
        longitude=lon,
        zoom=12
    )

    st.subheader("🗺️ Traffic Severity Map")
    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view_state
    ))



st.markdown("---")
st.caption("Hackathon Prototype | Live RAG Architecture | Pathway + AI")
