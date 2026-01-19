# Traffic-Live-AI-RAG-System
Project Overview
This project is developed for the Pathway Inter IIT hackathon challenge. It aims to solve urban traffic congestion by combining historical data analysis with real-time incident reporting. The system uses a Retrieval-Augmented Generation (RAG) approach to provide actionable AI-driven recommendations when new traffic incidents occur.

The core of the application is built on Pathway's philosophy of handling data that "thinks and adapts". By analyzing live incident reports against historical traffic patterns, the tool provides severity alerts, rerouting recommendations, and visualizes hotspots on an interactive map.

Key Features
Live Incident Reporting: Users can report incidents (accidents, floods, etc.) which are immediately appended to the traffic_live.csv dataset for processing.

AI Traffic Analysis: Utilizing a mock embedding system (as per current app.py implementation) to perform cosine similarity searches between live incidents and historical cases.

Dynamic Severity Alerts: Automated risk assessment that triggers "High" or "Medium" confidence alerts based on vehicle counts and incident severity.

Interactive Visualization: A live map powered by Pydeck that highlights incident locations using real-world coordinates (e.g., Bangalore junctions).

Actionable Insights: Recommended actions like "Deploy traffic police" or "Optimize signal timings" generated based on incident severity.

Tech Stack
Framework: Streamlit (UI/UX)
Data Processing: Pandas, NumPy
Machine Learning: Scikit-learn (Cosine Similarity for RAG)
Geospatial Visualization: Pydeck
Platform: Pathway (Contextual Intelligence)
Project Structure
app.py: The main Streamlit application containing the RAG logic, UI, and map visualization.
traffic_live.csv: A dynamic dataset that stores incoming live incident reports entered through the app.
traffic_original.csv: A historical dataset containing extensive traffic records (DateTime, Junction, Vehicles) used for similarity retrieval.
How It Works
Data Input: When an incident is logged in the Streamlit sidebar, the app captures the Junction ID, Incident Type, Severity, and Vehicle Count.

Historical Retrieval: The system scans traffic_original.csv to find similar past occurrences using cosine similarity.

Analysis: The AI evaluates the current severity. If the severity index is >= 4, it recommends immediate rerouting; otherwise, it suggests continued monitoring.

Mapping: The incident is plotted on the map using predefined junction coordinates.

Getting Started
1. Install dependencies:
pip install streamlit pandas numpy sklearn pydeck
2. Run the application:
streamlit run app.py
