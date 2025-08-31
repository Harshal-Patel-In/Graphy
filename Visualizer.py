import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
from utils.euler import has_eulerian_path, has_eulerian_cycle
from utils.random_walk import random_walk
import streamlit.components.v1 as components
import tempfile
import os
import json
import matplotlib.pyplot as plt
import io

st.set_page_config(layout="wide", page_title="Graphy- Graph Visualization")
st.title("🧠 EulerGraph: Interactive Graph Theory Explorer")


if "graph_data" not in st.session_state:
    st.session_state.graph_data = None
if "example" not in st.session_state:
    st.session_state.example = False
if "walk" not in st.session_state:
    st.session_state.walk = None
# Ensure there is a place to store CSV/DataFrame for other pages
if "graph_csv" not in st.session_state:
    st.session_state.graph_csv = None

# Sidebar Settings
st.sidebar.image("logo.png", width=400)
st.sidebar.header("Graph Settings")
use_directed = st.sidebar.checkbox("Use Directed Graph", value=False, key="directed")


# Example Graph button
if st.sidebar.button("Load Example Graph"):
    st.session_state.example = True
    df_example = pd.DataFrame({
        "source": ["A", "A", "B", "C", "D", "E"],
        "target": ["B", "C", "D", "E", "A", "A"]
    })
    st.session_state.graph_data = df_example
    st.session_state.graph_csv = df_example   # <-- keep DataFrame in session for graphy

# File Upload
uploaded_file = st.file_uploader("Upload CSV with 'source','target' columns", type="csv")
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if 'source' not in df.columns or 'target' not in df.columns:
            st.error("CSV must have 'source' and 'target' columns.")
            st.stop()
        st.session_state.graph_data = df
        st.session_state.graph_csv = df   # <-- keep DataFrame in session for graphy
        st.session_state.example = False
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

# Load Data
if st.session_state.graph_data is not None:
    # Visualizer expects a DataFrame; ensure df variable is a DataFrame
    if isinstance(st.session_state.graph_data, pd.DataFrame):
        df = st.session_state.graph_data
    elif st.session_state.graph_csv is not None:
        df = st.session_state.graph_csv
    else:
        st.error("Internal data format unexpected. Please re-upload or reload.")
        st.stop()
else:
    st.info("Upload a CSV or load an example graph.")
    st.stop()

# Create Graph
G = nx.DiGraph() if use_directed else nx.Graph()
G.add_edges_from(zip(df['source'], df['target']))

if G.number_of_nodes() == 0:
    st.warning("Graph has no nodes.")
    st.stop()

# Graph Info
with st.expander("📊 Graph Info"):
    st.write(f"Number of nodes: {G.number_of_nodes()}")
    st.write(f"Number of edges: {G.number_of_edges()}")
    st.write(f"Connected: {'Yes' if nx.is_connected(G.to_undirected()) else 'No'}")

# Euler Results
col1, col2 = st.columns(2)
with col1:
    st.subheader("Euler Path?")
    st.success(has_eulerian_path(G))

with col2:
    st.subheader("Euler Cycle?")
    st.success(has_eulerian_cycle(G))

# Random Walk
st.subheader("🚶 Random Walk")
start_node = st.selectbox("Select start node", list(G.nodes), key="start_node")
steps = st.slider("Steps", 1, 20, 5, key="steps")

if st.button("Run Random Walk"):
    try:
        walk = random_walk(G, start_node, steps)
        st.session_state.walk = walk  # Save to session state
    except Exception as e:
        st.error(f"Random walk failed: {e}")

# Show walk if already run
if st.session_state.walk:
    st.info(" → ".join(st.session_state.walk))
    st.download_button("Download Walk", data="\n".join(st.session_state.walk),
                       file_name="random_walk.txt")

# ==== Graph Visualization with Pyvis ====
net = Network(height="600px", width="100%", directed=use_directed)
net.from_nx(G)

#displaying graph
html = net.generate_html()
components.html(html, height=490)


# ==== Export Section ====
st.subheader("📥 Export Graph")

with st.expander("Export Graph"):
     img_buf = io.BytesIO()
     plt.figure(figsize=(8, 6))
     nx.draw(G, with_labels=True, node_color='skyblue', edge_color='gray',
             node_size=700, font_size=10)
     plt.tight_layout()
     plt.savefig(img_buf, format="png")
     st.download_button("📸 Download PNG", data=img_buf.getvalue(),
                        file_name="graph.png", mime="image/png")
     plt.close()

     # JSON
     graph_data = nx.node_link_data(G)
     graph_json = json.dumps(graph_data, indent=2)
     st.download_button("🧾 Download JSON", data=graph_json,
                        file_name="graph.json", mime="application/json")

     # GraphML
     graphml_buf = io.BytesIO()
     nx.write_graphml(G, graphml_buf)
     st.download_button("📄 Download GraphML", data=graphml_buf.getvalue(),
                        file_name="graph.graphml", mime="application/graphml+xml")

     # Adjacency List
     adj_list_text = "\n".join(nx.generate_adjlist(G))
     st.download_button(
         "📜 Download Adjacency List",
         data=adj_list_text,
         file_name="graph.adjlist",
         mime="text/plain"
     )

# Interactive JSON Viewer
st.subheader("View JSON Format")
with st.expander("📂 View JSON Data (Node-Link Format)"):
    st.json(graph_data)
