"""
Streamlit app for AzLegalRAG.
Interactive Q&A interface for Azerbaijani legal documents.
"""

import streamlit as st
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embed import load_vectorstore
from generate import create_rag_chain, get_llm
from retrieve import semantic_search

# Page config
st.set_page_config(
    page_title="AzLegalRAG - Azerbaijani Legal Q&A",
    page_icon="AZ",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.source-box {
    background-color: #f0f2f6;
    border-radius: 5px;
    padding: 10px;
    margin: 5px 0;
}
.answer-box {
    background-color: #e8f4f8;
    border-radius: 10px;
    padding: 20px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("AzLegalRAG")
st.markdown("**Azerbaijani Legal Document Q&A System**")
st.markdown("Ask questions about Azerbaijani laws in Azerbaijani or English.")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This system uses RAG (Retrieval-Augmented Generation) to answer 
    questions about Azerbaijani laws.
    
    **Data Source**: [e-qanun.az](https://e-qanun.az)
    
    **Models**:
    - Embedding: BAAI/bge-m3
    - LLM: Mistral-7B-Instruct
    
    **Dataset**: allmalab/eqanun (50,989 documents)
    """)
    
    st.header("Example Questions")
    examples = [
        "Emek muqavilesi nedir?",
        "Mehkeme qerari nece shekillendirilir?",
        "Nikah muqavilesi ucun ne teleb olunur?",
        "Vergiler hansi novlere bolunur?"
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state.query = ex


# Load models
@st.cache_resource
def load_chain():
    """Load vectorstore and create RAG chain."""
    with st.spinner("Loading models... (this may take a few minutes)"):
        vectorstore = load_vectorstore()
        llm = get_llm()
        chain = create_rag_chain(vectorstore, llm)
    return chain, vectorstore


# Search-only mode (without LLM)
@st.cache_resource
def load_search_only():
    """Load only the vectorstore for search."""
    with st.spinner("Loading search index..."):
        vectorstore = load_vectorstore()
    return vectorstore


# Mode selection
mode = st.radio(
    "Mode:",
    ["Search Only (Fast)", "Full RAG (Search + Answer)"],
    horizontal=True
)

# Query input
query = st.text_input(
    "Your question:",
    value=st.session_state.get("query", ""),
    placeholder="Emek muqavilesi nedir?"
)

# Search button
if st.button("Search", type="primary"):
    if not query:
        st.warning("Please enter a question.")
    else:
        if mode == "Search Only (Fast)":
            # Search-only mode
            vectorstore = load_search_only()
            results = semantic_search(vectorstore, query, k=5)
            
            st.markdown("### Relevant Documents")
            for i, doc in enumerate(results, 1):
                with st.expander(f"[{i}] Document {doc.metadata.get('id', 'Unknown')}"):
                    st.markdown(f"**Source**: [{doc.metadata['source']}]({doc.metadata['source']})")
                    st.markdown(doc.page_content)
        else:
            # Full RAG mode
            chain, vectorstore = load_chain()
            
            with st.spinner("Generating answer..."):
                result = chain({"query": query})
            
            # Display answer
            st.markdown("### Answer")
            st.markdown(f'<div class="answer-box">{result["result"]}</div>', unsafe_allow_html=True)
            
            # Display sources
            st.markdown("### Sources")
            for i, doc in enumerate(result["source_documents"], 1):
                with st.expander(f"[{i}] Document {doc.metadata.get('id', 'Unknown')}"):
                    st.markdown(f"**Source**: [{doc.metadata['source']}]({doc.metadata['source']})")
                    st.markdown(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)

# Footer
st.markdown("---")
st.markdown("""
**Citation**: If you use this system, please cite:
Isbarov et al., "Open foundation models for Azerbaijani language", SIGTURK 2024.
[Paper](https://aclanthology.org/2024.sigturk-1.2)
""")
