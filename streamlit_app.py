# Streamlit frontend for the AI Query System 

import sys
import os

# Add src to path
sys.path.append('src')

try:
    import streamlit as st
    streamlit_available = True
except ImportError:
    print("Streamlit not available. Install streamlit to use the web interface.")
    streamlit_available = False

if streamlit_available:
    import json
    from datetime import datetime
    from ai_query_system import AIQuerySystem

    # Page configuration
    st.set_page_config(
        page_title="AI Query System",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    if 'query_system' not in st.session_state:
        st.session_state.query_system = AIQuerySystem()
        st.session_state.initialized = False

    # Initialize the system
    @st.cache_resource
    def initialize_system():
        """Initialize the AI Query System"""
        try:
            st.session_state.query_system.initialize()
            return True, "System initialized successfully!"
        except Exception as e:
            return False, f"Failed to initialize system: {e}"

    # Main app
    def main():
        st.title("🤖 AI Query System")
        st.markdown("**A RAG-based chatbot that answers questions using document knowledge**")

        # Sidebar
        with st.sidebar:
            st.header("System Information")
            
            # Initialize button
            if st.button("Initialize System", type="primary"):
                with st.spinner("Initializing system..."):
                    success, message = initialize_system()
                    if success:
                        st.session_state.initialized = True
                        st.success(message)
                    else:
                        st.error(message)

            # System info
            if st.session_state.initialized:
                try:
                    info = st.session_state.query_system.get_system_info()
                    st.json(info)
                    
                    # Document list
                    st.subheader("Available Documents")
                    docs = st.session_state.query_system.list_documents()
                    for doc in docs:
                        st.text(f"📄 {doc}")
                        
                except Exception as e:
                    st.error(f"Error getting system info: {e}")

        # Main content area
        if not st.session_state.initialized:
            st.warning("Please initialize the system first using the button in the sidebar.")
            return

        # Query interface
        st.subheader("Ask a Question")
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            query = st.text_input(
                "Enter your question:",
                placeholder="e.g., What is machine learning?",
                key="query_input"
            )
        
        with col2:
            top_k = st.selectbox("Top results:", [1, 2, 3, 4, 5], index=2)

        if st.button("Submit Query", type="primary") and query:
            with st.spinner("Processing your query..."):
                try:
                    result = st.session_state.query_system.query(query, top_k=top_k)
                    
                    # Display response
                    st.subheader("Response")
                    st.write(result['response'])
                    
                    # Display metadata
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Sources Used", len(result['sources']))
                    
                    with col2:
                        st.metric("Context Chunks", result['context_chunks'])
                    
                    with col3:
                        st.metric("Model Used", result['model_used'])
                    
                    # Sources
                    if result['sources']:
                        st.subheader("Sources")
                        for i, source in enumerate(result['sources'], 1):
                            st.text(f"{i}. {source}")
                    
                    # Similarity scores
                    if 'similarity_scores' in result and result['similarity_scores']:
                        st.subheader("Similarity Scores")
                        for i, score in enumerate(result['similarity_scores'], 1):
                            st.text(f"Chunk {i}: {score:.3f}")
                    
                    # Feedback section
                    st.subheader("Feedback")
                    feedback_col1, feedback_col2 = st.columns(2)
                    
                    with feedback_col1:
                        helpful = st.radio("Was this helpful?", ["Yes", "No"], key=f"helpful_{query}")
                    
                    with feedback_col2:
                        comments = st.text_area("Comments (optional):", key=f"comments_{query}")
                    
                    if st.button("Submit Feedback"):
                        try:
                            st.session_state.query_system.add_feedback(
                                query, 
                                result['response'], 
                                helpful == "Yes", 
                                comments
                            )
                            st.success("Feedback submitted successfully!")
                        except Exception as e:
                            st.error(f"Error submitting feedback: {e}")
                
                except Exception as e:
                    st.error(f"Error processing query: {e}")

        # Query history
        st.subheader("Recent Queries")
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []

        if query and st.session_state.get('last_query') != query:
            st.session_state.query_history.insert(0, {
                'query': query,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            # Keep only last 5 queries
            st.session_state.query_history = st.session_state.query_history[:5]
            st.session_state.last_query = query

        for i, hist_query in enumerate(st.session_state.query_history):
            if st.button(f"🔄 {hist_query['query'][:50]}...", key=f"hist_{i}"):
                st.session_state.query_input = hist_query['query']
                st.experimental_rerun()

    if __name__ == "__main__":
        main()
        
else:
    # Fallback for when Streamlit is not available
    print("Streamlit Web Interface")
    print("=======================")
    print("Streamlit is not installed. To use the web interface:")
    print("1. Install streamlit: pip install streamlit")
    print("2. Run: streamlit run streamlit_app.py")
    print("\nAlternatively, you can use the command-line interface:")
    print("python3 src/ai_query_system.py")