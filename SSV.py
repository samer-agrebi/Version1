import json
import os 
import sys
import boto3
import langchain_community
from langchain_aws import BedrockEmbeddings
from langchain_community.llms import Bedrock
import numpy as np 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_models import BedrockChat
from langchain.chains import RetrievalQA
import uuid
import streamlit as st
from datetime import datetime
import time

OPENAI_API_KEY="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"
# Bedrock Clients 
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock) 

def data_ingestion():
    """Load and process PDF documents from data directory"""
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    docs = text_splitter.split_documents(documents)
    return docs

def get_vector_store(docs):
    """Create and save FAISS vector store from documents"""
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

def get_claude_3_Sonnet_llm():
    """Initialize Claude 3 Sonnet model"""
    llm = BedrockChat(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        client=bedrock,
        model_kwargs={'max_tokens': 512}
    )
    return llm

def get_Llama_3_70B_Instruct_llm():
    """Initialize LLaMA 3 70B model"""
    llm = Bedrock(
        model_id="meta.llama3-70b-instruct-v1:0",
        client=bedrock,
        model_kwargs={'max_gen_len': 512}
    )
    return llm

# Enhanced prompt template for vacuum technology sales assistant
prompt_template = """
Human: Act as a knowledgeable sales assistant for a vacuum technology company specializing in lifting solutions. Your task is to analyze customer requirements, match them with the company's product offerings, and recommend the most suitable solution. Follow these steps:
Understand the Customer's Needs:
Ask clarifying questions about:
- Industry (e.g., automotive, manufacturing, logistics)
- Weight & dimensions of the load (kg, size, shape)
- Required lifting height & movement (vertical/horizontal)
- Workspace constraints (overhead rails, floor space, etc.)
- Special requirements (ergonomics, automation, hygiene standards)
Compare with Technical Data:
Review the company's product range:
- Standard vacuum lifting systems (for general-purpose handling)
- Industrial manipulators (for precision and flexibility)
- Chain hoists & overhead rail systems (for heavy loads/long-distance transport)
- Custom end effectors (8,000+ variants for unique applications)
Recommend the Best Solution:
- Suggest the most efficient and ergonomic option based on the customer's inputs
- Highlight key benefits (e.g., safety, productivity gains, customization)
- If needed, propose a custom-designed end effector
Be professional, technical yet approachable, and prioritize customer satisfaction.
<context>
{context}
</context>
Question: {question}"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm, vectorstore_faiss, query):
    """Get response from LLM using the retrieval chain"""
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer["result"]

def apply_custom_css():
    """Apply custom CSS styling for a modern, professional look including voice input"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }
    
    .chat-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        min-height: 400px;
        max-height: 600px;
        overflow-y: auto;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 0 1rem auto;
        max-width: 80%;
        width: fit-content;
        margin-left: auto;
        margin-right: 0;
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    }
    
    .assistant-message {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        color: #1a202c;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem auto 1rem 0;
        max-width: 85%;
        width: fit-content;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .message-header {
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        opacity: 0.8;
    }
    
    .message-content {
        line-height: 1.6;
        font-size: 1rem;
    }
    
    .feedback-section {
        margin: 1rem 0;
        padding: 1rem;
        background: #f9fafb;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
        margin: 1rem 0;
    }
    
    .status-ready {
        background: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .status-processing {
        background: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    
    .voice-input-container {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 1rem 0;
        padding: 1rem;
        background: white;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    .voice-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        color: white;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    }
    
    .voice-button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    
    .voice-button.recording {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    
    .voice-status {
        flex: 1;
        padding: 0 1rem;
        font-weight: 500;
        color: #4a5568;
    }
    
    .voice-transcript {
        background: #f7fafc;
        border: 2px dashed #cbd5e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        min-height: 50px;
        color: #4a5568;
        font-style: italic;
    }
    
    @media (max-width: 768px) {
        .main-header h1 { font-size: 2rem; }
        .user-message, .assistant-message { max-width: 95%; }
        .voice-input-container { flex-direction: column; text-align: center; }
    }
    </style>
    """, unsafe_allow_html=True)

def render_voice_input_js():
    """Render JavaScript for Web Speech API voice input"""
    return """
    <script>
    window.SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    
    if (window.SpeechRecognition) {
        let recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';
        
        let isRecording = false;
        let voiceButton = null;
        let statusElement = null;
        let transcriptElement = null;
        
        function initVoiceInput() {
            voiceButton = document.getElementById('voice-button');
            statusElement = document.getElementById('voice-status');
            transcriptElement = document.getElementById('voice-transcript');
            
            if (voiceButton) {
                voiceButton.addEventListener('click', toggleRecording);
            }
        }
        
        function toggleRecording() {
            if (isRecording) {
                stopRecording();
            } else {
                startRecording();
            }
        }
        
        function startRecording() {
            isRecording = true;
            voiceButton.classList.add('recording');
            voiceButton.innerHTML = '⏸️';
            statusElement.innerHTML = '🎤 Listening... Speak now!';
            transcriptElement.innerHTML = 'Listening for your voice...';
            
            recognition.start();
        }
        
        function stopRecording() {
            isRecording = false;
            voiceButton.classList.remove('recording');
            voiceButton.innerHTML = '🎤';
            statusElement.innerHTML = '🎙️ Click microphone to speak';
            
            recognition.stop();
        }
        
        recognition.onresult = function(event) {
            const transcript = event.results[0][0].transcript;
            transcriptElement.innerHTML = transcript;
            statusElement.innerHTML = '🔄 Processing voice input...';
            
            // Function to find and update chat input with multiple attempts
            function updateChatInput(attempts = 0) {
                const maxAttempts = 10;
                
                // Enhanced selectors for Streamlit chat input
                const selectors = [
                    'input[placeholder*="Ask me about vacuum technology solutions"]',
                    'input[placeholder*="vacuum technology"]',
                    'div[data-testid="stChatInput"] input',
                    'div[data-testid="chatInput"] input',
                    'input[aria-label*="Ask me about vacuum technology"]',
                    'input[aria-label*="vacuum technology"]',
                    '.stChatInput input',
                    '.stChatInput textarea',
                    'input[type="text"]:not([aria-hidden="true"])',
                    'textarea[placeholder*="Ask"]',
                    'div[class*="stChatInput"] input',
                    'div[class*="stChatInput"] textarea'
                ];
                
                let chatInput = null;
                
                // Try to find input in current document
                for (let selector of selectors) {
                    const elements = document.querySelectorAll(selector);
                    for (let element of elements) {
                        if (element.offsetParent !== null) { // Check if element is visible
                            chatInput = element;
                            break;
                        }
                    }
                    if (chatInput) break;
                }
                
                // Try parent document if not found
                if (!chatInput && parent.document && parent.document !== document) {
                    for (let selector of selectors) {
                        const elements = parent.document.querySelectorAll(selector);
                        for (let element of elements) {
                            if (element.offsetParent !== null) {
                                chatInput = element;
                                break;
                            }
                        }
                        if (chatInput) break;
                    }
                }
                
                if (chatInput) {
                    try {
                        // Clear existing value
                        chatInput.value = '';
                        
                        // Set the new value
                        chatInput.value = transcript;
                        
                        // Create and dispatch input events
                        const inputEvent = new Event('input', { 
                            bubbles: true, 
                            cancelable: true,
                            composed: true
                        });
                        
                        const changeEvent = new Event('change', { 
                            bubbles: true, 
                            cancelable: true 
                        });
                        
                        // Dispatch events
                        chatInput.dispatchEvent(inputEvent);
                        chatInput.dispatchEvent(changeEvent);
                        
                        // Focus and trigger additional events
                        chatInput.focus();
                        chatInput.click();
                        
                        // Additional Streamlit-specific events
                        setTimeout(() => {
                            chatInput.dispatchEvent(new KeyboardEvent('keyup', {
                                bubbles: true,
                                key: 'Enter'
                            }));
                        }, 50);
                        
                        statusElement.innerHTML = '✅ Voice input automatically added to chat! Ready to send.';
                        
                        // Store transcript in session storage as backup
                        try {
                            sessionStorage.setItem('voiceTranscript', transcript);
                        } catch (e) {
                            console.log('Session storage not available');
                        }
                        
                    } catch (error) {
                        console.error('Error setting input value:', error);
                        statusElement.innerHTML = '⚠️ Voice captured! Please copy the text above to the input field.';
                    }
                } else {
                    // Retry if not found and attempts remaining
                    if (attempts < maxAttempts) {
                        setTimeout(() => updateChatInput(attempts + 1), 200);
                    } else {
                        statusElement.innerHTML = '⚠️ Voice captured! Please copy the text above to the input field.';
                        console.log('Chat input field not found after', maxAttempts, 'attempts');
                    }
                }
            }
            
            // Start trying to update the chat input
            updateChatInput();
        };
        
        recognition.onerror = function(event) {
            console.error('Speech recognition error:', event.error);
            statusElement.innerHTML = '❌ Voice input error. Please try again.';
            stopRecording();
        };
        
        recognition.onend = function() {
            stopRecording();
        };
        
        // Initialize when DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', initVoiceInput);
        } else {
            initVoiceInput();
        }
        
    } else {
        console.warn('Speech Recognition API not supported in this browser');
    }
    </script>
    """

def render_voice_input():
    """Render voice input interface with Web Speech API"""
    st.markdown("### 🎤 Voice Input")
    
    # Check for browser support message
    st.info("💡 **Hands busy? Don’t worry — hit the mic 🎤 and I’ll capture your voice.")
    
    # Voice input HTML interface
    voice_html = f"""
    <div class="voice-input-container">
        <button id="voice-button" class="voice-button" title="Click to start/stop voice input">
            🎤
        </button>
        <div class="voice-status">
            <div id="voice-status">🎙️ Click microphone to speak</div>
        </div>
    </div>
    <div id="voice-transcript" class="voice-transcript">
        Your voice input will appear here...
    </div>
    {render_voice_input_js()}
    """
    
    st.components.v1.html(voice_html, height=200, scrolling=False)
    
    # Fallback for unsupported browsers
    st.markdown("""
    <div style="margin-top: 1rem; padding: 1rem; background: #fff3cd; border-radius: 10px; border-left: 4px solid #ffc107;">
        <strong>📱 Browser Compatibility:</strong><br>
        • ✅ Chrome, Edge, Safari (recommended)<br>
        • ❌ Firefox (limited support)<br>
        • 💡 If voice input doesn't work, please use a supported browser or type your question below.
    </div>
    """, unsafe_allow_html=True)

def render_header():
    """Render the main application header"""
    st.markdown("""
    <div class="main-header">
        <h1>🌀 VacuMind AI Assistant</h1>
        <p>Your Intelligent Partner for Vacuum Technology Solutions</p>
    </div>
    """, unsafe_allow_html=True)

def render_model_selection():
    """Render model selection with enhanced UI"""
    st.markdown("### 🧠 Choose Your AI Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        claude_selected = st.session_state.get("selected_model") == "Claude"
        button_style = "🎭 Claude 3 Sonnet ✓" if claude_selected else "🎭 Claude 3 Sonnet"
        
        if st.button(button_style, key="claude_btn", use_container_width=True, type="primary" if claude_selected else "secondary"):
            st.session_state.selected_model = "Claude"
            st.rerun()
            
        st.markdown("<div style='text-align: center; margin-top: 0.5rem;'><small style='color: #718096;'>Advanced reasoning and analysis</small></div>", unsafe_allow_html=True)
    
    with col2:
        llama_selected = st.session_state.get("selected_model") == "LLaMA"
        button_style = "🦙 LLaMA 3 70B ✓" if llama_selected else "🦙 LLaMA 3 70B"
        
        if st.button(button_style, key="llama_btn", use_container_width=True, type="primary" if llama_selected else "secondary"):
            st.session_state.selected_model = "LLaMA"
            st.rerun()
            
        st.markdown("<div style='text-align: center; margin-top: 0.5rem;'><small style='color: #718096;'>Fast and efficient responses</small></div>", unsafe_allow_html=True)

def render_status_indicator():
    """Render system status indicator"""
    if st.session_state.get("selected_model"):
        model_name = st.session_state.selected_model
        st.markdown(f"""
        <div class="status-indicator status-ready">
            ✅ {model_name} Model Active
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-indicator status-processing">
            ⚠️ Please select an AI model to begin
        </div>
        """, unsafe_allow_html=True)

def render_chat_message(message, is_user=False):
    """Render individual chat message with enhanced styling"""
    if is_user:
        st.markdown(f"""
        <div class="user-message">
            <div class="message-header">You</div>
            <div class="message-content">{message["content"]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        model_name = st.session_state.get("selected_model", "Assistant")
        st.markdown(f"""
        <div class="assistant-message">
            <div class="message-header">{model_name} AI</div>
            <div class="message-content">{message["content"]}</div>
        </div>
        """, unsafe_allow_html=True)

def render_feedback_ui(message_id):
    """Render enhanced feedback UI"""
    if message_id not in st.session_state.get("feedback", {}):
        st.markdown("""
        <div class="feedback-section">
            <div style="font-weight: 500; margin-bottom: 0.5rem;">Was this response helpful?</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 3])
        
        with col1:
            if st.button("👍 Yes", key=f"pos_{message_id}", help="Mark as helpful"):
                if "feedback" not in st.session_state:
                    st.session_state.feedback = {}
                st.session_state.feedback[message_id] = "positive"
                st.rerun()
        
        with col2:
            if st.button("👎 No", key=f"neg_{message_id}", help="Mark as not helpful"):
                if "feedback" not in st.session_state:
                    st.session_state.feedback = {}
                st.session_state.feedback[message_id] = "negative"
                st.rerun()
    else:
        feedback = st.session_state.feedback[message_id]
        if feedback == "positive":
            st.success("✅ Thank you for your feedback!")
        else:
            st.info("📝 Thank you for your feedback. We'll work to improve!")

def initialize_session_state():
    """Initialize all session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None
        
    if "feedback" not in st.session_state:
        st.session_state.feedback = {}
    
    if "vector_store_ready" not in st.session_state:
        st.session_state.vector_store_ready = False

def main():
    # Page configuration
    st.set_page_config(
        page_title="VacuMind AI Assistant",
        page_icon="🌀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom styling
    apply_custom_css()
    
    # Initialize session state
    initialize_session_state()
    
    # Render header
    render_header()
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Model selection
        render_model_selection()
        
        # Status indicator
        render_status_indicator()
        
        # Voice input section
        render_voice_input()
        
        # Chat container
        st.markdown("### 💬 Conversation")
        
        chat_container = st.container()
        with chat_container:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            if not st.session_state.messages:
                st.markdown("""
                <div style="text-align: center; padding: 3rem; color: #718096;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">🚀</div>
                    <h3>Welcome to VacuMind!</h3>
                    <p>I'm your AI assistant for vacuum technology solutions. Ask me about:</p>
                    <ul style="text-align: left; display: inline-block; margin-top: 1rem;">
                        <li>Product recommendations</li>
                        <li>Technical specifications</li>
                        <li>Industry applications</li>
                        <li>Custom solutions</li>
                    </ul>
                    <p style="margin-top: 1rem; font-weight: bold; color: #667eea;">
                        💡 Can't type? Use the microphone button above to speak your question!
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Display chat messages
            for i, message in enumerate(st.session_state.messages):
                if message["role"] == "user":
                    render_chat_message(message, is_user=True)
                else:
                    render_chat_message(message, is_user=False)
                    
                    # Add feedback UI for assistant messages
                    message_id = message.get("message_id", f"msg_{i}")
                    if "message_id" not in message:
                        message["message_id"] = message_id
                    render_feedback_ui(message_id)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Sidebar content
        with st.container():
            st.markdown("### ⚡ System Controls")
            
            # Voice input status
            with st.expander("🎤 Voice Input Info", expanded=True):
                st.markdown("""
                **For customers who can't write:**
                - Click the 🎤 microphone button
                - Speak your question clearly  
                - The text will appear automatically
                - Then submit your question normally
                
                **Supported languages:** English (US)
                **Works best with:** Chrome, Safari, Edge
                """)
            
            # Vector store management
            with st.expander("📚 Knowledge Base"):
                st.markdown("Manage your document knowledge base:")
                
                if st.button("🔄 Update Knowledge Base", use_container_width=True):
                    with st.spinner("Processing documents..."):
                        try:
                            docs = data_ingestion()
                            get_vector_store(docs)
                            st.session_state.vector_store_ready = True
                            st.success("✅ Knowledge base updated!")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                
                # Display status
                if st.session_state.vector_store_ready:
                    st.success("✅ Knowledge base ready")
                else:
                    st.warning("⚠️ Knowledge base not initialized")
            
            # Statistics
            with st.expander("📊 Session Stats"):
                total_messages = len(st.session_state.messages)
                user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
                
                st.metric("Total Messages", total_messages)
                st.metric("Your Questions", user_messages)
                
                if st.session_state.get("selected_model"):
                    st.metric("Active Model", st.session_state.selected_model)
            
            # Quick actions
            with st.expander("🔧 Quick Actions"):
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    st.session_state.messages = []
                    st.session_state.feedback = {}
                    st.rerun()
                
                if st.button("🔄 Reset Session", use_container_width=True):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()
    
    # Chat input
    st.markdown("---")
    user_question = st.chat_input("Ask me about vacuum technology solutions... 🌪️")
    
    if user_question:
        if not st.session_state.get("selected_model"):
            st.error("⚠️ Please select an AI model before asking questions.")
            st.stop()
        
        # Add user message
        user_message_id = f"user_{uuid.uuid4().hex[:8]}"
        st.session_state.messages.append({
            "role": "user",
            "content": user_question,
            "message_id": user_message_id,
            "timestamp": datetime.now().isoformat()
        })
        
        try:
            # Load FAISS index
            faiss_index = FAISS.load_local(
                "faiss_index", 
                bedrock_embeddings, 
                allow_dangerous_deserialization=True
            )
            
            # Get selected model
            if st.session_state.selected_model == "Claude":
                llm = get_claude_3_Sonnet_llm()
            else:
                llm = get_Llama_3_70B_Instruct_llm()
            
            # Get response
            with st.spinner("🔄 Processing your request..."):
                response = get_response_llm(llm, faiss_index, user_question)
            
            # Add assistant message
            assistant_message_id = f"assistant_{uuid.uuid4().hex[:8]}"
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "message_id": assistant_message_id,
                "timestamp": datetime.now().isoformat(),
                "model": st.session_state.selected_model
            })
            
        except Exception as e:
            st.error(f"❌ Error processing your request: {str(e)}")
            st.info("💡 Make sure the knowledge base is initialized and try again.")
        
        st.rerun()

if __name__ == "__main__":
    main()