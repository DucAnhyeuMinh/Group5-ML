# app.py
import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go
import time

# --- Streamlit Config (PHẢI ĐẶT ĐẦU TIÊN) ---
st.set_page_config(
    page_title="AI Sentiment Analyzer", 
    page_icon="🎭", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load background image ---
@st.cache_data
def get_base64_image(image_path):
    """Convert image to base64 string"""
    import base64
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return None

# Load background image
bg_image = get_base64_image(r"c:\Users\this pc\Documents\ML\app\𝐀𝐞𝐬𝐭𝐡𝐞𝐭𝐢𝐜 𝐰𝐚𝐥𝐥𝐩𝐚𝐩𝐞𝐫.jpg")

# --- Custom CSS để làm đẹp giao diện ---
# Dynamic CSS based on background image
if bg_image:
    css_background = f"""
    /* Background với ảnh */
    .stApp {{
        background: url(data:image/jpeg;base64,{bg_image});
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }}
    """
else:
    css_background = """
    /* Fallback gradient background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #4facfe 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    """

st.markdown(f"""
<style>
    {css_background}
    
    /* Main container KHÔNG mờ */
    .main .block-container {{
        background: transparent;
        border-radius: 20px;
        padding: 2rem;
        border: none;
        box-shadow: none;
    }}
    
    /* Sidebar KHÔNG mờ */
    .css-1d391kg {{
        background: rgba(0, 0, 0, 0.3);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }}
    
    .main-header {{
        background: rgba(0, 0, 0, 0.4);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.4);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }}
    
    .feature-card {{
        background: rgba(0, 0, 0, 0.4);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.4);
        margin: 1rem 0;
        color: white;
        font-weight: 600;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }}
    
    .result-positive {{
        background: rgba(86, 171, 47, 0.8);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        border: 1px solid rgba(255, 255, 255, 0.4);
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }}
    
    .result-negative {{
        background: rgba(255, 65, 108, 0.8);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        border: 1px solid rgba(255, 255, 255, 0.4);
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }}
    
    .stButton > button {{
        background: rgba(102, 126, 234, 0.8);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.4);
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
        background: rgba(102, 126, 234, 1);
    }}
    
    /* Text areas và inputs */
    .stTextArea > div > div > textarea {{
        background: rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.4);
        border-radius: 10px;
        color: white;
        font-weight: 500;
    }}
    
    .stTextArea > div > div > textarea::placeholder {{
        color: rgba(255, 255, 255, 0.8);
        font-weight: 400;
    }}
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {{
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
        font-weight: bold;
    }}
    
    /* Regular text */
    .stMarkdown, .stText {{
        color: white;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
    }}
    
    /* Sidebar text */
    .css-1d391kg .stMarkdown {{
        color: white;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
    }}
    
    /* Success/Error messages */
    .stAlert {{
        background: rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.4);
        border-radius: 10px;
        color: white;
        font-weight: 500;
    }}
    
    /* Spinner */
    .stSpinner > div {{
        border-top-color: white !important;
    }}
</style>
""", unsafe_allow_html=True)

# --- Load trained Logistic Regression model và freqs ---
@st.cache_resource
def load_model_and_freqs():
    try:
        with open("logreg_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("freqs.pkl", "rb") as f:
            freqs = pickle.load(f)
        return model, freqs, True
    except FileNotFoundError:
        return None, None, False

model, freqs, model_loaded = load_model_and_freqs()

def extract_features(tweet, freqs):
    """
    Trả về vector 3 feature:
    0: bias term
    1: tổng tần suất từ positive
    2: tổng tần suất từ negative
    """
    x = np.zeros(3)
    x[0] = 1  # bias term
    words = tweet.lower().split()
    x[1] = sum([freqs.get((word, 1), 0) for word in words])
    x[2] = sum([freqs.get((word, 0), 0) for word in words])
    return x

def create_confidence_chart(prob, sentiment):
    """Tạo biểu đồ độ tin cậy"""
    colors = ['#56ab2f', '#ff416c'] if sentiment == 'Positive' else ['#ff416c', '#56ab2f']
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob * 100,
        title = {'text': f"Confidence Score<br><span style='font-size:0.8em;color:gray'>Sentiment: {sentiment}</span>"},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': colors[0]},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': colors[0]}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300, showlegend=False)
    return fig

# --- Streamlit UI ---

# Header section
st.markdown("""
<div class="main-header">
    <h1>🎭 AI Sentiment Analysis</h1>
    <p>Analyze text sentiment with Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Sidebar với thông tin
with st.sidebar:
    st.header("ℹ️ Information")
    st.markdown("""
    ### 🔧 Model
    - **Algorithm**: Logistic Regression
    - **Features**: 3 features (bias, pos_freq, neg_freq)
    - **Framework**: Scikit-learn
    
    ### 📊 How it works
    1. Extract features from text
    2. Calculate positive/negative word frequencies
    3. Predict sentiment + confidence score
    """)
    
    if model_loaded:
        st.success("✅ Model loaded successfully!")
    else:
        st.error("❌ Model files not found!")

# Main content
if not model_loaded:
    st.error("**Error**: Cannot load model. Please ensure `logreg_model.pkl` and `freqs.pkl` files exist!")
    st.stop()

# Input section
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Enter text for analysis")
    user_input = st.text_area(
        "Enter content...", 
        height=150,
        placeholder="Example: I love this movie! It's amazing and wonderful!",
        help="Enter any English text to analyze sentiment"
    )
    
    # Sample texts
    st.markdown("**📋 Sample texts:**")
    sample_col1, sample_col2, sample_col3 = st.columns(3)
    
    with sample_col1:
        if st.button("😊 Positive Sample"):
            st.rerun()
            user_input = "I absolutely love this product! It's amazing and works perfectly!"
    
    with sample_col2:
        if st.button("😞 Negative Sample"):
            st.rerun()
            user_input = "This is terrible! I hate it and it doesn't work at all!"
    
    with sample_col3:
        if st.button("😐 Neutral Sample"):
            st.rerun()
            user_input = "This product is okay. It works fine but nothing special."

with col2:
    st.subheader("🎯 Analysis Results")
    
    # Analyze button
    analyze_button = st.button("🔍 ANALYZE NOW", use_container_width=True)
    
    if analyze_button:
        if not user_input.strip():
            st.warning("⚠️ Please enter some text!")
        else:
            # Show loading
            with st.spinner("Analyzing..."):
                time.sleep(0.5)  # Tạo hiệu ứng loading
                
                # Extract features và predict
                input_vec = extract_features(user_input, freqs).reshape(1, -1)
                y_pred = model.predict(input_vec)[0]
                prob = model.predict_proba(input_vec).max()
                
                sentiment = "Positive" if y_pred == 1 else "Negative"
                
                # Hiển thị kết quả
                if sentiment == "Positive":
                    st.markdown(f"""
                    <div class="result-positive">
                        😊 POSITIVE<br>
                        Confidence: {prob:.1%}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-negative">
                        😞 NEGATIVE<br>
                        Confidence: {prob:.1%}
                    </div>
                    """, unsafe_allow_html=True)

# Results section (nếu có phân tích)
if analyze_button and user_input.strip():
    st.markdown("---")
    
    # Detailed results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Detailed Analysis")
        
        # Feature breakdown
        features = extract_features(user_input, freqs)
        
        st.markdown(f"""
        <div class="feature-card">
            <strong>🔢 Extracted Features:</strong><br>
            • Bias term: {features[0]}<br>
            • Positive words count: {features[1]}<br>
            • Negative words count: {features[2]}
        </div>
        """, unsafe_allow_html=True)
        
        # Prediction details
        proba_scores = model.predict_proba(input_vec)[0]
        st.markdown(f"""
        <div class="feature-card">
            <strong>🎯 Prediction Probabilities:</strong><br>
            • Negative: {proba_scores[0]:.1%}<br>
            • Positive: {proba_scores[1]:.1%}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("📈 Confidence Chart")
        # Confidence chart
        fig = create_confidence_chart(prob, sentiment)
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🤖 Powered by <strong>Logistic Regression</strong> | Built with ❤️ using <strong>Streamlit</strong></p>
    <p><em>Automated text sentiment analysis with high accuracy</em></p>
</div>
""", unsafe_allow_html=True)
