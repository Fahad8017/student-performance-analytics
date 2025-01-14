import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timezone

# Page Configuration
st.set_page_config(page_title="Student Performance Analytics", page_icon="📚", layout="wide")

# Updated CSS
st.markdown("""
    <style>
    /* Main background animation */
    .stApp {
        background: linear-gradient(-45deg, 
            rgba(142, 68, 173, 0.3),
            rgba(46, 204, 113, 0.3),
            rgba(52, 152, 219, 0.3)
        );
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }

    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Sidebar styling with yellow animation */
    section[data-testid="stSidebar"] {
        background: linear-gradient(165deg, 
            rgba(70, 130, 180, 0.85),  /* Steel blue */
            rgba(255, 215, 0, 0.6),    /* Yellow */
            rgba(65, 105, 225, 0.85)    /* Royal blue */
        ) !important;
        background-size: 400% 400% !important;
        animation: sidebarGradient 15s ease infinite !important;
        border-radius: 0 20px 20px 0;
    }

    @keyframes sidebarGradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    section[data-testid="stSidebar"] > div {
        background: transparent !important;
    }

    /* Remove timestamp and user info */
    .css-1dp5vir, .css-dateTime, .css-user-info {
        display: none !important;
    }

    /* Navigation header styling */
    .nav-header {
        background: rgba(255, 255, 255, 0.2);
        padding: 15px;
        border-radius: 15px;
        margin: 15px;
        backdrop-filter: blur(5px);
    }

    /* Navigation title styling */
    .nav-title {
        color: white !important;
        font-size: 28px !important;
        font-weight: 800 !important;
        text-align: center;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }

    /* Radio buttons styling - removed gradient */
    .stRadio > div {
        background: transparent;
        padding: 12px;
        border-radius: 15px;
        margin: 8px 15px;
        transition: all 0.3s ease;
    }

    .stRadio > div:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: translateX(5px);
    }

    /* Radio button text */
    .st-emotion-cache-1oytle0 {
        color: white !important;
        font-size: 16px !important;
        font-weight: 500 !important;
    }

    /* Expand main content area */
    .css-18e3th9 {
        padding: 1rem !important;
        max-width: 98% !important;
    }

    /* Remove extra boxes and layers */
    .element-container {
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Clean up prediction options */
    .stSelectbox, .stMultiSelect {
        background: transparent !important;
    }

    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1) !important;
        border: none !important;
        border-radius: 10px !important;
    }

    /* Make charts cleaner */
    .js-plotly-plot {
        background: transparent !important;
        border-radius: 15px !important;
    }

    /* Remove default padding and margins */
    .block-container {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }

    /* Hide extra decorative elements */
    .decoration {
        display: none !important;
    }

    /* Custom scrollbar */
    section[data-testid="stSidebar"] ::-webkit-scrollbar {
        width: 6px;
    }

    section[data-testid="stSidebar"] ::-webkit-scrollbar-track {
        background: transparent;
    }

    section[data-testid="stSidebar"] ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.3);
        border-radius: 3px;
    }

    /* Remove extra white spaces */
    .css-1544g2n {
        padding: 0 !important;
        margin: 0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    data = pd.read_csv(r"C:\Users\mr laptop\Desktop\students.csv")
    return data

data = load_data()

# Remove the date/time header
# Add current user's login with custom styling
st.markdown(f"""
    <div style='
        background-color: white;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
        text-align: right;
        font-size: 14px;
        color: #666;'>
        User: {st.session_state.get('user_login', 'Fahad8017')}
    </div>
""", unsafe_allow_html=True)

# Data Preprocessing
def preprocess_data(data):
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])
    return data, label_encoders

data_processed, label_encoders = preprocess_data(data.copy())

# Model Training
features = data_processed.drop(columns=['Term_3'])
target = data_processed['Term_3']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Updated sidebar
with st.sidebar:
    st.markdown("""
        <div class='nav-header'>
            <h2 class='nav-title'>Navigation</h2>
        </div>
    """, unsafe_allow_html=True)
    
    selected = st.radio("",
        options=["🏠 Dashboard", "📊 Analysis", "🔮 Prediction"],
        key="navigation",
        index=0)

# Dashboard Page
if selected == "🏠 Dashboard":
    st.title("🏠 Student Performance Dashboard")
    
    # Summary Cards at the top
    st.markdown("### 📊 Overall Performance Summary")
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    
    total_students = len(data)
    overall_avg = round((data['Term_1'].mean() + data['Term_2'].mean() + data['Term_3'].mean())/3, 1)
    total_passed = (data['Term_3'] >= 10).sum()
    top_students = (data['Term_3'] >= 15).sum()

    with summary_col1:
        st.markdown("""
        <div style='background-color: #2ecc71; padding: 20px; border-radius: 10px; color: white'>
            <h1 style='text-align: center; font-size: 24px;'>{}</h1>
            <p style='text-align: center; font-size: 16px; margin: 0;'>Total Students</p>
        </div>
        """.format(total_students), unsafe_allow_html=True)

    with summary_col2:
        st.markdown("""
        <div style='background-color: #3498db; padding: 20px; border-radius: 10px; color: white'>
            <h1 style='text-align: center; font-size: 24px;'>{}</h1>
            <p style='text-align: center; font-size: 16px; margin: 0;'>Overall Average</p>
        </div>
        """.format(overall_avg), unsafe_allow_html=True)

    with summary_col3:
        st.markdown("""
        <div style='background-color: #e74c3c; padding: 20px; border-radius: 10px; color: white'>
            <h1 style='text-align: center; font-size: 24px;'>{}</h1>
            <p style='text-align: center; font-size: 16px; margin: 0;'>Passed Students</p>
        </div>
        """.format(total_passed), unsafe_allow_html=True)

    with summary_col4:
        st.markdown("""
        <div style='background-color: #f1c40f; padding: 20px; border-radius: 10px; color: white'>
            <h1 style='text-align: center; font-size: 24px;'>{}</h1>
            <p style='text-align: center; font-size: 16px; margin: 0;'>Top Performers</p>
        </div>
        """.format(top_students), unsafe_allow_html=True)

    # Spacing
    st.markdown("<br>", unsafe_allow_html=True)

    # Term-wise Performance Cards with different text colors
    st.markdown("### 📝 Term-wise Performance")
    
    term_col1, term_col2, term_col3 = st.columns(3)
    
    # Term 1 Card with blue text
    with term_col1:
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 2px solid #e0e0e0'>
            <h3 style='text-align: center; color: #3498db; margin-bottom: 15px;'>Term 1</h3>
            <div style='display: grid; grid-template-columns: 1fr; gap: 10px;'>
                <div style='background-color: white; padding: 10px; border-radius: 5px; text-align: center;'>
                    <p style='margin: 0; color: #666;'>Average Grade</p>
                    <h2 style='margin: 5px 0; color: #2c3e50;'>{}</h2>
                </div>
                <div style='background-color: white; padding: 10px; border-radius: 5px; text-align: center;'>
                    <p style='margin: 0; color: #666;'>Pass Rate</p>
                    <h2 style='margin: 5px 0; color: #2c3e50;'>{}%</h2>
                </div>
                <div style='background-color: white; padding: 10px; border-radius: 5px; text-align: center;'>
                    <p style='margin: 0; color: #666;'>Failed Students</p>
                    <h2 style='margin: 5px 0; color: #e74c3c;'>{}</h2>
                </div>
                <div style='background-color: white; padding: 10px; border-radius: 5px; text-align: center;'>
                    <p style='margin: 0; color: #666;'>Top Performers</p>
                    <h2 style='margin: 5px 0; color: #f1c40f;'>{}</h2>
                </div>
            </div>
        </div>
        """.format(
            round(data['Term_1'].mean(), 1),
            round((data['Term_1'] >= 10).mean() * 100, 1),
            (data['Term_1'] < 10).sum(),
            (data['Term_1'] >= 15).sum()
        ), unsafe_allow_html=True)

    # Term 2 Card with green text
    with term_col2:
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 2px solid #e0e0e0'>
            <h3 style='text-align: center; color: #2ecc71; margin-bottom: 15px;'>Term 2</h3>
            <div style='display: grid; grid-template-columns: 1fr; gap: 10px;'>
                <div style='background-color: white; padding: 10px; border-radius: 5px; text-align: center;'>
                    <p style='margin: 0; color: #666;'>Average Grade</p>
                    <h2 style='margin: 5px 0; color: #2c3e50;'>{}</h2>
                </div>
                <div style='background-color: white; padding: 10px; border-radius: 5px; text-align: center;'>
                    <p style='margin: 0; color: #666;'>Pass Rate</p>
                    <h2 style='margin: 5px 0; color: #2c3e50;'>{}%</h2>
                </div>
                <div style='background-color: white; padding: 10px; border-radius: 5px; text-align: center;'>
                    <p style='margin: 0; color: #666;'>Failed Students</p>
                    <h2 style='margin: 5px 0; color: #e74c3c;'>{}</h2>
                </div>
                <div style='background-color: white; padding: 10px; border-radius: 5px; text-align: center;'>
                    <p style='margin: 0; color: #666;'>Top Performers</p>
                    <h2 style='margin: 5px 0; color: #f1c40f;'>{}</h2>
                </div>
            </div>
        </div>
        """.format(
            round(data['Term_2'].mean(), 1),
            round((data['Term_2'] >= 10).mean() * 100, 1),
            (data['Term_2'] < 10).sum(),
            (data['Term_2'] >= 15).sum()
        ), unsafe_allow_html=True)

    # Term 3 Card with purple text
    with term_col3:
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 2px solid #e0e0e0'>
            <h3 style='text-align: center; color: #9b59b6; margin-bottom: 15px;'>Term 3</h3>
            <div style='display: grid; grid-template-columns: 1fr; gap: 10px;'>
                <div style='background-color: white; padding: 10px; border-radius: 5px; text-align: center;'>
                    <p style='margin: 0; color: #666;'>Average Grade</p>
                    <h2 style='margin: 5px 0; color: #2c3e50;'>{}</h2>
                </div>
                <div style='background-color: white; padding: 10px; border-radius: 5px; text-align: center;'>
                    <p style='margin: 0; color: #666;'>Pass Rate</p>
                    <h2 style='margin: 5px 0; color: #2c3e50;'>{}%</h2>
                </div>
                <div style='background-color: white; padding: 10px; border-radius: 5px; text-align: center;'>
                    <p style='margin: 0; color: #666;'>Failed Students</p>
                    <h2 style='margin: 5px 0; color: #e74c3c;'>{}</h2>
                </div>
                <div style='background-color: white; padding: 10px; border-radius: 5px; text-align: center;'>
                    <p style='margin: 0; color: #666;'>Top Performers</p>
                    <h2 style='margin: 5px 0; color: #f1c40f;'>{}</h2>
                </div>
            </div>
        </div>
        """.format(
            round(data['Term_3'].mean(), 1),
            round((data['Term_3'] >= 10).mean() * 100, 1),
            (data['Term_3'] < 10).sum(),
            (data['Term_3'] >= 15).sum()
        ), unsafe_allow_html=True)
# Analysis Page
elif selected == "📊 Analysis":
    st.title("📊 Detailed Analysis")
    
    tab1, tab2 = st.tabs(["📈 Performance Analysis", "👥 Demographics"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Term Performance Comparison
            term_data = pd.melt(data[['Term_1', 'Term_2', 'Term_3']], 
                               var_name='Term', value_name='Grade')
            fig_box = px.box(term_data, x='Term', y='Grade',
                           title='Grade Distribution Across Terms',
                           color='Term',
                           color_discrete_sequence=['#3498db', '#2ecc71', '#e74c3c'])
            st.plotly_chart(fig_box, use_container_width=True)
            
            # Grade Distribution
            fig_grade = px.histogram(data, x='Term_3',
                                   title='Final Term Grade Distribution',
                                   labels={'Term_3': 'Grade'},
                                   color_discrete_sequence=['#2ecc71'])
            fig_grade.update_layout(bargap=0.1)
            st.plotly_chart(fig_grade, use_container_width=True)
        
        with col2:
            # Performance Trend
            avg_grades = {
                'Term': ['Term 1', 'Term 2', 'Term 3'],
                'Average': [data['Term_1'].mean(),
                          data['Term_2'].mean(),
                          data['Term_3'].mean()]
            }
            fig_trend = px.line(avg_grades, x='Term', y='Average',
                               title='Average Grade Trend',
                               markers=True,
                               line_shape='linear')
            fig_trend.update_traces(marker_size=10)
            st.plotly_chart(fig_trend, use_container_width=True)
            
            # Pass Rate Comparison
            pass_rates = {
                'Term': ['Term 1', 'Term 2', 'Term 3'],
                'Pass Rate': [(data['Term_1'] >= 10).mean() * 100,
                             (data['Term_2'] >= 10).mean() * 100,
                             (data['Term_3'] >= 10).mean() * 100]
            }
            fig_pass = px.bar(pass_rates, x='Term', y='Pass Rate',
                            title='Pass Rate by Term (%)',
                            color='Term',
                            color_discrete_sequence=['#3498db', '#2ecc71', '#e74c3c'])
            st.plotly_chart(fig_pass, use_container_width=True)
    
    with tab2:
        demo_col1, demo_col2 = st.columns(2)
        
        with demo_col1:
            # Gender Distribution
            gender_counts = data['gender'].value_counts()
            fig_gender = px.pie(values=gender_counts.values,
                              names=['Male', 'Female'],
                              title='Gender Distribution',
                              color_discrete_sequence=['#3498db', '#e74c3c'])
            fig_gender.update_traces(textposition='inside', 
                                   textinfo='percent+label',
                                   textfont_size=14)
            fig_gender.update_layout(showlegend=True,
                                   legend=dict(orientation="h", 
                                             yanchor="bottom", 
                                             y=1.02,
                                             xanchor="right", 
                                             x=1))
            st.plotly_chart(fig_gender, use_container_width=True)
        
        with demo_col2:
            # Age Distribution with Orange color
            fig_age = px.histogram(data, x='age',
                                 title='Age Distribution',
                                 color_discrete_sequence=['#FFA500'])  # Changed to orange
            fig_age.update_layout(
                bargap=0.2,
                xaxis_title="Age",
                yaxis_title="Count",
                showlegend=False,
                xaxis=dict(dtick=1),
                bargroupgap=0.1
            )
            fig_age.update_traces(marker_line_width=1,
                                marker_line_color="white")
            st.plotly_chart(fig_age, use_container_width=True)
            
# Prediction Page
elif selected == "🔮 Prediction":
    st.title("🔮 Grade Prediction")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns([1,1,1])
        
        with col1:
            st.subheader("Basic Information")
            gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
            age = st.number_input("Age", min_value=15, max_value=22, value=17)
            
            st.subheader("Academic")
            Term_1 = st.number_input("Term 1 Grade", 0, 20, 10)
            Term_2 = st.number_input("Term 2 Grade", 0, 20, 10)
        
        with col2:
            st.subheader("Family Background")
            Medu = st.selectbox("Mother's Education",
                ["No Education", "Primary", "Secondary", "Higher Secondary", "Degree"])
            Fedu = st.selectbox("Father's Education",
                ["No Education", "Primary", "Secondary", "Higher Secondary", "Degree"])
            Pstatus = st.radio("Parents Status", ["Together", "Apart"], horizontal=True)
        
        with col3:
            st.subheader("Additional Factors")
            absences = st.number_input("Number of Absences", 0, 93, 0)
            activities = st.radio("Extra-curricular Activities", ["Yes", "No"], horizontal=True)
            internet = st.radio("Internet Access at Home", ["Yes", "No"], horizontal=True)
        
        # Submit button centered
        submit_col1, submit_col2, submit_col3 = st.columns([1,1,1])
        with submit_col2:
            submitted = st.form_submit_button("Predict Grade", 
                use_container_width=True,
                type="primary")
    
    # Prediction Results
    if submitted:
        # Convert inputs to model format
        user_input = {
            'gender': 1 if gender == "Female" else 0,
            'age': age,
            'Medu': {"No Education": 0, "Primary": 1, "Secondary": 2, 
                     "Higher Secondary": 3, "Degree": 4}[Medu],
            'Fedu': {"No Education": 0, "Primary": 1, "Secondary": 2, 
                     "Higher Secondary": 3, "Degree": 4}[Fedu],
            'Pstatus': 1 if Pstatus == "Together" else 0,
            'Term_1': Term_1,
            'Term_2': Term_2,
            'activities': 1 if activities == "Yes" else 0,
            'internet': 1 if internet == "Yes" else 0,
            'absences': absences
        }
        
        # Create DataFrame and ensure column order matches training data
        input_df = pd.DataFrame([user_input])
        for col in features.columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[features.columns]
        
        try:
            prediction = model.predict(input_df)[0]
            prediction = round(prediction)
            
            # Display result in columns
            result_col1, result_col2, result_col3 = st.columns([1,2,1])
            
            with result_col2:
                st.markdown("""
                    <div style='background-color: #f0f2f6; 
                              padding: 20px; 
                              border-radius: 10px; 
                              text-align: center;'>
                """, unsafe_allow_html=True)
                
                st.metric("Predicted Grade", f"{prediction}/20")
                
                if prediction < 10:
                    st.error("⚠️ Below Passing Grade")
                    st.markdown("""
                        **Recommendations:**
                        * Increase study hours
                        * Seek additional tutoring
                        * Improve attendance
                        * Review previous materials
                        * Consider joining study groups
                    """)
                elif prediction < 15:
                    st.warning("📝 Average Performance")
                    st.markdown("""
                        **Suggestions for Improvement:**
                        * Join study groups
                        * Participate more in class
                        * Regular practice
                        * Focus on weak areas
                        * Set higher goals
                    """)
                else:
                    st.success("🎉 Excellent Performance Predicted!")
                    st.markdown("""
                        **Keep up the good work!**
                        * Maintain study routine
                        * Help classmates
                        * Consider advanced topics
                        * Participate in competitions
                        * Share study techniques
                    """)
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please ensure all inputs are filled correctly.")