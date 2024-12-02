import streamlit as st
import pandas as pd
import sqlite3
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from textblob import TextBlob
from nltk.tokenize import sent_tokenize
import nltk
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from report_generator import generate_report, display_report_button
from email_generator import generate_email_content, read_employee_emails
from email_sender import send_email

# Ensure NLTK 'punkt' tokenizer is available
nltk.download('punkt')

# Load data from SQLite database
@st.cache_data
def load_data():
    try:
        # Connect to the SQLite database
        db_file = 'feedback_database.db'
        table_name = 'feedback_table'
        connection = sqlite3.connect(db_file)
        
        # Query the table and load it into a DataFrame
        query = f"SELECT * FROM {table_name};"
        df = pd.read_sql_query(query, connection)
        
        # Close the connection
        connection.close()
        
        return df
    except Exception as e:
        st.error(f"Error loading data from database: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error

# Filter the data safely
def filter_data(df, ay, sem):
    try:
        return df[(df['AY'] == ay) & (df['SEM'] == sem)].dropna(subset=['Employee Code', 'Course Code', 'Comment'])  # Drop rows with NaN values
    except KeyError as e:
        st.error(f"Missing column in data: {e}")
        return pd.DataFrame()

# Get employee codes and courses with NaN handling
def get_employee_codes(df):
    try:
        return sorted(df['Employee Code'].dropna().unique())  # Drop NaN values in Employee Code
    except KeyError as e:
        st.error(f"Employee Code column missing: {e}")
        return []

def get_employee_courses(df, employee_code):
    try:
        return sorted(df[df['Employee Code'] == employee_code]['Course Code'].dropna().unique())  # Drop NaN values in Course Code
    except KeyError as e:
        st.error(f"Course Code column missing: {e}")
        return []

# Sentiment Analysis with error handling
def get_sentiment_textblob(comment):
    try:
        sentences = sent_tokenize(comment)
        polarity_sum = 0
        for sentence in sentences:
            analysis = TextBlob(sentence)
            polarity_sum += analysis.sentiment.polarity
        return polarity_sum / len(sentences) if sentences else 0
    except Exception as e:
        st.warning(f"Error in sentiment analysis: {e}")
        return 0

def classify_sentiment_textblob(polarity):
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Add this function to calculate faculty performance score
def calculate_faculty_performance_score(sentiments, comments):
    """
    Calculate a comprehensive performance score based on sentiments and comments.
    
    Args:
        sentiments (dict): Dictionary with sentiment counts
        comments (pd.DataFrame): DataFrame containing comments
    
    Returns:
        dict: Performance score details
    """
    try:
        # Base sentiment scoring
        total_comments = sum(sentiments.values())
        if total_comments == 0:
            return {
                'overall_score': 0,
                'sentiment_score': 0,
                'comment_depth_score': 0,
                'detailed_breakdown': {}
            }
        
        # Sentiment weights
        sentiment_weights = {
            'Positive': 1.0,
            'Neutral': 0.5,
            'Negative': 0.0
        }
        
        # Calculate sentiment-based score
        sentiment_score = sum(
            (sentiments[sentiment] * sentiment_weights[sentiment]) / total_comments * 100 
            for sentiment in ['Positive', 'Neutral', 'Negative']
        )
        
        # Comment depth and quality scoring
        comment_lengths = comments['Comment'].str.len()
        avg_comment_length = comment_lengths.mean()
        comment_depth_score = min(max(avg_comment_length / 100, 0), 1) * 100  # Normalize to 0-100
        
        # Combine scores with weights
        overall_score = (
            sentiment_score * 0.7 +  # Sentiment carries more weight
            comment_depth_score * 0.3
        )
        
        return {
            'overall_score': round(overall_score, 2),
            'sentiment_score': round(sentiment_score, 2),
            'comment_depth_score': round(comment_depth_score, 2),
            'detailed_breakdown': {
                'total_comments': total_comments,
                'sentiment_distribution': sentiments
            }
        }
    except Exception as e:
        st.error(f"Error in calculating performance score: {e}")
        return {
            'overall_score': 0,
            'sentiment_score': 0,
            'comment_depth_score': 0,
            'detailed_breakdown': {}
        }

# Add this function to track faculty performance over time
def get_faculty_performance_history(df, employee_code):
    """
    Retrieve performance history for a faculty member.
    
    Args:
        df (pd.DataFrame): Full feedback dataframe
        employee_code (str): Employee code to track
    
    Returns:
        pd.DataFrame: Performance history dataframe
    """
    try:
        # Group performance by Academic Year and Semester
        performance_history = []
        
        # Get unique combinations of AY and SEM
        ay_sem_combinations = df[df['Employee Code'] == employee_code][['AY', 'SEM']].drop_duplicates()
        
        for _, row in ay_sem_combinations.iterrows():
            ay, sem = row['AY'], row['SEM']
            
            # Filter data for specific AY and SEM
            course_comments = df[
                (df['Employee Code'] == employee_code) & 
                (df['AY'] == ay) & 
                (df['SEM'] == sem)
            ]
            
            # Calculate sentiments
            sentiments = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
            for comment in course_comments['Comment']:
                polarity = get_sentiment_textblob(comment)
                sentiment = classify_sentiment_textblob(polarity)
                sentiments[sentiment] += 1
            
            # Calculate performance score
            performance = calculate_faculty_performance_score(sentiments, course_comments)
            
            performance_history.append({
                'AY': ay,
                'SEM': sem,
                'Overall Score': performance['overall_score'],
                'Sentiment Score': performance['sentiment_score'],
                'Comment Depth Score': performance['comment_depth_score']
            })
        
        return pd.DataFrame(performance_history).sort_values(['AY', 'SEM'])
    except Exception as e:
        st.error(f"Error in retrieving performance history: {e}")
        return pd.DataFrame()

# Create a summary with error handling
def create_personalized_summary(ay, sem, course_name, employee_code, feedback):
    try:
        load_dotenv()
        groq_api_key = os.getenv('SECRET_KEY')
        model = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)
        parser = StrOutputParser()

        prompt = ChatPromptTemplate.from_template(
            """Generate a concise summary for the teacher {employee_code} teaching {course_name} in Academic Year {ay}, Semester {sem}, based on the following feedback:
            {feedback}"""
        )

        chain = prompt | model | parser
        summary = chain.invoke({
            "ay": ay,
            "sem": sem,
            "course_name": course_name,
            "employee_code": employee_code,
            "feedback": feedback
        })
        return summary
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return "Could not generate summary due to an error."

# HOD page
def hod_page():
    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user_type = None
        st.session_state.username = None
        st.experimental_rerun()

    st.title('Faculty Feedback Analysis (HOD View)')

    # Load data
    df = load_data()

    # Check if the data is valid
    if df.empty:
        st.warning("No data available to display.")
        st.stop()

    # Sidebar filters
    ay_options = sorted(df['AY'].dropna().unique())  # Drop NaN values in Academic Year
    sem_options = sorted(df['SEM'].dropna().unique())  # Drop NaN values in Semester

    selected_ay = st.sidebar.selectbox('Select Academic Year', ay_options)
    selected_sem = st.sidebar.selectbox('Select Semester', sem_options)

    filtered_df = filter_data(df, selected_ay, selected_sem)

    # Handle empty filtered data
    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
        st.stop()

    # Get employee codes and courses safely
    employee_codes = get_employee_codes(filtered_df)
    if not employee_codes:
        st.warning("No employee data available for the selected filters.")
        st.stop()

    selected_employee = st.sidebar.selectbox('Select Employee Code', employee_codes)

    employee_courses = get_employee_courses(filtered_df, selected_employee)
    if not employee_courses:
        st.warning(f"No courses available for Employee {selected_employee}.")
        st.stop()

    selected_course = st.sidebar.selectbox('Select Course Code', ['All'] + employee_courses)

    # Display feedback summary
    st.write(f"Showing feedback for: **{selected_employee}** in **{selected_ay}**, **{selected_sem}**")

    # Handle filtering by selected course
    if selected_course != 'All':
        course_comments = filtered_df[(filtered_df['Employee Code'] == selected_employee) & 
                                      (filtered_df['Course Code'] == selected_course)]
    else:
        course_comments = filtered_df[filtered_df['Employee Code'] == selected_employee]

    # Collect comments
    all_comments = " ".join(course_comments['Comment'].dropna().tolist())

    # Generate and display summary
    summary = create_personalized_summary(selected_ay, selected_sem, selected_course, selected_employee, all_comments)
    st.subheader("Course Feedback Summary")
    st.write(summary)

        # Performance scoring section
    st.subheader("Performance Scoring")

    # Calculate sentiments first
    sentiments = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
    for _, row in course_comments.iterrows():
        comment = row['Comment']
        polarity = get_sentiment_textblob(comment)
        sentiment = classify_sentiment_textblob(polarity)
        sentiments[sentiment] += 1

    # Calculate performance score using the now-defined sentiments
    performance = calculate_faculty_performance_score(sentiments, course_comments)
    
    # Display performance metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Score", f"{performance['overall_score']:.2f}%")
    with col2:
        st.metric("Sentiment Score", f"{performance['sentiment_score']:.2f}%")
    with col3:
        st.metric("Comment Depth Score", f"{performance['comment_depth_score']:.2f}%")
    
    # Performance history visualization
    st.subheader("Performance Trend")
    performance_history = get_faculty_performance_history(df, selected_employee)
    
    if not performance_history.empty:
        # Create line chart using Plotly for interactive visualization
        fig = go.Figure()
        
        # Add traces for different scores
        fig.add_trace(go.Scatter(
            x=[f"{row['AY']} Sem {row['SEM']}" for _, row in performance_history.iterrows()],
            y=performance_history['Overall Score'],
            mode='lines+markers',
            name='Overall Score'
        ))
        
        fig.add_trace(go.Scatter(
            x=[f"{row['AY']} Sem {row['SEM']}" for _, row in performance_history.iterrows()],
            y=performance_history['Sentiment Score'],
            mode='lines+markers',
            name='Sentiment Score'
        ))
        
        fig.add_trace(go.Scatter(
            x=[f"{row['AY']} Sem {row['SEM']}" for _, row in performance_history.iterrows()],
            y=performance_history['Comment Depth Score'],
            mode='lines+markers',
            name='Comment Depth Score'
        ))
        
        fig.update_layout(
            title='Faculty Performance Trend',
            xaxis_title='Academic Year and Semester',
            yaxis_title='Score (%)',
            height=400
        )
        
        st.plotly_chart(fig)
    else:
        st.warning("No performance history available for this faculty member.")


    # Sentiment analysis and pie chart
    st.subheader("Sentiment Distribution")

    sentiments = {'Positive': 0, 'Negative': 0, 'Neutral': 0}

    for _, row in course_comments.iterrows():
        comment = row['Comment']
        polarity = get_sentiment_textblob(comment)
        sentiment = classify_sentiment_textblob(polarity)
        sentiments[sentiment] += 1

    # Display sentiment counts
    st.write("**Sentiment Counts:**")
    for sentiment, count in sentiments.items():
        st.write(f"{sentiment}: {count}")

    # Display pie chart
    labels = list(sentiments.keys())
    sizes = list(sentiments.values())
    colors = ['#A0D1E6', '#FF8C8C', '#B2FFB2']  # Light shades for pie chart

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    # Add email functionality
    if st.button("Send Feedback Email"):
        # Load employee emails from the uploaded Excel file
        employee_emails = read_employee_emails("Employee_emails.xlsx")
        
        # Generate email content using Groq LLM
        email_content = generate_email_content(selected_employee, selected_course, summary)
        
        # Generate the report (as a PDF)
        report_bytes = generate_report(summary, sentiments, fig, selected_employee, selected_course, selected_ay, selected_sem)
        
        # Send the email
        sender_email = "shadan.anwar2005@gmail.com"  
        receiver_email = employee_emails.get(selected_employee)
        
        if receiver_email:
            email_sent = send_email(sender_email, receiver_email, email_content, report_bytes)
            if email_sent:
                st.success(f"Email successfully sent to {receiver_email}")
            else:
                st.error(f"Failed to send email to {receiver_email}")
        else:
            st.error(f"Email for {selected_employee} not found.")

    # Additional HOD-specific features
    st.subheader("Department Overview")
    
    # Calculate overall department sentiment
    department_sentiments = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
    for _, row in filtered_df.iterrows():
        comment = row['Comment']
        polarity = get_sentiment_textblob(comment)
        sentiment = classify_sentiment_textblob(polarity)
        department_sentiments[sentiment] += 1

    # Display department sentiment distribution
    st.write("Department Sentiment Distribution:")
    dept_labels = list(department_sentiments.keys())
    dept_sizes = list(department_sentiments.values())
    
    fig_dept, ax_dept = plt.subplots()
    ax_dept.pie(dept_sizes, labels=dept_labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax_dept.axis('equal')
    st.pyplot(fig_dept)

    # Display top performing courses based on positive sentiment
    st.subheader("Top Performing Courses")
    course_performance = filtered_df.groupby('Course Code').apply(lambda x: sum(x['Comment'].apply(get_sentiment_textblob) > 0) / len(x))
    top_courses = course_performance.nlargest(5)
    st.bar_chart(top_courses)

    # Option to view all faculty members' summaries
    if st.button("View All Faculty Summaries"):
        for emp_code in employee_codes:
            emp_comments = filtered_df[filtered_df['Employee Code'] == emp_code]
            emp_feedback = " ".join(emp_comments['Comment'].dropna().tolist())
            emp_summary = create_personalized_summary(selected_ay, selected_sem, "All Courses", emp_code, emp_feedback)
            st.subheader(f"Summary for {emp_code}")
            st.write(emp_summary)
            st.write("---")
