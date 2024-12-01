import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from textblob import TextBlob
from nltk.tokenize import sent_tokenize
import nltk
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import the existing modules
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

# Faculty page
def faculty_page():
    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user_type = None
        st.session_state.username = None
        st.experimental_rerun()

    st.title('Faculty Feedback Analysis')

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

    # Display report generation button
    display_report_button(summary, sentiments, fig, selected_employee, selected_course, selected_ay, selected_sem)

    # Show individual comments like in summarizer file
    st.subheader("Individual Comments")

    for _, row in course_comments.iterrows():
        st.write(f"- {row['Comment']}")

    st.write("---")  # Add a separator between courses