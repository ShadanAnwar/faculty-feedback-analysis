from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd

# Function to generate professional email
def generate_email_content(employee_name, course_name, feedback_summary):
    # GROQ API setup
    groq_api_key = "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"
    model = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)
    parser = StrOutputParser()

    prompt = ChatPromptTemplate.from_template(
        """Write a professional email to {employee_name} about the feedback for the course {course_name} in a positive tone. 
        Include the following summary of the feedback: {feedback_summary}. 
        The email should be concise, appreciative, and motivating."""
    )

    chain = prompt | model | parser

    email_content = chain.invoke({
        "employee_name": employee_name,
        "course_name": course_name,
        "feedback_summary": feedback_summary
    })

    return email_content

# Function to read employee emails from Excel
def read_employee_emails(file_path):
    df = pd.read_excel(file_path)
    return dict(zip(df['Employee Name'], df['Email']))  # Assuming the file has 'Employee Name' and 'Email' columns
