import matplotlib.pyplot as plt
from fpdf import FPDF
import os
import streamlit as st
from io import BytesIO

# Report Generation Function
def generate_report(summary, sentiments, fig, employee_name, course_name, ay, sem):
    pdf = FPDF()

    # Add a page
    pdf.add_page()

    # Title and Header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt=f"Feedback Report for {employee_name}", ln=True, align="C")
    pdf.ln(10)

    # Academic Information
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Academic Year: {ay}, Semester: {sem}", ln=True, align="L")
    pdf.cell(200, 10, txt=f"Course: {course_name}", ln=True, align="L")
    pdf.ln(10)

    # Feedback Summary
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Feedback Summary", ln=True, align="L")
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(200, 10, summary)
    pdf.ln(10)

    # Sentiment Counts
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Sentiment Analysis", ln=True, align="L")
    for sentiment, count in sentiments.items():
        pdf.cell(200, 10, txt=f"{sentiment}: {count}", ln=True, align="L")

    # Save the pie chart as an image
    image_path = "pie_chart.png"
    fig.savefig(image_path, format='png')
    plt.close(fig)  # Close the figure to free memory

    # Add the Pie Chart (Visualization)
    pdf.ln(10)
    pdf.image(image_path, x=10, y=None, w=100)

    # Clean up the saved image
    os.remove(image_path)

    # Save the report to a temporary file
    temp_file_path = "feedback_report.pdf"
    pdf.output(temp_file_path)  # Save the PDF to a file

    # Read the PDF into a BytesIO object
    with open(temp_file_path, 'rb') as f:
        report_bytes = BytesIO(f.read())

    # Clean up the temporary file
    os.remove(temp_file_path)

    return report_bytes

def display_report_button(summary, sentiments, fig, employee_name, course_name, ay, sem):
    if st.button("Generate and Download Report"):
        report_bytes = generate_report(summary, sentiments, fig, employee_name, course_name, ay, sem)
        st.download_button("Download Report", data=report_bytes.getvalue(), file_name="feedback_report.pdf")

