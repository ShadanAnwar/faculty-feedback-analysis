import smtplib
from email.mime.multipart import MIMEMultipart  # Import MIMEMultipart
from email.mime.text import MIMEText  # Import MIMEText for email body
from email.mime.application import MIMEApplication  # Import MIMEApplication for attachments

def send_email(sender_email, receiver_email, email_content, report_bytes, subject="Course Feedback Report"):
    msg = MIMEMultipart()  # Create a MIMEMultipart email object
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    # Attach email body
    msg.attach(MIMEText(email_content, 'plain'))

    # Attach PDF report
    report_attachment = MIMEApplication(report_bytes.getvalue(), _subtype="pdf")
    report_attachment.add_header('Content-Disposition', 'attachment', filename="feedback_report.pdf")
    msg.attach(report_attachment)

    # SMTP server setup for Gmail
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)  # Gmail SMTP server
        server.starttls()
        server.login('shadan.anwar2005@gmail.com', "nfvb fuzp cuic kvwx")  # Your email and app password
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()
        return True
    except Exception as e:
        print(f"Error sending email: {e}")  # Print error details
        return False
