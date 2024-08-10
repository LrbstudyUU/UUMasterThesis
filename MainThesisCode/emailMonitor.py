#Simple email bot to send emails in case of errors
import smtplib
from email.message import EmailMessage

class EmailBot:
    def __init__(self):
        self.msg = EmailMessage()
        self.msg['Subject'] = 'Test'
        self.msg['From'] = '' #removed for privacy
        self.msg['To'] = '' #removed for privacy
        pass

    def setEmailContent(self, content):
        self.msg.set_content(content)

    def sendEmail(self):
        with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
            smtp.login('', '') #for gmail, email and pass
            smtp.send_message(self.msg)