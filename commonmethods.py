import smtplib
import email
import constraints


def sendMail(body):
    user =  constraints.emailFrom
    password =  constraints.emialPassword
    smtpsrv = "smtp.office365.com"
    smtpserver = smtplib.SMTP(smtpsrv,587)
    message = email.message.EmailMessage()
    message.add_header("From",constraints.emailFrom )
    message.add_header("To", 'santhoshkrishna.sakalabattula@quadrantresource.com')
    message.add_header("Subject", 'Qfocast ML Model')
    body =  body
    message.set_content(body)

    smtpserver.ehlo()
    smtpserver.starttls()
    smtpserver.login(user, password)
    smtpserver.send_message (message)   
    smtpserver.close()

# sendMail()
