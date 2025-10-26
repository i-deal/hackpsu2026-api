import os
from twilio.rest import Client

TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_MESSAGING_SERVICE_SID = os.getenv('TWILIO_MESSAGING_SERVICE_SID')

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

def send_sms_notification(to_number: str, crime_details: str) -> str:
    """
    Send an SMS notification using Twilio.
    Args:
        to_number (str): The recipient's phone number (E.164 format).
        crime_details (str): The details of the crime to include in the message.
    Returns:
        str: The SID of the sent message.
    """
    message = client.messages.create(
        messaging_service_sid=TWILIO_MESSAGING_SERVICE_SID,
        body=f'A crime has been reported: {crime_details}. Please refer to your local police or news station for more information.',
        to=to_number
    )
    return message.sid
