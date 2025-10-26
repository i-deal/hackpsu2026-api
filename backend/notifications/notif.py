import os
from twilio_sms import send_sms_notification

if __name__ == "__main__":
    # Example usage
    to_number = os.getenv("TEST_PHONE_NUMBER", "+18777804236")
    crime_details = "[Crime Details]"
    sid = send_sms_notification(to_number, crime_details)
    print(f"Message sent with SID: {sid}")