import time
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from twilio.rest import Client

class LifeSupportSystem:
    def __init__(self):
        # Initial levels and parameters
        self.oxygen_level = 100  # Oxygen level in percentage
        self.water_recycled = 100  # Water available in percentage
        self.temperature = 22  # Room temperature in Celsius
        self.is_emergency = False
        self.data_history = []  # Store historical data for prediction
        self.health_history = []  # Store crew health history
        self.model = None  # Machine Learning model for predictive maintenance
        self.anomaly_detector = IsolationForest(contamination=0.1)  # Anomaly detection model
        self.health_model = RandomForestClassifier()  # Health prediction model
        self.num_crew_members = 5  # Example number of crew members

    def send_email(self, subject, body):
        sender_email = "your_email@gmail.com"  # Replace with your email
        receiver_email = "recipient_email@gmail.com"  # Replace with the recipient's email
        password = "your_password"  # Replace with your email account password

        # Create the email
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        try:
            # Set up the server
            server = smtplib.SMTP('smtp.gmail.com', 587)  # Use your email service's SMTP server
            server.starttls()  # Enable security
            server.login(sender_email, password)  # Login to your email
            server.sendmail(sender_email, receiver_email, msg.as_string())  # Send the email
            print("Email sent successfully!")
        except Exception as e:
            print(f"Failed to send email: {e}")
        finally:
            server.quit()  # Logout and close the connection

    def send_sms(self, body):
        # Twilio credentials
        account_sid = 'your_account_sid'  # Replace with your Twilio Account SID
        auth_token = 'your_auth_token'  # Replace with your Twilio Auth Token
        from_number = 'your_twilio_number'  # Replace with your Twilio phone number
        to_number = 'recipient_phone_number'  # Replace with the recipient's phone number

        client = Client(account_sid, auth_token)

        try:
            message = client.messages.create(
                body=body,
                from_=from_number,
                to=to_number
            )
            print("SMS sent successfully!")
        except Exception as e:
            print(f"Failed to send SMS: {e}")

    def record_data(self):
        # Record current system data
        self.data_history.append([self.oxygen_level, self.water_recycled, self.temperature])
        if len(self.data_history) > 100:  # Keep only the last 100 records
            self.data_history.pop(0)

    def train_predictive_model(self):
        # Train a linear regression model on historical data
        if len(self.data_history) < 10:  # Need enough data to train
            return
        df = pd.DataFrame(self.data_history, columns=["Oxygen", "Water", "Temperature"])
        X = df[["Oxygen", "Water"]]
        y = df["Temperature"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

    def predict_temperature(self):
        # Predict future temperature based on current levels
        if self.model:
            predicted_temp = self.model.predict([[self.oxygen_level, self.water_recycled]])
            return predicted_temp[0]
        return None

    def check_anomalies(self):
        # Check for anomalies in the recorded data
        if len(self.data_history) < 10:
            return
        df = pd.DataFrame(self.data_history, columns=["Oxygen", "Water", "Temperature"])
        predictions = self.anomaly_detector.fit_predict(df)
        anomalies = df[predictions == -1]  # Anomalies are labeled as -1
        if not anomalies.empty:
            print("Anomaly detected in system data!")

    def biometric_monitoring(self):
        # Simulate biometric data for crew members
        crew_health_data = []
        for i in range(self.num_crew_members):
            heart_rate = random.randint(60, 100)  # Heart rate between 60 and 100 bpm
            oxygen_saturation = random.uniform(90, 100)  # Oxygen saturation in percentage
            crew_health_data.append((heart_rate, oxygen_saturation))
            self.health_history.append((heart_rate, oxygen_saturation))
            print(f"Crew Member {i+1} - Heart Rate: {heart_rate} bpm, Oxygen Saturation: {oxygen_saturation:.2f}%")
        return crew_health_data

    def ai_health_assistance(self):
        # Analyze crew health data and provide recommendations
        for i, (heart_rate, oxygen_saturation) in enumerate(self.health_history):
            if heart_rate > 90:  # Elevated heart rate
                print(f"Recommendation for Crew Member {i+1}: Consider relaxation techniques.")
            if oxygen_saturation < 97:  # Low oxygen saturation
                print(f"Recommendation for Crew Member {i+1}: Increase oxygen levels in the habitat.")

            # Predict health status using the trained model
            self.predict_health(heart_rate, oxygen_saturation)

    def predict_health(self, heart_rate, oxygen_saturation):
        # Prepare the data for prediction
        health_data = np.array([[heart_rate, oxygen_saturation]])
        
        if len(self.health_history) > 5:  # Only train if there is enough data
            df_health = pd.DataFrame(self.health_history, columns=["HeartRate", "OxygenSaturation"])
            X = df_health.values
            y = (df_health["HeartRate"] > 90).astype(int)  # Example: 1 if heart rate is high
            
            # Train the health model
            self.health_model.fit(X, y)

            # Predict health condition (1: high risk, 0: normal)
            risk_prediction = self.health_model.predict(health_data)[0]
            if risk_prediction == 1:
                print("Health prediction: High risk for crew member.")
            else:
                print("Health prediction: Normal risk for crew member.")

    def oxygen_production(self):
        # Decrease oxygen gradually to simulate consumption
        self.oxygen_level -= random.uniform(0.5, 2.0)
        if self.oxygen_level < 20:
            self.is_emergency = True
            print(f"EMERGENCY! Oxygen levels critically low: {self.oxygen_level:.2f}%")
            self.send_email("Critical Alert: Oxygen Level", f"Oxygen levels critically low: {self.oxygen_level:.2f}%")
            self.send_sms(f"Critical Alert: Oxygen levels critically low: {self.oxygen_level:.2f}%")
        elif self.oxygen_level < 50:
            print(f"Warning: Oxygen levels dropping: {self.oxygen_level:.2f}%")
        else:
            print(f"Oxygen level: {self.oxygen_level:.2f}%")
    
    def water_recycling(self):
        self.water_recycled -= random.uniform(0.5, 1.5)
        if self.water_recycled < 20:
            self.is_emergency = True
            print(f"EMERGENCY! Water levels critically low: {self.water_recycled:.2f}%")
            self.send_email("Critical Alert: Water Level", f"Water levels critically low: {self.water_recycled:.2f}%")
            self.send_sms(f"Critical Alert: Water levels critically low: {self.water_recycled:.2f}%")
        elif self.water_recycled < 50:
            print(f"Warning: Water levels dropping: {self.water_recycled:.2f}%")
        else:
            print(f"Water recycled level: {self.water_recycled:.2f}%")

    def temperature_control(self):
        self.temperature += random.uniform(-0.5, 0.5)
        if self.temperature < 16 or self.temperature > 30:
            self.is_emergency = True
            print(f"EMERGENCY! Unsafe temperature detected: {self.temperature:.2f}°C")
        else:
            print(f"Temperature: {self.temperature:.2f}°C")

    def emergency_response(self):
        if self.is_emergency:
            print("ACTIVATING EMERGENCY PROTOCOLS!")
            self.oxygen_level = 75
            self.water_recycled = 75
            self.temperature = 22
            self.is_emergency = False
            print("Emergency resolved. Systems back to normal.")

            # Send email and SMS notifications
            self.send_email(
                subject="Emergency Response Activated",
                body=f"Oxygen levels critically low. Resolved. New levels: Oxygen: {self.oxygen_level:.2f}%, Water: {self.water_recycled:.2f}%, Temperature: {self.temperature:.2f}°C"
            )
            self.send_sms(
                body=f"Emergency resolved! New levels - Oxygen: {self.oxygen_level:.2f}%, Water: {self.water_recycled:.2f}%, Temperature: {self.temperature:.2f}°C"
            )

    def real_time_monitoring(self):
        print("\n--- Monitoring Life Support Systems ---")
        crew_health_data = self.biometric_monitoring()  # Monitor crew health
        self.ai_health_assistance()  # Provide health recommendations
        self.oxygen_production()
        self.water_recycling()
        self.temperature_control()
        self.record_data()  # Record the current data
        self.check_anomalies()  # Check for anomalies
        self.train_predictive_model()  # Train model on historical data
        predicted_temp = self.predict_temperature()
        if predicted_temp is not None:
            print(f"Predicted future temperature: {predicted_temp:.2f}°C")
        if self.is_emergency:
            self.emergency_response()
        print("--- End of Monitoring Cycle ---\n")

# Main simulation loop
def run_simulation():
    system = LifeSupportSystem()
    while True:
        system.real_time_monitoring()
        time.sleep(2)  # Simulate real-time delay (2 seconds between each monitoring)

# Start the life support system simulation
run_simulation()
