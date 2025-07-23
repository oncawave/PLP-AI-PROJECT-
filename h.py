import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def simulate_health_data(n=100):
    return pd.DataFrame({
        'timestamp': pd.date_range(start='2023-10-01', periods=n, freq='T'),
        'heart_rate': np.random.randint(60, 100, n),
        'blood_oxygen': np.random.randint(90, 100, n),
        'activity_level': np.random.choice(['low', 'moderate', 'high'], n)
    })

def detect_anomalies(df):
    model = IsolationForest(contamination=0.1, random_state=42)
    df['anomaly'] = model.fit_predict(df[['heart_rate', 'blood_oxygen']])
    df['anomaly'] = df['anomaly'].map({-1: 'Anomaly', 1: 'Normal'})
    return df, model

def train_and_evaluate(df, model):
    X = df[['heart_rate', 'blood_oxygen']]
    y = df['anomaly'].apply(lambda x: 1 if x == 'Anomaly' else 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

def main():
    df = simulate_health_data()
    df, model = detect_anomalies(df)
    train_and_evaluate(df, model)
    print(df[['timestamp', 'heart_rate', 'blood_oxygen', 'anomaly']].head())

if __name__ == '__main__':
