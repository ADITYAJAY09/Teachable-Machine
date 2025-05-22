import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Global variables to store state
data = None
features = []
target = ''
task_type = 'classification'
model = None
training_history = []
predictions = []
model_performance = None
selected_features = []
scaler = None
encoder = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_data(raw_data, selected_features, target_column):
    processed_data = []
    for row in raw_data:
        try:
            feature_values = [float(row[feature]) if not pd.isna(row[feature]) else 0.0 
                           for feature in selected_features]
            target_value = row[target_column]
            
            if all(not np.isnan(f) for f in feature_values) and not pd.isna(target_value):
                processed_data.append({'features': feature_values, 'target': target_value})
        except (ValueError, TypeError):
            continue
    return processed_data

def create_model(task_type):
    if task_type == 'classification':
        return RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        return RandomForestRegressor(n_estimators=100, random_state=42)

def train_model():
    global model, training_history, predictions, model_performance, scaler, encoder
    
    if data is None or len(selected_features) == 0 or not target:
        return {'error': 'Please upload data and select features and target'}, 400

    # Preprocess data
    processed_data = preprocess_data(data, selected_features, target)
    if not processed_data:
        return {'error': 'No valid data after preprocessing'}, 400
    
    # Prepare features and targets
    X = np.array([item['features'] for item in processed_data])
    y = np.array([item['target'] for item in processed_data])
    
    # Encode target if classification
    if task_type == 'classification':
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train model
    model = create_model(task_type)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate performance metrics
    if task_type == 'classification':
        accuracy = accuracy_score(y_test, y_pred)
        model_performance = {
            'test_metric': accuracy,
            'metric_name': 'accuracy'
        }
    else:
        mae = mean_absolute_error(y_test, y_pred)
        model_performance = {
            'test_metric': mae,
            'metric_name': 'mean_absolute_error'
        }
    
    # Store sample predictions
    predictions = []
    for i in range(min(10, len(X_test))):
        predictions.append({
            'actual': y_test[i],
            'predicted': y_pred[i],
            'features': X_test[i].tolist()
        })
    
    return {'status': 'Model trained successfully'}

def make_prediction(input_features):
    global model, scaler, encoder
    
    if model is None or scaler is None:
        return None
    
    try:
        # Normalize input
        input_features = np.array([input_features])
        normalized_input = scaler.transform(input_features)
        
        # Make prediction
        prediction = model.predict(normalized_input)
        
        if task_type == 'classification' and encoder:
            return encoder.inverse_transform([int(prediction[0])])[0]
        return prediction[0]
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    global data, features, target, task_type, selected_features
    
    if request.method == 'POST':
        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Read CSV data
                df = pd.read_csv(filepath)
                data = df.to_dict('records')
                features = list(df.columns)
                selected_features = features[:-1]
                target = features[-1]
                
                # Clean up
                os.remove(filepath)
        
        # Handle configuration changes
        if 'task_type' in request.form:
            task_type = request.form['task_type']
        
        if 'target' in request.form:
            target = request.form['target']
        
        if 'features' in request.form:
            selected_features = request.form.getlist('features')
        
        # Handle training request
        if 'train' in request.form:
            train_model()
    
    return render_template('index.html',
        data=data,
        features=features,
        target=target,
        task_type=task_type,
        selected_features=selected_features,
        training_history=training_history[-10:] if training_history else [],
        model_performance=model_performance,
        predictions=predictions,
        model_exists=model is not None
    )

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'No model trained yet'}), 400
    
    try:
        input_data = request.json
        input_features = [float(input_data[feature]) for feature in selected_features]
        prediction = make_prediction(input_features)
        
        if prediction is not None:
            return jsonify({'prediction': str(prediction)})
        else:
            return jsonify({'error': 'Prediction failed'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
