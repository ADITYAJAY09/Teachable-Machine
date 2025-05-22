import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_data(raw_data, selected_features, target_column):
    processed_data = []
    for row in raw_data:
        try:
            feature_values = [float(row[feature]) if not pd.isna(row[feature]) else 0.0 for feature in selected_features]
            if task_type == 'classification':
                target_value = int(float(row[target_column])) if not pd.isna(row[target_column]) else 0
            else:
                target_value = float(row[target_column]) if not pd.isna(row[target_column]) else 0.0
            
            if all(not np.isnan(f) for f in feature_values) and not np.isnan(target_value):
                processed_data.append({'features': feature_values, 'target': target_value})
        except (ValueError, TypeError):
            continue
    return processed_data

def create_model(input_shape, output_shape, task_type):
    model = Sequential()
    
    # Input layer
    model.add(Dense(
        units=max(32, input_shape * 2),
        activation='relu',
        input_shape=(input_shape,)
    ))
    
    # Hidden layers
    model.add(Dropout(0.3))
    model.add(Dense(
        units=max(16, input_shape),
        activation='relu'
    ))
    
    model.add(Dropout(0.2))
    
    # Output layer
    if task_type == 'classification':
        model.add(Dense(
            units=output_shape,
            activation='softmax' if output_shape > 2 else 'sigmoid'
        ))
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy' if output_shape > 2 else 'binary_crossentropy',
            metrics=['accuracy']
        )
    else:
        model.add(Dense(
            units=1,
            activation='linear'
        ))
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )
    
    return model

def train_model():
    global model, training_history, predictions, model_performance, scaler
    
    if data is None or len(selected_features) == 0 or not target:
        return {'error': 'Please upload data and select features and target'}, 400

    # Preprocess data
    processed_data = preprocess_data(data, selected_features, target)
    if not processed_data:
        return {'error': 'No valid data after preprocessing'}, 400
    
    # Prepare features and targets
    X = np.array([item['features'] for item in processed_data])
    y = np.array([item['target'] for item in processed_data])
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Determine output shape
    if task_type == 'classification':
        output_shape = len(np.unique(y_train))
    else:
        output_shape = 1
    
    # Create model
    model = create_model(len(selected_features), output_shape, task_type)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    # Store training history
    training_history = []
    for i in range(len(history.history['loss'])):
        epoch_data = {
            'epoch': i + 1,
            'loss': history.history['loss'][i],
            'val_loss': history.history['val_loss'][i]
        }
        
        if task_type == 'classification':
            epoch_data['accuracy'] = history.history['accuracy'][i]
            epoch_data['val_accuracy'] = history.history['val_accuracy'][i]
        else:
            epoch_data['mean_absolute_error'] = history.history['mean_absolute_error'][i]
            epoch_data['val_mean_absolute_error'] = history.history['val_mean_absolute_error'][i]
        
        training_history.append(epoch_data)
    
    # Evaluate model
    evaluation = model.evaluate(X_test, y_test, verbose=0)
    
    model_performance = {
        'test_loss': evaluation[0],
        'test_metric': evaluation[1],
        'metric_name': 'accuracy' if task_type == 'classification' else 'mean_absolute_error'
    }
    
    # Make predictions
    y_pred = model.predict(X_test)
    if task_type == 'classification':
        y_pred = np.argmax(y_pred, axis=1) if output_shape > 2 else np.round(y_pred).flatten()
    else:
        y_pred = y_pred.flatten()
    
    predictions = []
    for i in range(min(10, len(X_test))):
        predictions.append({
            'actual': y_test[i],
            'predicted': y_pred[i],
            'features': X_test[i].tolist()
        })
    
    return {'status': 'Model trained successfully'}

def make_prediction(input_features):
    global model, scaler
    
    if model is None or scaler is None:
        return None
    
    try:
        # Normalize input
        input_features = np.array([input_features])
        normalized_input = scaler.transform(input_features)
        
        # Make prediction
        prediction = model.predict(normalized_input)
        
        if task_type == 'classification':
            output_shape = model.layers[-1].output_shape[-1]
            if output_shape > 2:
                return np.argmax(prediction[0])
            else:
                return np.round(prediction[0][0])
        else:
            return prediction[0][0]
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
        training_history=training_history[-10:],
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
            return jsonify({'prediction': float(prediction)})
        else:
            return jsonify({'error': 'Prediction failed'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)