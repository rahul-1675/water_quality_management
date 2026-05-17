from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import io
from utils import load_model, get_metrics, water_quality_advice

app = Flask(__name__)

# Preload model on startup to make it fast
best_model = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/metrics')
def metrics():
    m = get_metrics()
    return jsonify(m)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = ['pH', 'Turbidity', 'DO', 'TDS', 'Temperature']
        
        # Ensure all features are present
        input_data = []
        for feature in features:
            if feature not in data:
                return jsonify({"error": f"Missing feature: {feature}"}), 400
            input_data.append(float(data[feature]))
            
        user_df = pd.DataFrame([input_data], columns=features)
        predicted_wqi = best_model.predict(user_df)[0]
        advice = water_quality_advice(predicted_wqi)
        
        return jsonify({
            "wqi": float(predicted_wqi),
            "advice": advice
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
            
        if not file.filename.endswith('.csv'):
            return jsonify({"error": "Only CSV files are allowed"}), 400
            
        df = pd.read_csv(file)
        
        features = ['pH', 'Turbidity', 'DO', 'TDS', 'Temperature']
        missing_cols = [col for col in features if col not in df.columns]
        if missing_cols:
            return jsonify({"error": f"Missing columns in CSV: {', '.join(missing_cols)}"}), 400
            
        # Predict
        input_df = df[features]
        df['Predicted_WQI'] = best_model.predict(input_df)
        df['Advice'] = df['Predicted_WQI'].apply(water_quality_advice)
        
        # Save to buffer
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name='batch_predictions.csv'
        )
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
