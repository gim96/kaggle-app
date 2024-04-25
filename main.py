from flask import Flask, request, render_template, redirect
import pandas as pd
import joblib

app = Flask(__name__)

# Load the saved model
model_path = 'random_forest_regressor_model.joblib'
model = joblib.load(model_path)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file:
            # Assuming the uploaded file is a CSV with headers and includes 'Id'
            data = pd.read_csv(file)
            # Prepare the data for prediction
            X_new = data.drop(['SalePrice'], axis=1, errors='ignore')  # In case 'SalePrice' not in uploaded data

            # Make predictions
            predictions = model.predict(X_new.drop(['Id'], axis=1, errors='ignore'))

            # Combine IDs and predictions for displaying
            results = [{'Id': int(id_val), 'SalePrice': float(pred)} for id_val, pred in zip(X_new['Id'], predictions)]
            return render_template('results.html', results=results)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
