from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from impute_model import deep_impute  # Import the imputation function from impute_model.py

app = Flask(__name__)

@app.route('/')
def index():
    # Home page with link to process the file
    return render_template('index.html')

@app.route('/process')
def process_file():
    # Directly use test.csv located in the my_imputation_app folder
    file_path = 'test.csv'

    # Load the scRNA-seq data and run imputation
    data = pd.read_csv(file_path)
    imputed_data, metrics = deep_impute(data)  # Impute missing values and calculate metrics

    # Save the imputed result in the static folder for easy download
    output_file_path = 'static/imputed_test.csv'
    imputed_data.to_csv(output_file_path, index=False)

    # Pass metrics and file path to the results page
    return render_template('results.html', metrics=metrics, file='imputed_test.csv')

if __name__ == '__main__':
    app.run(debug=True)
