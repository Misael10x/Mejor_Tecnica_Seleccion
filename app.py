from flask import Flask, render_template, send_file
import joblib

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/show-result')
def show_result():
    # Cargar el GridSearch y los mejores parámetros
    grid_search = joblib.load('grid_search.pkl')
    best_params = grid_search.best_params_

    # Renderizar el HTML con los mejores parámetros
    return render_template('result.html', best_params=best_params)

@app.route('/download-report')
def download_report():
    # Descargar el reporte de clasificación
    return send_file('classification_report.txt', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
