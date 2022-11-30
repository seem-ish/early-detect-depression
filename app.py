from flask import Flask, request, render_template, abort
import pickle
from preprocess import preprocess
from werkzeug.utils import secure_filename
import os
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_DIRECTORY'] = 'static/files'


# [ 156.24722222  229.03021087 1954.          172.05875562   -3.92856925]


@app.route('/')
def home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    file = request.files['file']

    if file:
        flpath = os.path.join(
            app.config['UPLOAD_DIRECTORY'],
            secure_filename(file.filename)
        )
        file.save(flpath)

        X = preprocess(flpath)
        print(type(X))
        model_rc = pickle.load(open('models/random_classifier_model.pkl', 'rb'))
        print(model_rc.predict(X.reshape(1, -1)))
        output = model_rc.predict_proba(X.reshape(1, -1))
        return render_template('index.html', prediction_text=f'Percentage {output}')
    else:
        abort(400, "Error")


if __name__ == "__main__":
    app.run()
