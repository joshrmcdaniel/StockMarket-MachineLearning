# import train
import os
import base64
import zipfile

from io import BytesIO
from flask import Flask, render_template, send_file, redirect
from flask_restful import reqparse
from plot import plot_predictions
from train import train_model


app = Flask(__name__)


@app.route('/train/')
def train():
    req_parse = reqparse.RequestParser()
    req_parse.add_argument('company', type=str, required=True)
    req_parse.add_argument('prediction_base', type=int, default=90)
    req_parse.add_argument('epochs', type=int, default=25)
    req_parse.add_argument('batch_size', type=int, default=64)

    args = req_parse.parse_args()

    train_model(args.company, args.prediction_base, args.batch_size, args.epochs)
    return redirect(f'/plot/{args.company}')


@app.route('/download/<string:model>')
def download(model):
    mem = BytesIO()
    zip = zipfile.ZipFile(mem, 'w', zipfile.ZIP_DEFLATED)
    for cur, _dirs, files in os.walk(f'models/{model}.model'):
        for f in files:
            path_to_model = os.path.join(cur, f)
            zip_path = path_to_model[path_to_model.index(os.path.sep)+1:]
            zip.writestr(zip_path, open(path_to_model, 'rb').read())
    zip.close()
    mem.seek(0)
    return send_file(mem, mimetype='application/zip', as_attachment=True, attachment_filename=f'{model.upper()}.model.zip')


@app.route('/models/')
def models():
    path = os.getcwd()
    models = [model[:model.rindex('.')] for model in os.listdir('models') if os.path.isdir(os.path.join(path, 'models', model))]
    print(models)
    return render_template('models.html', header='Models', models=models)


@app.route('/plot/<string:model>')
def plot(model):
    model = model.upper()
    if not os.path.exists(f'models/{model}.model'):
        return f'Model for {model} not found'
    graph = plot_predictions(model, 90)
    buf = BytesIO()
    graph.savefig(buf, format="png")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return render_template('plot.html', company=model, data=data)


if __name__ == '__main__':
    app.run(debug=True)
