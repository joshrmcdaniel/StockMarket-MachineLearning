# import train
import os
import base64

from io import BytesIO
from flask import Flask, render_template
from plot import plot_predictions

app = Flask(__name__)


@app.route('/train/')
def train_model():
    pass


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
    app.run()