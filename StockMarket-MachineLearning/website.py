# import train
import os

from flask import Flask, render_template

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
    if not os.path.exists(f'models/{model}.model'):
        return f'Model for {model} not found'


if __name__ == '__main__':
    app.run()