import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import csv
from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = 'random string'




@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    columns = ['preg', 'plas', 'pres', 'skin', 'insulin', 'mass', 'pedi', 'age', 'outcome']
    diabetes = pd.read_csv("diabetes_csv.csv", skipinitialspace=True, skiprows=1, names=columns, nrows=600)

    # Normalizing the columns with a lambda expression.
    FEATURES = ['preg', 'plas', 'pres', 'insulin', 'mass', 'pedi', 'age']

    feat_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

    # preparing features and labels for a train test split
    x_data = diabetes.drop('outcome', axis=1)
    labels = diabetes['outcome']
    x_data.loc[x_data['insulin'] == 0, 'insulin'] = x_data['insulin'].mean()
    print(x_data)
    X_train, X_test, y_train, y_test = train_test_split(x_data, labels, test_size=0.33, random_state=101)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    print(X_train)

    input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=10, num_epochs=100, shuffle=True)
    model = tf.estimator.LinearClassifier(feature_columns=feat_cols, n_classes=2, model_dir="ctrain", )

    print(model.train(input_fn=input_func))

    eval_input_func = tf.estimator.inputs.pandas_input_fn(
        x=X_test,
        y=y_test,
        batch_size=10,
        num_epochs=1,
        shuffle=False)

    results = model.evaluate(eval_input_func)
    print(results)
    preg = request.form['preg']
    glucose = request.form['glu']
    bloodpressure = request.form['bp']
    triceps = request.form['tri']
    insulin = request.form['insu']
    bmi = request.form['bmi']
    pedidegree = request.form['pedi']
    age = request.form['age']
    with open('predict.csv', mode='w') as predict_file:
        employee_writer = csv.writer(predict_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        employee_writer.writerow([preg,glucose,bloodpressure,triceps,insulin,bmi,pedidegree,age,'-'])
    pred = pd.read_csv("predict.csv", skipinitialspace=True, names=columns, nrows=1)
    predi = pred.drop('outcome', axis=1)
    pred_input_func = tf.estimator.inputs.pandas_input_fn(
      x=predi,
      batch_size=10,
      num_epochs=1,
      shuffle=False)

    #predictions of generator, so we must convert it to a list
    predictions = model.predict(pred_input_func)
    final=[]
    for key in predictions:
        final.append(key['class_ids'][0])
    if final[0] == 0:
        flash('You have no Diabetes')
        return redirect(url_for('index'))
    else:
        flash('You have Diabetes')
        return redirect(url_for('index'))


if __name__ == "__main__":
    app.run(debug=False)