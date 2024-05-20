import pickle

import numpy as np
from flask import Flask, render_template, url_for, request, jsonify

app = Flask(__name__)

menu = [{"name": "Лаба 1", "url": "p_knn"},
        {"name": "Лаба 2", "url": "p_lab2"},
        {"name": "Лаба 3", "url": "p_lab3"}]

loaded_model_knn = pickle.load(open('model/nerw_file.pickle', 'rb'))
loaded_model_tree = pickle.load(open('model/Wine_model', 'rb'))
loaded_model_class = pickle.load(open('model/obuv_model', 'rb'))

@app.route("/")
def index():
    return render_template('index.html', title="Лабораторные работы, выполненные ФИО", menu=menu)


@app.route("/p_knn", methods=['POST', 'GET'])
def f_lab1():
    if request.method == 'GET':
        return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2'])]])
        pred = loaded_model_knn.predict(X_new)
        return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu,
                               class_model="Это: " + pred[0])

@app.route('/knn_api', methods=['get'])
def get_activity():
    request_data = request.get_json()
    X_new = np.array([[float(request_data['age']),
                       float(request_data['weight'])]])
    pred = loaded_model_knn.predict(X_new)

    return jsonify(activity=pred[0])

@app.route('/knn_api_v2', methods=['get']) # http://127.0.0.1:5000/knn_api_v2?age=20&weight=60
def get_sort():
    X_new = np.array([[float(request.args.get('age')),
                       float(request.args.get('weight'))]])
    pred = loaded_model_knn.predict(X_new)

    return jsonify(activity=pred[0])

@app.route("/p_lab2", methods=['POST', 'GET'])
def f_lab2():
    if request.method == 'GET':
        return render_template('lab2.html', title="Логистическая регрессия", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3']),
                           float(request.form['list4']),
                           float(request.form['list5']),
                           float(request.form['list6']),
                           float(request.form['list7']),
                           float(request.form['list8']),
                           float(request.form['list9']),
                           float(request.form['list10']),
                           float(request.form['list11'])]])
        pred = str(loaded_model_tree.predict(X_new))
        return render_template('lab2.html', title="Логистическая регрессия", menu=menu,
                               class_model="Это: " + pred[0])

@app.route("/p_lab3", methods=['POST', 'GET'])
def f_lab3():
    if request.method == 'GET':
        return render_template('lab3.html', title="Логистическая регрессия", menu=menu, class_model='')
    if request.method == 'POST':
        X_new= np.array([[float(request.form['list1']),
                            float(request.form['list2']),
                            float(request.form['list3'])]])
        pred = str(loaded_model_class.predict(X_new))
        return render_template('lab3.html', title="Логистическая регрессия", menu=menu,
                               class_model="Это: " + pred)


if __name__ == "__main__":
    app.run(debug=True)
