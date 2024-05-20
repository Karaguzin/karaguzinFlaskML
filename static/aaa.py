from requests import get
age = input('Введите age = ')
weight = input('Введите weight = ')
print(get(f'http://127.0.0.1:5000/knn_api', json={'age':age,'weight':weight}).json())