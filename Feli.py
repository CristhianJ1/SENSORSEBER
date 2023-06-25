import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    # Importamos el dataset del 2017
    dataset = pd.read_csv('./datos/dataO1.csv')

    # Mostramos el reporte estadístico
    print(dataset.describe())

    # Seleccionamos los features que vamos a usar
    X = dataset[['Rain','Temperature','RH','Wind Speed','Wind Direction','PLANTA','FRUTO','SEVERIDAD (%)']]
    y = dataset['INCIDENCIA']

    # Dividimos los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    # Creamos y ajustamos el modelo de regresión lineal
    modelLinear = LinearRegression().fit(X_train, y_train)
    y_predict_linear = modelLinear.predict(X_test)

    # Creamos y ajustamos el modelo de Lasso
    modelLasso = Lasso(alpha=0.2).fit(X_train, y_train)
    y_predict_lasso = modelLasso.predict(X_test)

    # Creamos y ajustamos el modelo de Ridge
    modelRidge = Ridge(alpha=1).fit(X_train, y_train)
    y_predict_ridge = modelRidge.predict(X_test)

    # Creamos y ajustamos el modelo de ElasticNet
    modelElasticNet = ElasticNet(random_state=0).fit(X_train, y_train)
    y_pred_elastic = modelElasticNet.predict(X_test)

    # Calculamos el error medio cuadrado para cada modelo
    linear_loss = mean_squared_error(y_test, y_predict_linear)
    lasso_loss = mean_squared_error(y_test, y_predict_lasso)
    ridge_loss = mean_squared_error(y_test, y_predict_ridge)
    elastic_loss = mean_squared_error(y_test, y_pred_elastic)

    # Mostramos las pérdidas de cada modelo
    print("Linear Loss:", linear_loss)
    print("Lasso Loss:", lasso_loss)
    print("Ridge Loss:", ridge_loss)
    print("ElasticNet Loss:", elastic_loss)

    # Mostramos los coeficientes de cada modelo
    print("="*32)
    print("Coeficientes linear:")
    print(modelLinear.coef_)

    print("="*32)
    print("Coeficientes lasso:")
    print(modelLasso.coef_)

    print("="*32)
    print("Coeficientes ridge:")
    print(modelRidge.coef_)

    print("="*32)
    print("Coeficientes elastic net:")
    print(modelElasticNet.coef_)

    # Calculamos la exactitud (score) de cada modelo
    print("="*32)
    print("Score Lineal:", modelLinear.score(X_test, y_test))
    print("Score Lasso:", modelLasso.score(X_test, y_test))
    print("Score Ridge:", modelRidge.score(X_test, y_test))
    print("Score ElasticNet:", modelElasticNet.score(X_test, y_test))
