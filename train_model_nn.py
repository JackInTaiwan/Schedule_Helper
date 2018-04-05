import numpy as np
from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import validation_curve
from sklearn.externals import joblib



def labels_encoding (ys, max_labels, comple=False) :
    """
    This is bin encoding, and it would produce 
    the vectors and complementary vectors as well. 
    :param ys: [np.array, shape=(# of data, )] 
    :param max_labels: [int] how many are labels
    :return: [np.array, shape=(# of data, size of vector)] encoded labels in an array
    """

    ys_new = []
    power = 0

    while 2 ** power < max_labels :
        power += 1

    for y in ys :
        y_new = [0] * power
        # concluding complementary labels
        if comple == False : y_new += [0]
        else : y_new += [1]
        for i, num in enumerate(reversed(bin(y)[2:])) :
            y_new[i] = int(num)
        ys_new.append(y_new)
    ys_new = np.array(ys_new)

    return ys_new


def neuralNetworkClaasification(x_train, y_train, hidden_layer_sizes, model_index) :
    ### Pipe of MLPClassification
    pipe_mlps = Pipeline([
        ('mlps', MLPC(hidden_layer_sizes=hidden_layer_sizes, activation='logistic',
                      solver='lbfgs', batch_size='auto',
                      learning_rate='constant', learning_rate_init=0.0001,
                      max_iter=500, shuffle=True))
    ])

    ### Param range for `alpha`
    param_range = [10 ** i for i in range(-5, 0)]
    train_scores, test_scores = validation_curve(
        estimator=pipe_mlps,
        X=x_train,
        y=y_train,
        param_name='mlps__alpha',
        param_range=param_range,
        cv=3
    )
    param_alpha = param_range[np.argmax(test_scores.sum(axis=1))]

    ### MLP Classification
    mlpc_fin = MLPC(hidden_layer_sizes=(50, 5), activation='logistic',
                 solver='lbfgs', alpha=param_alpha, batch_size='auto',
                 learning_rate='constant', learning_rate_init=0.0001,
                 max_iter=1000, shuffle=True)
    mlpc_fin.fit(x_train, y_train)
    print ("... Precision: %s" % mlpc_fin.score(x_train, y_train))

    ### Save model
    path = "/media/jack/Data/Ubuntu/PycharmProjects/MachineLearning/py_ML_8"
    joblib.dump(mlpc_fin, path + "/models_nn/model_nn_%s.pkl" % model_index)


def model_training (data, model_index, max_labels) :
    x_train = data[:, 0:-1]
    y_train = data[:, -1]
    y_train = labels_encoding(y_train, max_labels)
    neuralNetworkClaasification(x_train, y_train, (100, 3), model_index)