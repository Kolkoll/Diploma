from collections import OrderedDict
import csv
import numpy as np
import time
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn import model_selection
import matplotlib.pyplot as plt
import pickle
import zipfile
import os


def sh(s):
    sum = np.float64(0)
    for i, c in enumerate(s):
        sum += i * ord(c)
    return sum


def read_ids_data(data_file, activity_type='', labels_file=''):
    selected_parameters = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
                           'wrong_fragment', 'urgent', 'serror_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                           'dst_host_srv_count', 'count']
    label_dict = OrderedDict()
    result = []
    class_labels = []
    with open(labels_file) as lf:
        labels = csv.reader(lf)
        for label in labels:
            if len(label) == 1 or label[1] == 'continuous':
                label_dict[label[0]] = lambda l: np.float64(l)
            elif label[1] == 'symbolic':
                label_dict[label[0]] = lambda l: sh(l)
    f_list = [i for i in label_dict.values()]
    n_list = [i for i in label_dict.keys()]
    if activity_type == 'normal':
        data_type = lambda t: t == 'normal'
    elif activity_type == 'abnormal':
        data_type = lambda t: t != 'normal'
    elif activity_type == 'full':
        data_type = lambda t: True
    else:
        raise ValueError('`activity_type` must be "normal", "abnormal" or "full"')
    print('Reading {} activity from the file "{}"...'.format(activity_type, data_file))
    with open(data_file) as df:
        data = csv.reader(df)
        for d in data:
            if data_type(d[-2]):    # -2 на неполном датасете, -1 на полном датасете
                # Skip last two fields and add only specified fields.
                net_params = [f_list[n](i) for n, i in enumerate(d[:-2]) if n_list[n] in selected_parameters]
                result.append(net_params)
                if d[-2] == 'normal':
                    class_labels.append(0)
                else:
                    class_labels.append(1)
    print('Records count: {}'.format(len(result)))
    return result, class_labels


def evaluation_alg(results_alg, true_results):
    evaluation_measures = {'True Positive': 0, 'True Negative': 0, 'False Positive': 0, 'False Negative': 0}
    for index, result in enumerate(results_alg):
        if result == true_results[index] == 1:
            evaluation_measures['True Positive'] += 1
        elif result == true_results[index] == 0:
            evaluation_measures['True Negative'] += 1
        elif true_results[index] == 0 and result == 1:
            evaluation_measures['False Positive'] += 1
        elif true_results[index] == 1 and result == 0:
            evaluation_measures['False Negative'] += 1
    return evaluation_measures


def plotting_ROC_curve(fpr, tpr, roc_auc, activation_function: str, solver: str):
    print("Plotting ROC curve...")
    plt.figure()
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.plot(fpr, tpr, 'g', label='ROC ' + solver + ' curve (area = %0.2f)' % roc_auc)
    plt.legend(loc='lower right')
    plt.savefig('ROC, activation function ' + activation_function + ', algorithm ' + solver)
    print("ROC curve plotted")


def test_detector(X_train, y_train, X_test, y_test, activation: str, solver: str, hidden_layer_sizes: tuple,
                  protocol_results: list, max_number_of_iterations: int, learning_rate_init: float,
                  save_trained_model: bool):
    MLP = MLPClassifier(activation=activation, solver=solver, learning_rate='constant',
                        learning_rate_init=learning_rate_init, verbose=True, hidden_layer_sizes=hidden_layer_sizes,
                        max_iter=max_number_of_iterations)
    start_time = time.time()
    MLP.fit(X_train, y_train)
    protocol_results.append("Training time = {}\n".format(round(time.time() - start_time, 2)))
    if save_trained_model is True:
        with open('MLP trained', 'wb') as f:
            pickle.dump(MLP, f)
    elif save_trained_model is False:
        pass
    check = MLP.predict(X_test)
    quality_measures = evaluation_alg(check, y_test)
    for key, value in quality_measures.items():
        protocol_results.append('Measure {} = {}\n'.format(key, value))
    protocol_results.append('The sum of FP and FN measures = {}\n'.format(quality_measures['False Positive'] +
                                                                          quality_measures['False Negative']))
    protocol_results.append('Number of epochs: {}\n'.format(MLP.n_iter_))
    protocol_results.append('Minimum of loss function: {}\n'.format(MLP.loss_))
    raw_results_before_threshold = MLP.predict_proba(X_test)
    fpr, tpr, _ = metrics.roc_curve(y_test, raw_results_before_threshold[:, 1])
    roc_auc = metrics.auc(fpr, tpr)
    return fpr, tpr, roc_auc


# Осуществляет предварительную обработку датасета
def preprocessing_dataset(paths_to_dataset: list, labels_file, norm: str, train_test_splitted: bool):
    if train_test_splitted is True:     # Эта функция возвращает выборки и метки классов в виде списка кортежей
        training_set = paths_to_dataset[0]
        testing_set = paths_to_dataset[1]
        train_data, train_class_labels = read_ids_data(training_set, activity_type='full', labels_file=labels_file)
        train_data = preprocessing.normalize(train_data, axis=1, norm=norm, copy=False)
        test_data, test_class_labels = read_ids_data(testing_set, activity_type='full', labels_file=labels_file)
        test_data = preprocessing.normalize(test_data, axis=1, norm=norm, copy=False)
        return train_data, train_class_labels, test_data, test_class_labels
    elif train_test_splitted is False:
        # Эта функция возвращает выборки в виде массива numpy, а метки классов в виде списка
        full_set = paths_to_dataset[0]
        full_data, class_labels = read_ids_data(full_set, activity_type='full', labels_file=labels_file)
        full_data = preprocessing.normalize(full_data, axis=1, norm=norm, copy=False)
        splitted_data = model_selection.train_test_split(full_data, class_labels, test_size=0.25)
        return splitted_data[0], splitted_data[2], splitted_data[1], splitted_data[3]


# Главная функция, содержит значения параметров: датасет, алгоритм, ФА. Запускает класификацию и возвращает результат
def main():
    name_of_dataset = 'NSL_KDD separated by University of CyberSecurity of Canada'
#    path_to_dataset = ['Datasets/NSL_KDD/KDDCup99_full.csv']
    path_to_dataset = ['Datasets/NSL_KDD_Canada/KDDTrain+.csv', 'Datasets/NSL_KDD_Canada/KDDTest+.csv']
    labels_file = 'Datasets/KDD/Field Names.csv'
    solvers = ('sgd', 'adam')
    activation_function = 'logistic'
    hidden_layer_sizes = (12,)
    max_number_of_iterations = 1500
    learning_rate_value = 0.001
    save_trained_model = False
    train_test_splitted = True
    X_train, y_train, X_test, y_test = preprocessing_dataset(path_to_dataset, labels_file=labels_file, norm='l2',
                                                             train_test_splitted=train_test_splitted)
    for solver in solvers:
        protocol_results = list()
        start_time = time.time()
        fpr, tpr, roc_auc = test_detector(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                          activation=activation_function, solver=solver,
                                          hidden_layer_sizes=hidden_layer_sizes, protocol_results=protocol_results,
                                          max_number_of_iterations=max_number_of_iterations,
                                          learning_rate_init=learning_rate_value, save_trained_model=save_trained_model)
        protocol_results.append('Training and classification time = {}\n'.format(round(time.time() - start_time, 2)))
        plotting_ROC_curve(fpr, tpr, roc_auc, activation_function, solver)
        with open('Protocol.txt', 'w') as file:
            file.writelines(protocol_results)
        with open('Parameters of classification.txt', 'w') as file:
            file.write('This file contains information about parameters, given by user:\n')
            file.writelines(['Name of dataset: {}\n'.format(name_of_dataset),
                             'Training neural network: Multilayer perceptron\n',
                             'Training algorithm: {}\n'.format(solver),
                             'Activation function: {}\n'.format(activation_function),
                             'Amount of neurons in one hidden layer: {}\n'.format(hidden_layer_sizes[0]),
                             'Maximum number of iterations: {}\n'.format(max_number_of_iterations),
                             'Value of learning rate: {}\n'.format(learning_rate_value)])
        hash_value = hash(name_of_dataset + activation_function + str(max_number_of_iterations) + solver +
                          str(hidden_layer_sizes[0]) + str(learning_rate_value))
        print("Creating zip archive with results...")
        with zipfile.ZipFile(os.getcwd() + r'\results MLP' + str(hash_value) + '.zip', 'w') as zip_archive:
            zip_archive.write('Protocol.txt')
            zip_archive.write('Parameters of classification.txt')
            zip_archive.write('ROC, activation function ' + activation_function + ', algorithm ' + solver + '.png')
            if save_trained_model is True:
                zip_archive.write(os.getcwd() + 'MLP trained')
        print("Creating zip archive finished")


# Точка входа
if __name__ == "__main__":
    main()
