from math import sqrt
import operator
from collections import OrderedDict
from sklearn import preprocessing
from sklearn import model_selection
import csv
import numpy as np
import networkx as nx
import os
import zipfile
import time
import copy
from queue import PriorityQueue


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
            if data_type(d[-2]):
                # Skip last two fields and add only specified fields.
                net_params = tuple(f_list[n](i) for n, i in enumerate(d[:-2]) if n_list[n] in selected_parameters)
                result.append(net_params)
                if d[-2] == 'normal':
                    class_labels.append(0)
                else:
                    class_labels.append(1)
    print('Records count: {}'.format(len(result)))
    return result, class_labels


# Реализация нейронного газа в n-мерном пространстве
class GNG:
    def __init__(self, data, eps_b: float, eps_n: float, max_age: int, lambda_: int, alpha: float, d: float,
                 max_nodes: int, verbose: bool):
        self._graph = nx.Graph()    # Объект библиотеки networkx, в котором хранится граф нейронов
        self._data = data   # Обучающая выборка
        self._dev_params = None     # Параметры отклонения
        self._count = 0     # Индекс нейронов
        self._start_time = time.time()  # Время начала работы (см. использование в методах)
        self._eps_b = eps_b     # Скорость обучения нейрона победителя
        self._eps_n = eps_n     # Скорость обучения соседей нейрона победителя
        self._max_age = max_age     # Максимальный возраст дуги (ребра) между нейронами
        self._lambda = lambda_      # Период между итерациями порождения новых нейронов
        self._alpha = alpha     # Затухание накопленных ошибок при создании новых нейронов
        self._d = d     # Затухание ошибок каждую итерацию
        self._max_nodes = max_nodes     # Максимальное количество нейронов
        self.__add_initial_nodes()  # Добавление парочки нейронов в начале алгоритма
        self._verbose = verbose     # Параметр печати/непечати прогресса обучения
        self._every_lambda_iteration = 0    # Параметр счетчика итераций до первого кратного lambda
        self._indexes_queue = PriorityQueue()   # Очередь неиспользованных индексов с приоритетом (меньшим значениям)

    def train(self, amount_of_epochs: int, protocol_results: list):
        self._dev_params = None
        graph = self._graph
        max_nodes = self._max_nodes
        d = self._d
        ld = self._lambda
        alpha = self._alpha
        update_winner = self.__update_winner
        data = self._data
        start_time = self._start_time = time.time()
        train_step = self.__train_step
        current_epoch = 1
        every_lambda_iteration = self._every_lambda_iteration
        while current_epoch <= amount_of_epochs:
            t1 = 0
            t2 = 0
            time_2closest_neurons = 0
            time_neighbours_winner = 0
            time_max_ages = 0
            time_alone_neurons = 0
            time_coord_winner = 0
            time_overall = 0
            for x in data:
                every_lambda_iteration += 1
                begin = time.time()
                time_2closest_neurons, time_neighbours_winner, time_max_ages, time_alone_neurons, time_coord_winner, time_overall = update_winner(x, time_2closest_neurons, time_neighbours_winner, time_max_ages, time_alone_neurons, time_coord_winner, time_overall)
                begin = time.time() - begin
                t1 += begin
                begin = time.time()
                train_step(every_lambda_iteration, alpha, ld, d, max_nodes, graph)
                begin = time.time() - begin
                t2 += begin
            tm = time.time() - start_time
            if self._verbose:
                print('Время обучения = {} s, Время на образ = {} s, Эпоха = {}/{}, Нейронов = {}'.
                      format(round(tm, 2), tm / len(data), current_epoch, amount_of_epochs, len(self._graph)))
            current_epoch += 1
#            print("Общее время работы update_winner: {}".format(t1))
#            print("Общее время работы train_step: {}".format(t2))
#            print("Общее время работы time_2closest_neurons: {}".format(time_2closest_neurons))
#            print("Общее время работы time_neighbours_winner: {}".format(time_neighbours_winner))
#            print("Общее время работы time_max_ages: {}".format(time_max_ages))
#            print("Общее время работы time_alone_neurons: {}".format(time_alone_neurons))
#            print("Общее время работы time_coord_winner: {}".format(time_coord_winner))
#            print("Общее время работы time_overall: {}".format(time_overall))
        protocol_results.append("Время обучения: {} s\n".format(round(time.time() - start_time, 3)))
        if self._verbose:
            print('Обучение завершено, время обучения = {} s'.format(round(time.time() - start_time, 2)))

    def __train_step(self, i, alpha, ld, d, max_nodes, graph):
        g_nodes = graph.nodes
        # Если текущая итерация делится на lambda (ld) без остатка и не достигнуто максимальное количество нейронов
        if i % ld == 0 and len(graph) < max_nodes:
            # Найти нейрон с наибольшей ошибкой
            errorvectors = nx.get_node_attributes(graph, 'error')
            node_largest_error = max(errorvectors.items(), key=operator.itemgetter(1))[0]
            # Найти соседа найденного нейрона с наибольшей ошибкой
            neighbors = graph.neighbors(node_largest_error)
            max_error_neighbor = 0
            max_error = -1
            for n in neighbors:
                ce = g_nodes[n]['error']
                if ce > max_error:
                    max_error = ce
                    max_error_neighbor = n
            # Уменьшить ошибку двух найденных нейронов
            new_max_error = alpha * errorvectors[node_largest_error]
            graph.nodes[node_largest_error]['error'] = new_max_error
            graph.nodes[max_error_neighbor]['error'] = alpha * max_error
            new_node = None
            # Создать новый нейрон между найденными двумя
            if self._indexes_queue.empty() is True:
                print("ОЧередь свободных индексов пуста")
                self._count += 1
                new_node = self._count
            elif self._indexes_queue.empty() is False:
                print("ОЧередь свободных индексов непуста")
                new_node = self._indexes_queue.get()
                print(new_node)
            print(graph.nodes)
            graph.add_node(new_node,
                           pos=self.__get_average_dist(g_nodes[node_largest_error]['pos'],
                                                       g_nodes[max_error_neighbor]['pos']), error=new_max_error)
            # Создать ребра между новым нейроном и найденными двумя
            graph.add_edge(new_node, max_error_neighbor, age=0)
            graph.add_edge(new_node, node_largest_error, age=0)
            # Удалить ребро между найденными двумя
            graph.remove_edge(max_error_neighbor, node_largest_error)
        # Уменьшить ошибки всех нейронов
        for n in graph.nodes():
            oe = g_nodes[n]['error']
            g_nodes[n]['error'] -= d * oe

    # Обнаружение аномалий обсчетом порога путем подсчета корня из отношения расстояния то ближайшего и его индекса
    def _calculate_deviation_params(self, distance_function_params={}):
        if self._dev_params is not None:
            return self._dev_params
        clusters = {}
        dcvd = self._determine_closest_vertice
        dlen = len(self._data)
        #dmean = np.mean(self._data, axis=1)
        #deviation = 0
        for node in self._data:
            n = dcvd(node, **distance_function_params)
            cluster = clusters.setdefault(frozenset(nx.node_connected_component(self._graph, n[0])), [0, 0])
            cluster[0] += n[1]
            cluster[1] += 1
        clusters = {k: sqrt(v[0]/v[1]) for k, v in clusters.items()}
        self._dev_params = clusters
        return clusters

    # Инициализация двух нейронов в начале работы алгоритма
    def __add_initial_nodes(self):
        # Выбор 2 стартовых нейронов из обучающей выборки рандомом
        node1 = self._data[np.random.randint(0, len(self._data))]
        node2 = self._data[np.random.randint(0, len(self._data))]
        # убеждаемся, что не сегенировали одинковых позиций
        if self.__is_nodes_equal(node1, node2):
            raise ValueError("Rerun ---------------> similar nodes selected")
        self._count = 0
        self._graph.add_node(self._count, pos=node1, error=0)
        self._count += 1
        self._graph.add_node(self._count, pos=node2, error=0)
        self._graph.add_edge(self._count - 1, self._count, age=0)

    def __is_nodes_equal(self, n1, n2):
        return len(set(n1) & set(n2)) == len(n1)

    def __update_winner(self, curnode, time_2closest_neuron, time_neighbours_winer, time_maxx_ages, time_alone_neuronss, time_coordd_winner, time_overal):
        begin_overall = time.time()
        # поиск билжайшего нейрона и второго ближайшего нейрона
        begin = time.time()
        winner1, winner2 = self._determine_2closest_vertices(curnode)
        begin = time.time() - begin
        time_2closest_neuron += begin
        winner_node1 = winner1[0]   # Индекс (номер) нейрона победителя (ближайшего) - 1 место
        winner_node2 = winner2[0]   # Индекс (номер) нейрона, ближайшего после победителя - 2 место
        win_dist_from_node = winner1[1]    # Расстояние между победителем и входным вектором
        graph = self._graph     # GNG сеть
        g_nodes = graph.nodes   # Нейроны GNG сети
        g_nodes[winner_node1]['error'] += win_dist_from_node**2   # Для победителя добавляем расстояние^2 в ошибку
        # Обновить местоположение нейрона победителя
        begin = time.time()
        g_nodes[winner_node1]['pos'] += self._eps_b * (curnode - g_nodes[winner_node1]['pos'])
        begin = time.time() - begin
        time_coordd_winner += begin
        # Обновить местоположение соседей нейрона-победителя (все, что соединены ребрами с ним)
        begin = time.time()
        eps_n = self._eps_n
        for n in nx.all_neighbors(graph, winner_node1):
            g_nodes[n]['pos'] += eps_n * (curnode - g_nodes[n]['pos'])
        begin = time.time() - begin
        time_neighbours_winer += begin
        # Увеличить возраст всех ребер, выходящих из нейрона-победителя
        for e in graph.edges(winner_node1, data=True):
            e[2]['age'] += 1
        # Создать ребро между 2мя ближайшими нейронами (если его нет) или обнулить его возраст (если оно есть)
        graph.add_edge(winner_node1, winner_node2, age=0)
        # Удалить ребра, возраст которых превышает максимальный  ИСПРАВИТЬ!!!!
        begin = time.time()
        age_of_edges = nx.get_edge_attributes(graph, 'age')
        max_age = self._max_age
        for edge, age in age_of_edges.items():
            if age >= max_age:
                graph.remove_edge(edge[0], edge[1])
        begin = time.time() - begin
        time_maxx_ages += begin
        begin = time.time()
        # Если есть нейроны, у которых нет ребер, удалить их
        alone_neurons = [node for node in g_nodes if not graph.__getitem__(node)]
        for alone_neuron in alone_neurons:
            self._indexes_queue.put(copy.deepcopy(alone_neuron))
        graph.remove_nodes_from(alone_neurons)
#        for node in g_nodes:
#            if not graph.neighbors(node):
#                self._indexes_queue.put(node)
#                graph.remove_node(node)
        begin = time.time() - begin
        time_alone_neuronss += begin
        begin_overall = time.time() - begin_overall
        time_overal += begin_overall
        return time_2closest_neuron, time_neighbours_winer, time_maxx_ages, time_alone_neuronss, time_coordd_winner, time_overal

    def __get_average_dist(self, a, b):
        return (a + b) / 2

    # Поиск двух ближайших нейронов
    def _determine_2closest_vertices(self, curnode):
        # Where this curnode is actually the x,y index of the data we want to analyze.
        pos = nx.get_node_attributes(self._graph, 'pos')    # Возвращает словарь {номер нейрона: координаты}
        l_pos = len(pos)
        if l_pos == 0:
            return None, None
        elif l_pos == 1:
            return pos[0], None
        kv = list(zip(*pos.items()))  # Преобразование сети в список (0 эл - номера нейронов, 1 эл - вектор координат)
        amount_of_neurons = len(kv[1])
        distances = list()
        for index in range(amount_of_neurons):
            # Норма (корень из суммы квадратов координат) - Евклид
            distances.append(np.linalg.norm(np.subtract(kv[1][index], curnode), ord=2))
        i0, i1 = np.argsort(distances)[0:2]
        winner1 = tuple((kv[0][i0], distances[i0]))  # (номер нейрона, расстояние)
        winner2 = tuple((kv[0][i1], distances[i1]))
        return winner1, winner2

    # Обнаружение аномалий обсчетом порога путем подсчета корня из отношшения расстояния то ближайшего и его индекса
    def detect_anomalies_1(self, data, adaptive_training: bool, save_step: int, protocol_results: list):
        anomalies_counter, anomaly_records_counter, normal_records_counter = 0, 0, 0
        y_classes = list()
        start_time = self._start_time = time.time()
        for i, d in enumerate(data):
            risk_level = self.test_node(d, adaptive_training)
            if risk_level != 0:
                y_classes.append(1)
                anomaly_records_counter += 1
                anomalies_counter += 1
            else:
                normal_records_counter += 1
                y_classes.append(0)
            if i % save_step == 0 and self._verbose:
                tm = time.time() - start_time
                print('Аномальные пакеты: {}, Нормальные пакеты: {}, Время обнаружения: {} s, Время на пакет: {} s'.
                      format(anomaly_records_counter, normal_records_counter, round(tm, 2), tm / i if i else 0))
        tm = time.time() - start_time
        print('{} [Аномлаьные пакеты: {}, Нормальные пакеты: {}, Время работы: {} s, Среднее время на пакет: {} s]'.
              format('Аномалии были обнаружены (count = {})'.format(anomalies_counter) if anomalies_counter else
                     'Аномалий не обнаружено', anomaly_records_counter, normal_records_counter, round(tm, 2),
                     tm / len(data)))
        protocol_results.append("Время обнаружения аномалий в тестовой выборке: {} s\n".format(round(tm, 3)))
        return y_classes

    # Обнаружение аномалий обсчетом порога путем подсчета среднеквадратичного отклонения между ближайшим и его соседями
    def detect_anomalies_2(self, data, adaptive_training: bool, save_step: int, protocol_results: list):
        anomalies_counter, anomaly_records_counter, normal_records_counter = 0, 0, 0
        y_classes = list()
        start_time = self._start_time = time.time()
        iteration = 0
        for packet in data:
            start_time_on_packet = time.time()
            iteration += 1
            anomaly = self.test_packet(packet)
            if anomaly is True:
                y_classes.append(1)
                anomaly_records_counter += 1
                anomalies_counter += 1
            elif anomaly is False:
                normal_records_counter += 1
                y_classes.append(0)
            if iteration % save_step == 0 and self._verbose:
                tm = time.time() - start_time_on_packet
                print('Аномальные пакеты: {}, Нормальные пакеты: {}, Время обнаружения: {} s, Время на запись: {} s'.
                      format(anomaly_records_counter, normal_records_counter, round(tm, 2), tm / iteration if iteration
                      else 0))
        tm = time.time() - start_time
        print('{} [Аномальные пакеты: {}, Нормальные пакеты: {}, Время работы: {} s, Среднее время на пакет: {} s]'.
              format('Аномалии были обнаружены (count = {})'.format(anomalies_counter) if anomalies_counter else
                     'Аномалий не обнаружено', anomaly_records_counter, normal_records_counter, round(tm, 2),
                     tm / len(data)))
        protocol_results.append("Время обнаружения аномалий в тестовой выборке: {} s\n".format(round(tm, 3)))
        return y_classes

    # Обнаружение аномалий обсчетом порога путем подсчета среднеквадратичного отклонения между ближайшим и его соседями
    def test_packet(self, packet):
        n, dist = self._determine_closest_vertice(packet)
        neighbours = nx.node_connected_component(self._graph, n)
        coordinates_neighbours = nx.get_node_attributes(nx.subgraph(self._graph, neighbours), 'pos')
        distances_packet_neighbours = list()
        for coordinates_neighbour in coordinates_neighbours.values():
            distances_packet_neighbours.append(np.linalg.norm(np.subtract(packet, coordinates_neighbour), ord=2))
        distances_packet_neighbours = np.asarray(distances_packet_neighbours)
        deviation = np.std(distances_packet_neighbours)
        if dist <= deviation:
            return False
        elif dist > deviation:
            return True

    # Обнаружение аномалий обсчетом порога путем подсчета корня из отношения расстояния то ближайшего и его индекса
    def test_node(self, node, adaptive_training: bool):
        n, dist = self._determine_closest_vertice(node)
        # Заранее прогоняет всю тестовую выборку, считает отклонения для каждого образа. Потом просто их возвращает
        dev = self._calculate_deviation_params()
        dev = dev.get(frozenset(nx.node_connected_component(self._graph, n)), dist + 1)
        dist_sub_dev = dist - dev
        if dist_sub_dev > 0:
            return dist_sub_dev
        if adaptive_training:
            self._dev_params = None
            self._train_on_data_item(node)
        return 0

    # Поиск ближайшего нейрона
    def _determine_closest_vertice(self, current_packet):
        pos = nx.get_node_attributes(self._graph, 'pos')
        kv = list(zip(*pos.items()))
        amount_of_neurons = len(kv[1])
        distances = list()
        for index in range(amount_of_neurons):
            # Норма (корень из суммы квадратов координат) - Евклид
            distances.append(np.linalg.norm(np.subtract(kv[1][index], current_packet), ord=2))
        i0 = np.argsort(np.asarray(distances))[0]
        return kv[0][i0], distances[i0]


def estimate_quality_algorithm(y_known, y_classified: list):
    evaluation_measures = {'True Positive': 0, 'True Negative': 0, 'False Positive': 0, 'False Negative': 0}
    for index, result in enumerate(y_classified):
        if result == y_known[index] == 1:
            evaluation_measures['True Positive'] += 1
        elif result == y_known[index] == 0:
            evaluation_measures['True Negative'] += 1
        elif y_known[index] == 0 and result == 1:
            evaluation_measures['False Positive'] += 1
        elif y_known[index] == 1 and result == 0:
            evaluation_measures['False Negative'] += 1
    return evaluation_measures


def test_detector(X_train, y_train, X_test, y_test, alg, hyper_parameters: dict, save_trained_model: bool,
                  protocol_results: list, verbose: bool):
    frame = '=' * 100
    print('{}\n{}\n{}'.format(frame, '{} обучение GNG...'.format(alg.__name__), frame))
    gng = alg(X_train, eps_b=hyper_parameters['learning rate winner'], eps_n=hyper_parameters['learning rate neighbour']
              , max_age=hyper_parameters['max age'], lambda_=hyper_parameters['lambda'], alpha=hyper_parameters['alpha']
              , d=hyper_parameters['error attenuation'], max_nodes=hyper_parameters['max neurons'], verbose=verbose)
    gng.train(amount_of_epochs=hyper_parameters['amount of epochs'], protocol_results=protocol_results)
    print('{}\n{}\n{}'.format(frame, 'Тестирование GNG на обучающей выборке (нормальных пакетах)...', frame))
    y_classified = gng.detect_anomalies_2(X_train, adaptive_training=False, save_step=100, protocol_results=
                                          protocol_results)
    quality_measures = estimate_quality_algorithm(y_train, y_classified)
    for key, value in quality_measures.items():
        protocol_results.append(('Мера на обучающей выборке {} = {}\n'.format(key, value)))
    protocol_results.append('Сумма FP и FN мер = {}\n'.format(quality_measures['False Positive'] +
                                                              quality_measures['False Negative']))
    print('{}\n{}\n{}'.format(frame, 'Тестирование GNG на тестовой выборке (полный набор пакетов) без адаптивного '
                                     'обучения...', frame))
    y_classified = gng.detect_anomalies_2(X_test, adaptive_training=False, save_step=100, protocol_results=
                                          protocol_results)
    quality_measures = estimate_quality_algorithm(y_test, y_classified)
    for key, value in quality_measures.items():
        protocol_results.append(('Мера на тестовой выборке {} = {}\n'.format(key, value)))
    protocol_results.append('Сумма FP и FN мер = {}\n'.format(quality_measures['False Positive'] +
                                                              quality_measures['False Negative']))


def preprocessing_dataset(paths_to_dataset: list, labels_file, norm: str, train_test_splitted: bool, normalize: bool):
    # Эта функция возвращает выборки и метки классов в виде списка кортежей
    if train_test_splitted is True and normalize is True:
        training_set = paths_to_dataset[0]
        testing_set = paths_to_dataset[1]
        train_data, train_class_labels = read_ids_data(training_set, activity_type='normal', labels_file=labels_file)
        train_data = preprocessing.normalize(train_data, axis=1, norm=norm, copy=False)
        test_data, test_class_labels = read_ids_data(testing_set, activity_type='full', labels_file=labels_file)
        test_data = preprocessing.normalize(test_data, axis=1, norm=norm, copy=False)
        return train_data, train_class_labels, test_data, test_class_labels
    elif train_test_splitted is False and normalize is True:
        # Эта функция возвращает выборки в виде массива numpy, а метки классов в виде списка
        full_set = paths_to_dataset[0]
        normal_data, class_labels_normal = read_ids_data(full_set, activity_type='normal', labels_file=labels_file)
        normal_data = preprocessing.normalize(normal_data, axis=1, norm=norm, copy=False)
        splitted_normal_data = model_selection.train_test_split(normal_data, class_labels_normal, test_size=0.2)
        anomaly_data, class_labels_anomaly = read_ids_data(full_set, activity_type='abnormal', labels_file=labels_file)
        anomaly_data = preprocessing.normalize(np.array(anomaly_data, dtype='float64'), axis=1, norm=norm, copy=False)
        splitted_full_test = np.vstack((splitted_normal_data[1], anomaly_data))
        splitted_normal_data[3].extend(class_labels_anomaly)
        # X_train,y_train,X_test,y_test
        return splitted_normal_data[0], splitted_normal_data[2], splitted_full_test, splitted_normal_data[3]
    elif train_test_splitted is True and normalize is False:
        training_set = paths_to_dataset[0]
        testing_set = paths_to_dataset[1]
        train_data, train_class_labels = read_ids_data(training_set, activity_type='normal', labels_file=labels_file)
        train_data = np.asarray(train_data, dtype='float64')
        test_data, test_class_labels = read_ids_data(testing_set, activity_type='full', labels_file=labels_file)
        test_data = np.asarray(test_data, dtype='float64')
        return train_data, train_class_labels, test_data, test_class_labels
    elif train_test_splitted is False and normalize is False:
        full_set = paths_to_dataset[0]
        normal_data, class_labels_normal = read_ids_data(full_set, activity_type='normal', labels_file=labels_file)
        splitted_normal_data = model_selection.train_test_split(normal_data, class_labels_normal, test_size=0.2)
        anomaly_data, class_labels_anomaly = read_ids_data(full_set, activity_type='abnormal', labels_file=labels_file)
        anomaly_data = np.array(anomaly_data, dtype='float64')
        splitted_full_test = np.vstack((splitted_normal_data[1], anomaly_data))
        splitted_normal_data[3].extend(class_labels_anomaly)
        # X_train,y_train,X_test,y_test
        return splitted_normal_data[0], splitted_normal_data[2], splitted_full_test, splitted_normal_data[3]


def main():
    name_of_dataset = 'KDD separated by author'
#    path_to_dataset = ['Datasets/KDD/Small Training Set.csv']
    path_to_dataset = ['Datasets/KDD/Small Training Set.csv', 'Datasets/KDD/KDDTest-21.txt']
    labels_file = 'Datasets/KDD/Field Names.csv'
    norm = 'l1'
    train_test_splitted = True
    save_trained_model = False
    protocol_results = list()
    verbose = True
    X_train, y_train, X_test, y_test = preprocessing_dataset(path_to_dataset, labels_file=labels_file, norm=norm,
                                                             train_test_splitted=train_test_splitted, normalize=True)
    hyper_parameters = {'learning rate winner': 0.05, 'learning rate neighbour': 0.0006, 'max age': 15,
                        'lambda': len(X_train) // 5, 'alpha': 0.5, 'error attenuation': 0.005,
                        'max neurons': 170, 'amount of epochs': 80}
    start_time = time.time()
    test_detector(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, alg=GNG,
                  hyper_parameters=hyper_parameters, save_trained_model=save_trained_model,
                  protocol_results=protocol_results, verbose=verbose)
    print('Общее время работы = {}'.format(round(time.time() - start_time, 3)))
    with open('Protocol.txt', 'w') as file:
        file.writelines(protocol_results)
    with open('Parameters of classification.txt', 'w') as file:
        file.write('Этот файл содержит информацию о параметрах, заданных пользователем:\n')
        file.writelines(['Имя датасета: {}\n'.format(name_of_dataset), 'Путь к датасету: {}\n'.format(path_to_dataset),
                         'Нейронная сеть: Self Organized Map\n',
                         'Алгоритм обучения: Growing Neural Gas\n',
                         'Скорость обучения нейрона-победителя: {}\n'.format(hyper_parameters['learning rate winner']),
                    'Скорость обучения соседей победителя: {}\n'.format(hyper_parameters['learning rate neighbour']),
                         'Максимальный возраст ребра: {}\n'.format(hyper_parameters['max age']),
                         'Период между итерациями порождения новых нейронов: {}\n'.format(hyper_parameters['lambda']),
     'Затухание накопленных ошибок при создании новых нейронов: {}\n'.format(hyper_parameters['alpha']),
                         'Затухание ошибок каждую итерацию: {}\n'.format(hyper_parameters['error attenuation']),
                         'Максимальное количество нейронов: {}\n'.format(hyper_parameters['max neurons']),
                         'Количество эпох: {}\n'.format(hyper_parameters['amount of epochs'])])
    hash_value = hash(name_of_dataset + labels_file + str(hyper_parameters['learning rate winner']) +
                      str(hyper_parameters['learning rate neighbour']) + str(hyper_parameters['max age']) +
                      str(hyper_parameters['lambda']) + str(hyper_parameters['alpha']) +
                      str(hyper_parameters['error attenuation']) + str(hyper_parameters['max neurons']) +
                      str(hyper_parameters['amount of epochs']))
    print("Creating zip archive with results...")
    with zipfile.ZipFile(os.getcwd() + r'\results GNG' + str(hash_value) + '.zip', 'w') as zip_archive:
        zip_archive.write('Protocol.txt')
        zip_archive.write('Parameters of classification.txt')
        if save_trained_model is True:
            zip_archive.write(os.getcwd() + 'GNG trained')
    print("Creating zip archive finished")


if __name__ == "__main__":
    main()
