# Генетические алгоритмы Leonov Vladislav 181-311
# ФИО автора: Леонов Владислав Денисович
# E-mail автора: bourhood@gmail.com
# Группа: 224-322
# Университет: Московский Политехнический Университет
# Год разработки: 2023
# Учебный курс: https://online.mospolytech.ru/course/view.php?id=10055
########################################################################################################################

# Загружаем библиотеки
import os
import sys                                                            # Предоставляет системе особые параметры и функции
import math
import time
import random                                                         # Для генерации рандомных значений
import numpy as np                                                    # Для работы с массивами
import networkx as nx                                                 # Для рисования графов маршрутов (можно использовать с нейроными сетями) (pip install networkx)

# Библиотеки GUI интерфейса (pip install PyQt5) (Qt Designer to edit *.ui - https://build-system.fman.io/qt-designer-download)
from PyQt5 import uic
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# Связываем matplotlib и PyQt5
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt                                       # Для вывода графиков
from matplotlib.lines import Line2D                                   # Для рисования линий
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# Внедряем многопоточность 
from _other.simple_thread import SimpleThread

''' -------- Главная форма ------- '''
# Унаследовав класс FigureCanvas, этот класс является не только PyQt5 Qwidget, 
# Но и Matplotlib FigureCanvas, который является ключом для соединения pyqt5 и matplotlib.
class Widget_Draw_Graph(FigureCanvas): 
    EDGE_ACTIVE_COLOR = 'r'
    EDGE_COLOR        = '#aaa'
    NODE_COLOR        = '#0099FF'
    NODE_COLOR_START  = '#90EE90'
    NODE_COLOR_END    = '#F08080'
    AdjacencyMatrixTodense = None
    G = None
    pos = None
    edge_weight = None
    node_colors = None
    edge_colors = None
    edgelist_colored = None
    node_list, edge_list, weight_list = None, None, None

    def __init__(self, parent=None, in_width=1000, in_height=1000, in_dpi=100, TypeGraph='Рандомный граф', TopologyGraph='circular_layout'):
        # Создайте рисунок. Примечание. Этот рисунок является рисунком под matplotlib, а не рисунком под matplotlib.pyplot
        # Вызовите метод add_subplot для рисунка, аналогично методу subplot в matplotlib.pyplot
        self.local_width = in_width; self.local_height = in_height; self.local_dpi = in_dpi; 

        self.figure = Figure(figsize=(self.local_width, self.local_height), dpi = self.local_dpi) 
        self.axes = self.figure.add_subplot(111) 

        # Генерируем верширны, границы и веса к ним
        Widget_Draw_Graph.node_list, Widget_Draw_Graph.edge_list, Widget_Draw_Graph.weight_list = self.generate_edges_list(TypeGraph)

        # Отрисовываем вершины, границы и веса к ним 
        self.draw_graph(is_redraw=False, TypeGraph=TypeGraph)

        # Инициализировать родительский класс
        FigureCanvas.__init__(self, self.figure) 
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def generate_edges_list(self, TypeGraph, startV=0, endV=-1, node_count=-1):
        # Выставляем стандартные значения при загрузки приложения - вершин 5, начальная точка 0, конечная точка 1
        if node_count == -1: node_count = 5; 
        if startV >= node_count: startV = node_count - 1; 
        if endV == -1 or endV >= node_count: endV = node_count - 1; 

        # Добавить ребра по списку
        node_list = []
        [node_list.append(i + 1) for i in range(node_count)]

        # Генерируем список вершин
        edge_list = []
        weight_list = []
        for i in range(node_count):
            for j in range(i):
                local_weight = random.randint(1, 10)
                # Полносвязный граф
                if TypeGraph == 'Полносвязный граф':
                    edge_list.append((i+1, j+1))
                    weight_list.append(local_weight)
                # Рандомный граф
                elif TypeGraph == 'Рандомный граф':
                    if random.random() > 0.5:
                        edge_list.append((i+1, j+1))
                        weight_list.append(local_weight)
                # Неполносвязный граф
                elif TypeGraph == 'Неполносвязный граф':
                    if (abs(i-j) not in (1,node_count-1)):
                        edge_list.append((i+1, j+1))
                        weight_list.append(local_weight)
                
        # Проверяем, чтобы у вершины было хоть 1 соединение, если нет, то рандомно добавляем его
        edge_to_list = []
        edge_from_list = []
        for edge in edge_list:
            edge_to_list.append(edge[1])
            edge_from_list.append(edge[0])

        for i in range(node_count):
            node_number = i + 1

            # Проверяем, чтобы Start и End были соединены
            if i == startV:
                if node_number not in edge_to_list and node_number not in edge_from_list:
                    j = endV
                    local_weight = random.randint(1, 10)
                    edge_list.append((i+1, j+1))
                    weight_list.append(local_weight)

            # Проверяем, чтобы у каждой вершины было хоть 1 соединение
            if node_number not in edge_to_list and node_number not in edge_from_list:
                j = i
                while i == j:
                    j = random.randint(0, node_count)
                local_weight = random.randint(1, 10)
                edge_list.append((i+1, j+1))
                weight_list.append(local_weight)

        return node_list, edge_list, weight_list
    
    def clear_graph(self):
        for ax in self.figure.axes:
            ax.clear()
        self.draw()

    def update_weight_on_graph(self):
        G_OLD = Widget_Draw_Graph.G
        pos_local = Widget_Draw_Graph.pos
        edge_weight = Widget_Draw_Graph.edge_weight
        nx.draw_networkx_edge_labels(G_OLD, ax = self.axes, pos = pos_local, edge_labels = edge_weight, font_size=8)
        self.draw()

    def draw_best_graph(self, best_ways, startV, endV):
        G_OLD = Widget_Draw_Graph.G
        pos_local = Widget_Draw_Graph.pos
        edge_weight = Widget_Draw_Graph.edge_weight
        edge_numbers = G_OLD.number_of_edges()
        edge_list = Widget_Draw_Graph.edge_list
        EDGE_ACTIVE_COLOR = Widget_Draw_Graph.EDGE_ACTIVE_COLOR

        # Перебираем все полученные маршруты и пририсовываем маршруты графику
        prev = startV
        edgelist_colored = []
        best_way = best_ways[endV]
        full_way = best_way[:best_way.index(endV)+1]
        for next in full_way:
            if next > edge_numbers: continue; 

            for added_edge in edge_list:
                if prev+1 == next+1: continue; 
                if prev+1 in added_edge and next+1 in added_edge:
                    edgelist_colored.append(added_edge)
                    break

            prev = next

        nx.draw_networkx_edges(G_OLD, pos = pos_local, ax = self.axes, edgelist=edgelist_colored, width=8, alpha=1, edge_color=EDGE_ACTIVE_COLOR)
        nx.draw_networkx_edge_labels(G_OLD, ax = self.axes, pos = pos_local, edge_labels = edge_weight, font_size=8)

        Widget_Draw_Graph.edgelist_colored = edgelist_colored.copy()

        self.draw()

    def draw_graph(self, startV=0, endV=-1, node_count=-1, is_redraw=True, is_need_new_weight=True, TypeGraph='Рандомный граф', AdjacencyMatrixTodense=None, inTopologyGraph='circular_layout'):
        if node_count == -1: node_count = 5; 

        node_list, edge_list, weight_list = Widget_Draw_Graph.node_list, Widget_Draw_Graph.edge_list, Widget_Draw_Graph.weight_list

        # Создаем график для связи всех параметров
        G=nx.Graph()

        # Добавляем веришны (ноды) (кружочки) по списку
        [G.add_node(node) for node in node_list]

        # Добавить ребра (соединения вершин) по списку
        [G.add_edge(edge[0], edge[1], weight=weight) for edge, weight in zip(edge_list, weight_list)]

        pos_local = nx.circular_layout(G)
        if inTopologyGraph == 'circular_layout':
            pos_local = nx.circular_layout(G)
        elif inTopologyGraph == 'random_layout':
            pos_local = nx.random_layout(G)
        elif inTopologyGraph == 'shell_layout':
            pos_local = nx.shell_layout(G)
        elif inTopologyGraph == 'spiral_layout':
            pos_local = nx.spiral_layout(G)
        elif inTopologyGraph == 'kamada_kawai_layout':
            pos_local = nx.kamada_kawai_layout(G)

        edge_weight = nx.get_edge_attributes(G,'weight')
        edge_numbers = G.number_of_edges()
        node_numbers = G.number_of_nodes()
        edge_colors = [Widget_Draw_Graph.EDGE_COLOR for i in range(edge_numbers)]
        node_colors = [Widget_Draw_Graph.NODE_COLOR for i in range(node_numbers)]

        if startV >= node_numbers: startV = node_numbers - 1; 
        if endV == -1 or endV >= node_numbers: endV = node_numbers - 1; 
        node_colors[startV] = Widget_Draw_Graph.NODE_COLOR_START
        node_colors[endV]   = Widget_Draw_Graph.NODE_COLOR_END

        nx.draw(G, ax = self.axes, with_labels = True, edge_color=edge_colors, pos = pos_local, node_color=node_colors, node_size = 1000, width = 3)
        nx.draw_networkx_edge_labels(G, ax = self.axes, pos = pos_local, edge_labels = edge_weight, font_size=8)
        
        # pip install scipy
        # Формируем матрицу смежности для ген. алгоритма
        N = nx.adjacency_matrix(G)
        A = N.todense()
        list_edges = A.tolist()
        for i in range(len(list_edges)):
            for j in range(len(list_edges[0])):
                value = list_edges[i][j]
                if value == 0:
                    if i == j: list_edges[i][j] = 0; # 0, так как значение входит само в себя
                    else: list_edges[i][j] = 100;    # Большое число, чтобы там где нет дорог, не обучались

        Widget_Draw_Graph.AdjacencyMatrixTodense = list_edges
        Widget_Draw_Graph.G = G.copy()
        Widget_Draw_Graph.pos = pos_local.copy()
        Widget_Draw_Graph.edge_weight = edge_weight.copy()
        Widget_Draw_Graph.edge_colors = edge_colors.copy()
        Widget_Draw_Graph.node_colors = node_colors.copy()
        #print(N)
        #print(A)    
        # Экспортировать созданную выше матрицу A весов соединений в файл csv
        # np.savetxt('correlation_matrix.csv',A, delimiter = ',')
        
        if is_redraw: self.draw(); 
            
class Window(QMainWindow):       
    def __init__(self, *args, **kwargs):
        super(Window, self).__init__(*args, **kwargs)

        self.formOpening()                                            # Настройки и запуск формы
        
        # Выбор типа сети
        list_items = ["Полносвязный граф", "Неполносвязный граф", "Рандомный граф"]
        self.ui.TypeGraph_ComboBox.addItems(list_items)
        TypeGraph = self.ui.TypeGraph_ComboBox.currentText()

        # Выбор топологии
        list_items_topology = ["circular_layout", "random_layout", 'shell_layout', 'spiral_layout', 'kamada_kawai_layout']
        self.ui.TopologyGraph_ComboBox.addItems(list_items_topology)
        TopologyGraph = self.ui.TopologyGraph_ComboBox.currentText()

        # Настройки рандома
        if self.ui.UseRandomSettings_CheckBox.isChecked():
            RANDOM_SEED = 42         # Зерно для того, чтобы рандом всегда был одним и тем же
            RANDOM_SEED = self.ui.UseRandomSettings_IntSpinBox.value()
            random.seed(RANDOM_SEED) # Присваиваем зерно для рандома
        else:
            random.seed(int(1000 * time.time()) % 2**32)

        # Инициализация виджета для вывода графиков
        layout = QVBoxLayout()
        self.canvas = Widget_Draw_Graph(self, TypeGraph=TypeGraph, TopologyGraph=TopologyGraph)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar, Qt.AlignCenter)
        layout.addWidget(self.canvas, Qt.AlignCenter)
        self.ui.RouteGraph_Widget.setLayout(layout)

        # Подписки на события
        self.ui.bUseGenAlg.clicked.connect(self.UseGenAlg_Clicked)                                          # Использовать генетический алгоритм
        self.ui.NODE_START_INDEX_IntSpinBox.valueChanged.connect(self.NODE_START_INDEX_IntSpinBox_Changed)  # Изменение начальной вершины
        self.ui.NODE_END_INDEX_IntSpinBox.valueChanged.connect(self.NODE_START_INDEX_IntSpinBox_Changed)    # Изменение конечной вершины
        self.ui.NODE_COUNT_IntSpinBox.valueChanged.connect(self.NODE_COUNT_IntSpinBox_Changed)              # Изменение кол-ва вершин
        self.ui.UseRandomSettings_CheckBox.stateChanged.connect(self.UseRandomSettings_CheckBox_Changed)    # Изменения параметра использовать настройки рандома
        
        self.ui.TopologyGraph_ComboBox.currentTextChanged.connect(self.UseRandomSettings_CheckBox_currentTextChanged) 

        self.ui.bUseGenRandomGraph.clicked.connect(self.NODE_COUNT_IntSpinBox_Changed)

        self.ui.bDropWayGraph.clicked.connect(self.NODE_START_INDEX_IntSpinBox_Changed)

        self.ui.WorkType_RadioButton.clicked.connect(self.check)

        # Выводим данные весов в таблицу приложения
        self.SendDataWeightToDataGrid_AdjacencyMatrixTodense()
        
        self.ui.tableWidget_Weight.cellChanged.connect(self.check_change)
        
        self.ui.SystemMassage_TextBrowser.setText('Статус: приложение готово к работе')
    
    def formOpening(self):
        # Настройки окна главной формы
        file_ui_path = 'GUI.ui' if os.path.isfile('GUI.ui') is True else 'L1/GUI.ui'
        file_icon_path = 'surflay.ico' if os.path.isfile('surflay.ico') is True else 'L1/surflay.ico'
        self.ui = uic.loadUi(file_ui_path)                            # GUI, должен быть в папке с main.py
        self.ui.setWindowTitle('Леонов Владислав 224-322 - Лаб. 1')   # Название главного окна
        self.ui.setWindowIcon(QIcon(file_icon_path))                  # Иконка на гланое окно
        self.ui.show()                                                # Открываем окно формы  

    # method called by radio button
    def check(self):
        # checking if it is checked
        if self.ui.WorkType_RadioButton.isChecked():
            self.ui.NextGeneration_PushButton.setEnabled(True)
        # if it is not checked
        else:
            self.ui.NextGeneration_PushButton.setEnabled(False)
             

    def NODE_START_INDEX_IntSpinBox_Changed(self):
        get_value_TypeGraph = self.ui.TypeGraph_ComboBox.currentText()
        get_value_startV    = self.ui.NODE_START_INDEX_IntSpinBox.value() - 1
        get_value_endV      = self.ui.NODE_END_INDEX_IntSpinBox.value() - 1
        get_value_NodeCount = self.ui.NODE_COUNT_IntSpinBox.value()
        TopologyGraph = self.ui.TopologyGraph_ComboBox.currentText()
        self.canvas.clear_graph()
        self.canvas.draw_graph(startV=get_value_startV, endV=get_value_endV, node_count=get_value_NodeCount, TypeGraph=get_value_TypeGraph, is_need_new_weight=False, inTopologyGraph=TopologyGraph)

    def NODE_COUNT_IntSpinBox_Changed(self):
        get_value_TypeGraph = self.ui.TypeGraph_ComboBox.currentText()
        get_value_NodeCount = self.ui.NODE_COUNT_IntSpinBox.value()
        get_value_startV    = self.ui.NODE_START_INDEX_IntSpinBox.value()
        get_value_endV      = self.ui.NODE_END_INDEX_IntSpinBox.value()
        TopologyGraph = self.ui.TopologyGraph_ComboBox.currentText()

        if get_value_NodeCount <= get_value_endV:   self.ui.NODE_END_INDEX_IntSpinBox.setValue(get_value_NodeCount); 
        if get_value_startV == get_value_endV:      self.ui.NODE_END_INDEX_IntSpinBox.setValue(get_value_NodeCount); 
        if get_value_NodeCount >= get_value_startV: self.ui.NODE_START_INDEX_IntSpinBox.setValue(1); 

        get_value_startV -= 1; 
        get_value_endV -= 1; 

        Widget_Draw_Graph.node_list, Widget_Draw_Graph.edge_list, Widget_Draw_Graph.weight_list =  self.canvas.generate_edges_list(startV=get_value_startV, endV=get_value_endV, node_count=get_value_NodeCount, TypeGraph=get_value_TypeGraph)

        # Рисуем на графике
        self.canvas.clear_graph()
        self.canvas.draw_graph(startV=get_value_startV, endV=get_value_endV, node_count=get_value_NodeCount, TypeGraph=get_value_TypeGraph, inTopologyGraph=TopologyGraph)

        # Выводим данные весов в таблицу приложения после отрисовки графика так как обновится AdjacencyMatrixTodense
        self.SendDataWeightToDataGrid_AdjacencyMatrixTodense()

    def UseRandomSettings_CheckBox_currentTextChanged(self, value):
        get_value_TypeGraph = self.ui.TypeGraph_ComboBox.currentText()
        get_value_startV    = self.ui.NODE_START_INDEX_IntSpinBox.value() - 1
        get_value_endV      = self.ui.NODE_END_INDEX_IntSpinBox.value() - 1
        get_value_NodeCount = self.ui.NODE_COUNT_IntSpinBox.value()
        self.canvas.clear_graph()
        self.canvas.draw_graph(startV=get_value_startV, endV=get_value_endV, node_count=get_value_NodeCount, TypeGraph=get_value_TypeGraph, is_need_new_weight=False, inTopologyGraph=value)

    def UseRandomSettings_CheckBox_Changed(self):
        get_value_IsUseRandomSettings = self.ui.UseRandomSettings_CheckBox.isChecked()
        if get_value_IsUseRandomSettings:
            self.ui.UseRandomSettings_Label.setEnabled(True)
            self.ui.UseRandomSettings_IntSpinBox.setEnabled(True)
        else:
            self.ui.UseRandomSettings_Label.setEnabled(False)
            self.ui.UseRandomSettings_IntSpinBox.setEnabled(False)

    is_changed_value_on_check_change = False
    def check_change(self):
        if Window.is_changed_value_on_check_change: return; 
    
        # Получаем параметры
        current_item = self.ui.tableWidget_Weight.currentItem()
        if current_item is None: return; 
        row = current_item.row()
        col = current_item.column()
        cell_text = current_item.text()

        # Изменяем в матрице для ген. алг.
        Widget_Draw_Graph.AdjacencyMatrixTodense[row][col] = int(cell_text)
        Widget_Draw_Graph.AdjacencyMatrixTodense[col][row] = int(cell_text)
        Window.D = Widget_Draw_Graph.AdjacencyMatrixTodense

        # Изменяем в весах для перерисовки
        edge = (col + 1, row + 1) if (col + 1, row + 1) in Widget_Draw_Graph.edge_list else (row + 1, col + 1)
        index_edge = Widget_Draw_Graph.edge_list.index(edge)
        Widget_Draw_Graph.weight_list[index_edge] = int(cell_text)

        # Изменяем в весах для перерисовки
        edge = (row + 1, col + 1) if (row + 1, col + 1) in Widget_Draw_Graph.edge_weight else (col + 1, row + 1)
        Widget_Draw_Graph.edge_weight[edge] = int(cell_text)

        self.canvas.update_weight_on_graph()

        # Изменяем в таблице приложения
        Window.is_changed_value_on_check_change = True
        self.ui.tableWidget_Weight.setItem(col, row, QTableWidgetItem(cell_text))
        Window.is_changed_value_on_check_change = False

    def SendDataWeightToDataGrid_AdjacencyMatrixTodense(self):
        AdjacencyMatrixTodense = Widget_Draw_Graph.AdjacencyMatrixTodense

        count_row = len(AdjacencyMatrixTodense)
        count_cell = len(AdjacencyMatrixTodense[0])

        self.ui.tableWidget_Weight.clear()
        self.ui.tableWidget_Weight.setRowCount(count_row)
        self.ui.tableWidget_Weight.setColumnCount(count_cell)

        for i in range(count_row):
            for j in range(count_cell):
                value = AdjacencyMatrixTodense[i][j]
                self.ui.tableWidget_Weight.setItem(i, j, QTableWidgetItem(str(value)))

        self.ui.tableWidget_Weight.resizeColumnsToContents()
        self.ui.tableWidget_Weight.resizeRowsToContents()

    def Send_TableWidget_GenAlg(self, inList):
        if inList is None: return; 
    
        count_row = len(inList)
        count_cell = len(inList[0])

        self.ui.tableWidget_GenAlg.clear()
        self.ui.tableWidget_GenAlg.setRowCount(count_row)
        self.ui.tableWidget_GenAlg.setColumnCount(count_cell)

        for i in range(count_row):
            for j in range(count_cell):
                value = inList[i][j]
                self.ui.tableWidget_GenAlg.setItem(i, j, QTableWidgetItem(str(value)))

        self.ui.tableWidget_GenAlg.resizeColumnsToContents()
        self.ui.tableWidget_GenAlg.resizeRowsToContents()

    def bUseGenRandomGraph_Clicked(self):
        pass
    
    def bDropWayGraph_Clicked(self):
        self.canvas.drop_best_graph()

    D = None
    def UseGenAlg_Clicked(self):
        self.NODE_START_INDEX_IntSpinBox_Changed()

        '''
        # Константы задачи
        inf = 100
        D = ((0, 3, 1, 3, inf, inf),
            (3, 0, 4, inf, inf, inf),
            (1, 4, 0, inf, 7, 5),
            (3, inf, inf, 0, inf, 2),
            (inf, inf, 7, inf, 0, 4),
            (inf, inf, 5, 2, 4, 0))
        '''
        ''' Кратчайший путь равен 20 '''

        Window.D = Widget_Draw_Graph.AdjacencyMatrixTodense
        print(f'Матрица смежности{Window.D}')
        self.ui.SystemMassage_TextBrowser.setText(f'Матрица смежности{Window.D}')
        
        startV = 0                                        # Стартовая вершина
        LENGTH_D = len(Window.D)                          # Длина таблицы маршрутов (кол-во вершин)
        LENGTH_CHROM = len(Window.D) * len(Window.D[0])   # Длина хромосомы, полежащей оптимизации
        
        startV = self.ui.NODE_START_INDEX_IntSpinBox.value() - 1
        endV = self.ui.NODE_END_INDEX_IntSpinBox.value() - 1

        # Константы генетического алгоритма
        POPULATION_SIZE = 500    # Кол-во индивидуумов в популяции
        P_CROSSOVER = 0.9        # Вероятность скрещивания
        P_MUTATION = 0.1         # Вероятность мутации
        MAX_GENERATIONS = 50     # Максимальное кол-во поколений

        POPULATION_SIZE = self.ui.POPULATION_SIZE_IntSpinBox.value()
        P_CROSSOVER     = self.ui.P_CROSSOVER_DoubleSpinBox.value()
        P_MUTATION      = self.ui.P_MUTATION_DoubleSpinBox.value()
        MAX_GENERATIONS = self.ui.MAX_GENERATIONS_IntSpinBox.value()

        # Настройки рандома
        if self.ui.UseRandomSettings_CheckBox.isChecked():
            RANDOM_SEED = 42         # Зерно для того, чтобы рандом всегда был одним и тем же
            RANDOM_SEED = self.ui.UseRandomSettings_IntSpinBox.value()
            random.seed(RANDOM_SEED) # Присваиваем зерно для рандома
        else:
            random.seed(int(1000 * time.time()) % 2**32)

        class FitnessMin():
            '''
            Значения приспособленности особи\n
            '''
            def __init__(self):
                self.values = 0

        class Individual(list):
            '''
            Особь\n
            Каждое решение – элемент популяции, называется особью
            '''
            def __init__(self, *args):
                super().__init__(*args)
                self.fitness = FitnessMin()

        def oneDijkstraFitness(individual):
            '''
            Функция приспособленности (англ. fitness function)\n
            Механизм оценки приспособленности каждой особи к текущим условиями окружающей среды\n
            Алгоритм Дейкстры: https://python-scripts.com/dijkstras-algorithm
            '''
            s = 0
            for n, path in enumerate(individual):
                # n - текущий номер узла до которого определяем маршрут
                index_path = path.index(n)
                path = path[:index_path+1]
                # Считаем длину текущего маршрута, используя значения смежной матрицы D
                si = startV
                for j in path:
                    edge_values = Window.D[si]
                    s += int(edge_values[j])
                    si = j
            return s


        def individualCreator():
            '''
            Создаем особь\n
            [[0,1,2...],[...],...,[...]]\n
            Которая не повторяется в пределах списка
            '''
            random_values_list = []
            while len(random_values_list) < LENGTH_D:
                random_values_path = []
                # Пока длина списка меньше N, выполняем цикл
                while len(random_values_path) < LENGTH_D:
                    # Генерируем случайное целое число от 0 до Длины - 1
                    number = random.randint(0, LENGTH_D - 1)
                    # Проверяем, есть ли это число уже в списке
                    if number not in random_values_path:
                        # Если нет, добавляем в список
                        random_values_path.append(number)
                # Проверяем, есть ли список уже в списке
                if random_values_path not in random_values_list:
                    # Если нет, добавляем в список
                    random_values_list.append(random_values_path)
            return Individual(random_values_list)

        def populationCreator(n=0):
            '''Создаем популяцию из n особей'''
            return list([individualCreator() for i in range(n)])

        def clone(value):
            # Создаем хромосому на основе списка (value[:] - делает копию списка)
            ind = Individual(value[:])
            ind.fitness.values = value.fitness.values
            return ind

        def selTournament(population, p_len):
            '''Турнирный отбор'''
            offspring = []
            for n in range(p_len):
                # Получаем 3 рандомных НЕ одинаковых индекса
                i1 = i2 = i3 = 0
                while i1 == i2 or i1 == i3 or i2 == i3:
                    i1, i2, i3 = random.randint(0, p_len - 1), random.randint(0, p_len - 1), random.randint(0, p_len - 1)
                p1, p2, p3 = population[i1], population[i2], population[i3]
                min_offspring = min([p1, p2, p3], key=lambda ind: ind.fitness.values)
                offspring.append(min_offspring)
            return offspring

        def cxOrdered(in_parent1, in_parent2):
            '''Упорядоченное скрещивание'''
            '''https://proproprogs.ru/ga/ga-obzor-metodov-otbora-skreshchivaniya-i-mutacii'''
            copy_parent1, copy_parent2 = in_parent1[:], in_parent2[:]
            for parent1, parent2 in zip(copy_parent1, copy_parent2):
                ind1, ind2 = parent1[:], parent2[:]

                # Выбираем случайный участок хромосомы одного из родителей
                # Генерируем случайное целое число от 0 до Длины - 1 и Проверяем, есть ли это число уже в списке
                random_values = []
                size = min(len(ind1), len(ind2))
                while len(random_values) < 2:
                    number = random.randint(0, size - 1); 
                    if number not in random_values: random_values.append(number); 
                a, b = min(random_values), max(random_values)

                # Копируем этот участок в потомков (двухточечное скрещивание)
                temp1, temp2 = ind1[:], ind2[:]
                ind1, ind2 = [100] * size, [100] * size
                ind1[a:b+1], ind2[a:b+1] = temp2[a:b+1], temp1[a:b+1]
                
                # Заполняем оставшиеся позиции потомков генами родителя в том же порядке
                index1, index2 = b+1, b+1
                for i in range(size): 
                    # Переходим к началу хромосомы, если достигли конца
                    if index1 >= size: index1 = 0
                    if index2 >= size: index2 = 0

                    # Проверяем, есть ли ген родителя в потомке
                    # Если нет, то добавляем его в свободную позицию и Переходим к следующей позиции
                    if temp1[i] not in ind1:
                        ind1[index1] = temp1[i]
                        index1 += 1
                    
                    # Проверяем, есть ли ген родителя в потомке
                    # Если нет, то добавляем его в свободную позицию и Переходим к следующей позиции
                    if temp2[i] not in ind2:
                        ind2[index2] = temp2[i]
                        index2 += 1
                        
                in_parent1[copy_parent1.index(parent1)], in_parent2[copy_parent2.index(parent2)] = ind1, ind2

        def cxOrdered_v2(in_parent1, in_parent2):
            '''Упорядоченное скрещивание'''
            '''На основе библиотеки https://deap.readthedocs.io/en/master/api/tools.html#deap.tools.cxOrdered'''
            '''Goldberg 1989'''
            copy_parent1, copy_parent2 = in_parent1[:], in_parent2[:]
            for child1, child2 in zip(copy_parent1, copy_parent2):
                ind1, ind2 = child1[:], child2[:]

                # Выбираем случайный участок хромосом
                size = min(len(ind1), len(ind2))
                a, b = random.sample(range(size), 2)
                if a > b:
                    a, b = b, a
                
                # Создаем дыры, участки, которые будут НЕизменяемыми если True
                # Причем дыры 1 потомка берутся из родителя 2 и наоборот
                holes1, holes2 = [True] * size, [True] * size
                for i in range(size):
                    if i < a or i > b:
                        holes1[ind2[i]] = False
                        holes2[ind1[i]] = False

                # Заполняем 
                # Мы должны где-то сохранить исходные значения, прежде чем все перемешать
                temp1, temp2 = ind1, ind2
                k1, k2 = b + 1, b + 1
                for i in range(size):
                    value_temp1 = temp1[(i + b + 1) % size]
                    if not holes1[value_temp1]:
                        ind1[k1 % size] = temp1[(i + b + 1) % size]
                        k1 += 1

                    if not holes2[temp2[(i + b + 1) % size]]:
                        ind2[k2 % size] = temp2[(i + b + 1) % size]
                        k2 += 1

                # Меняем местами содержимое между a и b
                for i in range(a, b + 1):
                    new_val1, new_val2 = ind2[i], ind1[i]
                    ind1[i], ind2[i] = new_val1, new_val2
                    
                in_parent1[copy_parent1.index(child1)], in_parent2[copy_parent2.index(child2)] = ind1, ind2


        def mutationShuffleIndexes(in_mutant, indpd=0.01):
            '''
            Мутация порядка генов в хромосоме\n
            Мутация - это случайный обмен двух генов местами
            '''
            # Получаем длину хромосомы
            copy_mutant = in_mutant[:]
            for select_mutant in copy_mutant:
                new_mutant = select_mutant[:]
                length = len(new_mutant)
                # Для каждого гена в хромосоме
                for i in range(length):
                    # Случайно выбираем, будет ли мутация
                    if random.random() < indpd:
                        j = i
                        # Случайно выбираем другой ген для обмена местами
                        while j == i:
                            j = random.randint(0, length - 1)
                        # Меняем местами гены i и j
                        new_value_i, new_value_j = new_mutant[j], new_mutant[i]
                        new_mutant[i], new_mutant[j] = new_value_i, new_value_j
                        in_mutant[copy_mutant.index(select_mutant)] = new_mutant

        # Обнуляем статистику
        generationCounter = 0
        minFitnessValues = []
        avgFitnessValues = []
        vals = []
        best = None

        # Создаем начальную популяцию
        population = populationCreator(n=POPULATION_SIZE)

        fitnessValues = list(map(oneDijkstraFitness, population))
        for individual, fitnessValue in zip(population, fitnessValues):
            individual.fitness.values = fitnessValue

        fitnessValues = [ind.fitness.values for ind in population]
        # До тех пор пока не найдем лучшее решенее или не пройдем все поколения
        while generationCounter < MAX_GENERATIONS:
            
            # Запоминаем параметры сети
            # if self.ui.WorkType_RadioButton.isChecked() is True: pass; 

            generationCounter += 1
            # * Отбираем лучшие особи путем турнира
            offspring = selTournament(population, len(population))
            offspring = list(map(clone, offspring))

            # * Выполняем скрещивание неповторяющихся пар родителей 
            offspring_even, offspring_odd = offspring[::2], offspring[1::2]
            for child1, child2 in zip(offspring_even, offspring_odd):
                # Если меньше вероятности скрещивания, то скрещиваем
                # Если срабатывает условие, то родители становятся потомками, иначе они остаются родителями в популяции
                if random.random() < P_CROSSOVER/LENGTH_D:
                    cxOrdered_v2(child1, child2)

            # * Выполняем мутацию
            for mutant in offspring:
                if random.random() < P_MUTATION/LENGTH_D:
                    mutationShuffleIndexes(mutant, 1.0/LENGTH_CHROM/10)

            # * Обновляем значение приспособленности новой популяции
            freshFitnessValues = list(map(oneDijkstraFitness, offspring))
            for individual, fitnessValue in zip(offspring, freshFitnessValues):
                individual.fitness.values = fitnessValue

            # * Обновляем список популяции
            population[:] = offspring

            # * Обновляем список значений приспособленности каждой особи в популяции
            fitnessValues = [ind.fitness.values for ind in population]

            # * Формируем статистику
            vals.append(fitnessValues)

            minFitness = min(fitnessValues)
            avgFitness = sum(fitnessValues) / len (fitnessValues)
            minFitnessValues.append(minFitness)
            avgFitnessValues.append(avgFitness)
            print(f"Поколение {generationCounter}: Мин. приспособ. = {minFitness}, Средняя присособ. = {avgFitness}")
            self.ui.SystemMassage_TextBrowser.append(f"Поколение {generationCounter}: Мин. приспособ. = {minFitness}, Средняя присособ. = {avgFitness}")

            best_index = fitnessValues.index(minFitness)
            best = population[best_index]
            print(f"Лучший индивидуум = {best}")
            self.ui.SystemMassage_TextBrowser.append(f"Лучший индивидуум = {best}")

            try:
                # Считаем длину текущего маршрута, используя значения смежной матрицы D
                prev = startV
                best_sum_way = 0
                best_way = best[endV]
                full_way = best_way[:best_way.index(endV)+1]
                for next in full_way:
                    edge_values = Window.D[prev]
                    best_sum_way += int(edge_values[next])
                    prev = next

                # Выводим информацию
                print(f"Лучший путь от {startV} до {endV} = {best_way} = {full_way}, приспособленность пути = {best_sum_way}\n")
                self.ui.SystemMassage_TextBrowser.append(f"Лучший путь от {startV} до {endV} = {best_way} = {full_way}, приспособленность пути = {best_sum_way}\n")
            except: 
                print("\n")
                self.ui.SystemMassage_TextBrowser.append("\n") 

        # * Выводим таблицу популяции
        self.Send_TableWidget_GenAlg(population)

        # * Выводим собранную статистику в виде графиков
        plt.plot(minFitnessValues, color='red')
        plt.plot(avgFitnessValues, color='green')
        plt.xlabel('Поколение')
        plt.ylabel('Мин/средняя приспособленность')
        plt.title('Зависимость минимальной и средней приспособленности от поколения')
        plt.show()

        get_value_startV = self.ui.NODE_START_INDEX_IntSpinBox.value() - 1
        get_value_endV = self.ui.NODE_END_INDEX_IntSpinBox.value() - 1
        self.canvas.draw_best_graph(best, startV=get_value_startV, endV=get_value_endV)

        #_draw_graph(startV, 0, 0)
        
        # * Интерактивный вывод статистики приспособленности
        import time
        plt.ion()
        fig, ax = plt.subplots()
        line, = ax.plot(vals[0], ' o', markersize=1)
        for v in vals:
            line.set_ydata(v)
            plt.draw()
            plt.gcf().canvas.flush_events()
            time.sleep(0.4)
        plt.ioff()
        plt.show()
        

''' --------Запуск формы------- '''
if __name__ == '__main__':
    app = QApplication(sys.argv)                                      # Объект приложения (экземпляр QApplication)
    win = Window()                                                    # Создание формы
    # Вход в главный цикл приложения
    sys.exit(app.exec_())                                             # Выход после закрытия приложения