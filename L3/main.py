# Распознавание изображений на базе НС обратного распространения Leonov Vladislav 181-311
# ФИО автора: Леонов Владислав Денисович
# E-mail автора: bourhood@gmail.com
# Группа: 224-322
# Университет: Московский Политехнический Университет
# Год разработки: 11.05.2023
# Учебный курс: https://online.mospolytech.ru/course/view.php?id=10055
########################################################################################################################

# Загружаем библиотеки
import os
import sys                                                            # Предоставляет системе особые параметры и функции
import datetime                                                       # Для вывода текущего времени
import traceback                                                      # Для лучшего вывода ошибок в консоль
import numpy as np                                                    # Для работы с массивами
import utils.DataSet_utils as DataSet_utils                           # Для работы с датасетом к файлам (self)
import utils.OpenFiles_utils as OpenFiles_utils                       # Для выбора пути к файлам (self)
import utils.Perceptron_utils as Perceptron_utils                     # Для работы с перцептроном к файлам (self)

# Библиотеки GUI интерфейса (pip install PyQt5) (Qt Designer to edit *.ui - https://build-system.fman.io/qt-designer-download)
from PyQt5 import uic
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

''' -------- Главная форма ------- '''   
class Window(QMainWindow):       
    def __init__(self, *args, **kwargs):
        super(Window, self).__init__(*args, **kwargs)

        import random
        self.formOpening()                                            # Настройки и запуск формы
        random.seed(55)                                               # Присваиваем зерно для рандома
        
        # Подписки на события
        self.ui.bLoadImgGZ.clicked.connect(self.bLoadImgGZ_clicked)
        self.ui.bLoadLabelsGZ.clicked.connect(self.bLoadLabelsGZ_clicked)
        self.ui.bLoadImgGZ_Recognition.clicked.connect(self.bLoadImgGZ_Recognition_clicked)
        self.ui.bLoadLabelsGZ_Recognition.clicked.connect(self.bLoadLabelsGZ_Recognition_clicked)
        self.ui.bPerceptronLearn.clicked.connect(self.bPerceptronLearn_clicked)
        self.ui.bPerceptron_Recognition.clicked.connect(self.bPerceptron_Recognition_clicked)
        self.ui.bPerceptron_GetStats.clicked.connect(self.bPerceptron_GetStats_clicked)
        self.ui.b_Perceptron_Learn_ShowWeights.clicked.connect(self.b_Perceptron_Learn_ShowWeights_clicked)

        self.ui.cbListLoadedImages.currentTextChanged.connect(self.cbListLoadedImages_TextChanged) 
        self.ui.cbListLoadedImages_Recognition.currentTextChanged.connect(self.cbListLoadedImages_Recognition_TextChanged) 
        self.ui.cb_Perceptron_Learn.currentTextChanged.connect(self.cb_Perceptron_Learn_TextChanged) 
        self.ui.cb_Perceptron_Learn_2.currentTextChanged.connect(self.cb_Perceptron_Learn_TextChanged) 

        # Загружаем изображения и разметку
        self.LoadImgGZ("dataset/train-images-idx3-ubyte.gz" if os.path.isfile('dataset/train-images-idx3-ubyte.gz') is True else 'L3/dataset/train-images-idx3-ubyte.gz', exception_off=True)
        self.LoadLabelsGZ("dataset/train-labels-idx1-ubyte.gz" if os.path.isfile('dataset/train-labels-idx1-ubyte.gz') is True else 'L3/dataset/train-labels-idx1-ubyte.gz', exception_off=True)
        self.LoadImgGZ_Recognition("dataset/t10k-images-idx3-ubyte.gz" if os.path.isfile('dataset/t10k-images-idx3-ubyte.gz') is True else 'L3/dataset/t10k-images-idx3-ubyte.gz', exception_off=True)
        self.LoadLabelsGZ_Recognition("dataset/t10k-labels-idx1-ubyte.gz" if os.path.isfile('dataset/t10k-labels-idx1-ubyte.gz') is True else 'L3/dataset/t10k-labels-idx1-ubyte.gz', exception_off=True)

        self.ui.SystemMassage_TextBrowser.setText(f'{datetime.datetime.now()}: Приложение готово к работе')

    def formOpening(self):
        # Настройки окна главной формы
        file_ui_path = 'GUI.ui' if os.path.isfile('GUI.ui') is True else 'L3/GUI.ui'
        file_icon_path = 'surflay.ico' if os.path.isfile('surflay.ico') is True else 'L3/surflay.ico'
        self.ui = uic.loadUi(file_ui_path)                            # GUI, должен быть в папке с main.py
        self.ui.setWindowTitle('Леонов Владислав 224-322 - Лаб. 3')   # Название главного окна
        self.ui.setWindowIcon(QIcon(file_icon_path))                  # Иконка на гланое окно
        self.ui.show()                                                # Открываем окно формы  

    def SystemMassage_TextBrowser_append(self, text):
        self.ui.SystemMassage_TextBrowser.append(f"{datetime.datetime.now()}: {text}")

    def bLoadImgGZ_clicked(self):
        try:
            folder_path = "dataset" if os.path.isdir('dataset') is True else 'L3/dataset'
            selected_files = OpenFiles_utils.getOpenFilesAndDirs(directory=folder_path)
            if (selected_files is None or len(selected_files) <= 0): return; 

            selected_file_path = selected_files[0]
            self.LoadImgGZ(selected_file_path)
        except Exception as ex:
            print(ex); traceback.print_exc(); 
    
    def bLoadImgGZ_Recognition_clicked(self):
        try:
            folder_path = "dataset" if os.path.isdir('dataset') is True else 'L3/dataset'
            selected_files = OpenFiles_utils.getOpenFilesAndDirs(directory=folder_path)
            if (selected_files is None or len(selected_files) <= 0): return; 

            selected_file_path = selected_files[0]
            self.LoadImgGZ_Recognition(selected_file_path)
        except Exception as ex:
            print(ex); traceback.print_exc(); 

    def LoadImgGZ(self, in_path, exception_off=False):
        try:
            if (in_path is None or len(in_path) <= 0): return; 

            selected_file_path = in_path
            self.SystemMassage_TextBrowser_append(f'Загружаются изображения по пути {selected_file_path}')
            
            '''Получаем изображения из архива'''
            imgs, img_count = DataSet_utils.training_images(selected_file_path)
            self.train_images = DataSet_utils.transformation_zero_one(imgs[:])
            self.SystemMassage_TextBrowser_append(f'Загружено изображений {img_count} шт.')
            self.ui.labelCountLoadedImg.setText(f'Всего изображений: {img_count} шт.')
            
            '''Заполняем всплывающее окно'''
            list_images_name = [f"{index}" for index, img in enumerate(imgs)]
            self.ui.cbListLoadedImages.clear()
            self.ui.cbListLoadedImages.addItems(list_images_name)

            '''Активируем активность других виджетов'''
            self.ui.bLoadLabelsGZ.setEnabled(True)
        except Exception as ex:
            if exception_off is False: print(ex); traceback.print_exc(); 

    def LoadImgGZ_Recognition(self, in_path, exception_off=False):
        try:
            if (in_path is None or len(in_path) <= 0): return; 

            selected_file_path = in_path
            self.SystemMassage_TextBrowser_append(f'Загружаются изображения тестовой выборки по пути {selected_file_path}')
            
            '''Получаем изображения из архива'''
            imgs, img_count = DataSet_utils.training_images(selected_file_path)
            self.test_images = DataSet_utils.transformation_zero_one(imgs[:])
            self.SystemMassage_TextBrowser_append(f'Загружено изображений тестовой выборки {img_count} шт.')
            self.ui.labelCountLoadedImg_Recognition.setText(f'Всего изображений: {img_count} шт.')
            
            '''Заполняем всплывающее окно'''
            list_images_name = [f"{index}" for index, img in enumerate(imgs)]
            self.ui.cbListLoadedImages_Recognition.clear()
            self.ui.cbListLoadedImages_Recognition.addItems(list_images_name)

            '''Активируем активность других виджетов'''
            self.ui.bLoadLabelsGZ_Recognition.setEnabled(True)
        except Exception as ex:
            if exception_off is False: print(ex); traceback.print_exc(); 

    def bLoadLabelsGZ_clicked(self):
        try:
            folder_path = "dataset" if os.path.isdir('dataset') is True else 'L3/dataset'
            selected_files = OpenFiles_utils.getOpenFilesAndDirs(directory=folder_path)
            if (selected_files is None or len(selected_files) <= 0): return; 

            selected_file_path = selected_files[0]
            self.LoadLabelsGZ(selected_file_path)
        except Exception as ex:
            print(ex); traceback.print_exc(); 
    
    def bLoadLabelsGZ_Recognition_clicked(self):
        try:
            folder_path = "dataset" if os.path.isdir('dataset') is True else 'L3/dataset'
            selected_files = OpenFiles_utils.getOpenFilesAndDirs(directory=folder_path)
            if (selected_files is None or len(selected_files) <= 0): return; 

            selected_file_path = selected_files[0]
            self.LoadLabelsGZ_Recognition(selected_file_path)
        except Exception as ex:
            print(ex); traceback.print_exc(); 

    def LoadLabelsGZ(self, in_path, exception_off=False):
        try:
            selected_file_path = in_path
            self.SystemMassage_TextBrowser_append(f'Загружаются разметки по пути {selected_file_path}')
            
            '''Получаем разметки из архива'''
            labels, labels_count = DataSet_utils.training_labels(selected_file_path)
            self.train_labels = DataSet_utils.transformation_one_hot_encoding(labels[:])
            self.SystemMassage_TextBrowser_append(f'Загружено разметок {labels_count} шт.')
            self.ui.labelCountLoadedLabels.setText(f'Всего разметок: {labels_count} шт.')
            
            '''Заполняем всплывающее окно'''
            exist_list_names_img = [self.ui.cbListLoadedImages.itemText(i) for i in range(self.ui.cbListLoadedImages.count())]
            list_images_name = [f"img_{exist_label}-label_{labels[index]}" for index, exist_label in enumerate(exist_list_names_img)]
            self.ui.cbListLoadedImages.clear()
            self.ui.cbListLoadedImages.addItems(list_images_name)

            '''Активируем активность других виджетов'''
            self.ui.bPerceptronLearn.setEnabled(True)
        except Exception as ex:
            if exception_off is False: print(ex); traceback.print_exc(); 
    
    def LoadLabelsGZ_Recognition(self, in_path, exception_off=False):
        try:
            selected_file_path = in_path
            self.SystemMassage_TextBrowser_append(f'Загружаются разметки тестовой выборки по пути {selected_file_path}')
            
            '''Получаем разметки из архива'''
            labels, labels_count = DataSet_utils.training_labels(selected_file_path)
            self.test_labels = DataSet_utils.transformation_one_hot_encoding(labels[:])
            self.SystemMassage_TextBrowser_append(f'Загружено разметок тестовой выборки {labels_count} шт.')
            self.ui.labelCountLoadedLabels_Recognition.setText(f'Всего разметок: {labels_count} шт.')
            
            '''Заполняем всплывающее окно'''
            exist_list_names_img = [self.ui.cbListLoadedImages_Recognition.itemText(i) for i in range(self.ui.cbListLoadedImages_Recognition.count())]
            list_images_name = [f"img_{exist_label}-label_{labels[index]}" for index, exist_label in enumerate(exist_list_names_img)]
            self.ui.cbListLoadedImages_Recognition.clear()
            self.ui.cbListLoadedImages_Recognition.addItems(list_images_name)

            '''Активируем активность других виджетов'''
            #self.ui.bLoadWeightsCSV_Recognition.setEnabled(True)
        except Exception as ex:
            if exception_off is False: print(ex); traceback.print_exc(); 

    def bPerceptronLearn_clicked(self):
        try:
            # Получаем данные из формы приложения
            get_epoch_count = self.ui.sbEpochCount.value()
            get_learning_rate = self.ui.sbLearningRate.value()
            get_count_image_in_train_list = self.ui.sbImageInTrainCount.value()

            # Создание модели перцептрона (input_layer(784)->hiden_1(50)->hiden_2(50)->output_layer(10))
            self.perceptron = Perceptron_utils.Perceptron()
            
            # Обучение модели перцептрона
            self.weights_history = []
            self.perceptron.train(in_train_images=self.train_images, in_train_labels=self.train_labels, in_count_use_img=get_count_image_in_train_list, in_epochs=get_epoch_count, in_learning_rate=get_learning_rate, ui_progress_bar=self.ui.pbLearning, window=self, in_test_images=self.test_images, in_test_labels=self.test_labels)
            
            # Заполняем cb эпох
            list_names_history = []
            for i, x in enumerate(self.weights_history):
                list_names_history.append(f"Эпоха {i}")    
            self.ui.cb_Perceptron_Learn.clear()
            self.ui.cb_Perceptron_Learn.addItems(list_names_history)

            # Заполняем cb слоев в эпохах
            if self.weights_history is not None and len(self.weights_history) > 0:
                list_names_history = []
                for i, x in enumerate(self.weights_history[0]):
                    list_names_history.append(f"Слой сети {i}")    
                self.ui.cb_Perceptron_Learn_2.clear()
                self.ui.cb_Perceptron_Learn_2.addItems(list_names_history)

            '''Активируем активность других виджетов'''
            self.ui.bPerceptron_Recognition.setEnabled(True)
            self.ui.bPerceptron_GetStats.setEnabled(True)
        except Exception as ex:
            print(ex); traceback.print_exc(); 

    def Send_tw_Perceptron_Learn_waights(self, widget, in_weights, is_need_clear=False):
        if in_weights is None or len(in_weights) < 2: return; 

        if is_need_clear: widget.clear(); 

        count_row = len(in_weights)
        count_column = len(in_weights[0])

        widget.setRowCount(count_row)
        widget.setColumnCount(count_column)

        for j in range(count_column):
            widget.setHorizontalHeaderItem(j, QTableWidgetItem(f"{j}"))

        for i in range(count_row):
            for j in range(count_column):
                value = in_weights[i][j]
                widget.setItem(i, j, QTableWidgetItem(str(round(value, 10))))

        #widget.resizeColumnsToContents()
        #widget.resizeRowsToContents()
        
    def bPerceptron_Recognition_clicked(self):
        try:
            get_value_text = self.ui.cbListLoadedImages_Recognition.currentText()
            index_value = int(get_value_text) if self.intTryParse(get_value_text)[1] else self.intTryParse(get_value_text.replace("img_", "").replace("label_", "").split("-")[0])[0]
            if isinstance(index_value, int) is False: return; 
            
            image = self.test_images[index_value][:]
            result_recognition = self.perceptron.recognition(image)
            self.ui.labelResult_Recognition.setText(f'<b>Результат распознования - {result_recognition}</b>')
        except Exception as ex:
            print(ex); traceback.print_exc(); 
    
    def bPerceptron_GetStats_clicked(self):
        try:
            text = Perceptron_utils.stat(self.perceptron, self.test_images, self.test_labels)
            self.SystemMassage_TextBrowser_append(text)
        except Exception as ex:
            print(ex); traceback.print_exc(); 

    def b_Perceptron_Learn_ShowWeights_clicked(self):
        try:
            if self.weights_history is None: return; 
        
            get_value_text = self.ui.cb_Perceptron_Learn.currentText()
            index_value = int(get_value_text) if self.intTryParse(get_value_text)[1] else self.intTryParse(get_value_text.replace("Эпоха ", ""))[0]
            if isinstance(index_value, int) is False: return; 
        
            get_value_text = self.ui.cb_Perceptron_Learn_2.currentText()
            index_value_2 = int(get_value_text) if self.intTryParse(get_value_text)[1] else self.intTryParse(get_value_text.replace("Слой сети ", ""))[0]
            if isinstance(index_value_2, int) is False: return;  
            
            Perceptron_utils.view_weigts(self.weights_history[index_value][index_value_2])
        except Exception as ex:
            print(ex); traceback.print_exc(); 

    def cbListLoadedImages_TextChanged(self):
        '''https://stackoverflow.com/questions/40427435/extract-images-from-idx3-ubyte-file-or-gzip-via-python'''
        '''Выводим изображение'''  
        try:
            from PyQt5 import QtGui
            import matplotlib.pyplot as plt
            get_value_text = self.ui.cbListLoadedImages.currentText()
            index_value = int(get_value_text) if self.intTryParse(get_value_text)[1] else self.intTryParse(get_value_text.replace("img_", "").replace("label_", "").split("-")[0])[0]
            if isinstance(index_value, int) is False: return; 
            image = (self.train_images[index_value]*255).astype(np.uint8)
            image = np.asarray(image).squeeze()
            self.qimage = QImage(image,image.shape[1],image.shape[0], QImage.Format_Grayscale8)
            self.ui.labelSelectedImg.setPixmap(QtGui.QPixmap.fromImage(self.qimage))
            #plt.axis('off')
            #plt.imshow(image)
            #plt.show()
        except Exception as ex:
            print(ex); traceback.print_exc(); 

    def cb_Perceptron_Learn_TextChanged(self):
        try:
            get_value_text = self.ui.cb_Perceptron_Learn.currentText()
            index_value = int(get_value_text) if self.intTryParse(get_value_text)[1] else self.intTryParse(get_value_text.replace("Эпоха ", ""))[0]
            if isinstance(index_value, int) is False: return;

            get_value_text = self.ui.cb_Perceptron_Learn_2.currentText()
            index_value_2 = int(get_value_text) if self.intTryParse(get_value_text)[1] else self.intTryParse(get_value_text.replace("Слой сети ", ""))[0]
            if isinstance(index_value_2, int) is False: return;  
            
            self.Send_tw_Perceptron_Learn_waights(self.ui.tw_Perceptron_Learn, self.weights_history[index_value][index_value_2], is_need_clear=True)
        except Exception as ex:
            print(ex); traceback.print_exc(); 

    def cbListLoadedImages_Recognition_TextChanged(self):
        '''https://stackoverflow.com/questions/40427435/extract-images-from-idx3-ubyte-file-or-gzip-via-python'''
        '''Выводим изображение'''  
        try:
            from PyQt5 import QtGui
            import matplotlib.pyplot as plt
            get_value_text = self.ui.cbListLoadedImages_Recognition.currentText()
            index_value = int(get_value_text) if self.intTryParse(get_value_text)[1] else self.intTryParse(get_value_text.replace("img_", "").replace("label_", "").split("-")[0])[0]
            if isinstance(index_value, int) is False: return; 
            image = (self.test_images[index_value]*255).astype(np.uint8)
            image = np.asarray(image).squeeze()
            self.qimage = QImage(image,image.shape[1],image.shape[0], QImage.Format_Grayscale8)
            self.ui.labelSelectedImg_Recognition.setPixmap(QtGui.QPixmap.fromImage(self.qimage))
            #plt.axis('off')
            #plt.imshow(image)
            #plt.show()
        except Exception as ex:
            print(ex); traceback.print_exc(); 

    def intTryParse(self, value):
        '''Метод для попытки перевездит значение в число'''
        try:
            return int(value), True
        except ValueError:
            return value, False

''' --------Запуск формы------- '''
if __name__ == '__main__':                                            # Выполнение условия, если запущен этот файл python, а не если он подгружен через import
    app = QApplication(sys.argv)                                      # Объект приложения (экземпляр QApplication)
    win = Window()                                                    # Создание формы
    sys.exit(app.exec_())                                             # Вход в главный цикл приложения и Выход после закрытия приложения