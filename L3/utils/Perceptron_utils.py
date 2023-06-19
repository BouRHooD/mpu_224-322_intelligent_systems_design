# Распознавание изображений на базе НС обратного распространения Leonov Vladislav 181-311
# ФИО автора: Леонов Владислав Денисович
# E-mail автора: bourhood@gmail.com
# Группа: 224-322
# Университет: Московский Политехнический Университет
# Год разработки: 11.05.2023
# Учебный курс: https://online.mospolytech.ru/course/view.php?id=10055
########################################################################################################################

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

'''Функции активации''' # Softmax
class Sigmoid:
    output = []
    def activate_neurons(self, input):
        self.output = 1 / (1 + np.exp(-input))
        return self.output

    def backward(self, error):
        # Производная сигмоиды * error
        return self.output * (1 - self.output) * error

class ReLu:
    '''ReLu - Rectified Linear Unit (Выпрямленные линейные единицы)'''
    input = []
    output = []
    def activate_neurons(self, input_sum_neurons):
        self.input = input_sum_neurons
        return np.maximum(0, input_sum_neurons)

    def backward(self, error):
        # Производная от функции активации
        error[self.input < 0] = 0
        return error

'''Класс слоя'''
class LayerPerceptron:
    '''Класс слоя перцептрона'''
    def __init__(self, input_size, output_size, input_name_layer, in_func_activation):
        self.input = []                                 # Входные значения в слой сети
        self.output = []                                # Выходные значения из слоя сети

        self.used_func_activation = in_func_activation  # Используемая функция активации
        self.name_layer = input_name_layer              # Название слоя
        self.generate_weight(input_size, output_size)   # Генерируем начальные веса для слоя

    def generate_weight(self, input_size, output_size):
        self.Weights = np.random.uniform(-0.15, 0.15, size=(input_size, output_size))
        self.bias = np.random.uniform(-0.15, 0.15, size=output_size)
    
    '''Формирование весов'''
    def out_neurons(self, input):
        '''Выходные значения нейронов (вектор выходных значений)'''
        self.input = input
        # Перемножаем матрицу входного слоя на матрицу весов (∑weight*x) (dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])) + bias
        self.output = np.dot(input, self.Weights) + self.bias
        # Используем функцию активации
        self.output = self.used_func_activation.activate_neurons(self.output)
        return self.output
    
    def create_csv(self, count):
        import os
        folder_ui_path = '_out' if os.path.isdir('_out') is True else 'L3/_out'
        np.savetxt(f"{folder_ui_path}/{self.name_layer}_weights_{count}.csv", self.Weights, delimiter=",")
    
'''Класс перцептрона'''
class Perceptron:
    '''Класс перцептрона'''
    def __init__(self):
        # Создание модели перцептрона (input_layer(784)->hidden(50)->l2_output(10))
        self.l1_hidden = LayerPerceptron(784, 50, 'l1_hidden', ReLu())
        self.l2_output = LayerPerceptron(50, 10, 'l2_output', Sigmoid())
        
    def create_csv_weights_perceptron(self, count):
        self.l1_hidden.create_csv(count=count)
        self.l2_output.create_csv(count=count)

    '''Формирование весов'''
    def out_perceptron(self, input):
        input = input.reshape((1, -1))
        output = self.l1_hidden.out_neurons(input)    # func_activate(sum(X*Weights) + bias)
        output = self.l2_output.out_neurons(output)   # func_activate(sum(X*Weights) + bias) 
        return output

    '''Обучение'''
    def train(self, in_train_images, in_train_labels, in_count_use_img=1000, in_epochs=1, in_learning_rate=0.001, ui_progress_bar=None, window=None):
        '''Обучение'''
        self.count_train_imgages = len(in_train_images[:in_count_use_img])    # Пользовательское кол-во изображений для обучения
        
        # Запоминаем веса до обучения
        if window is not None: window.weights_history.append([self.l1_hidden.Weights, self.l2_output.Weights]); 

        # Проходимся по эпохам обучения
        for epoch in range(in_epochs):
            # ProgressBar 
            if ui_progress_bar is not None: ui_progress_bar.setValue(0); 
            if ui_progress_bar is not None: ui_progress_bar.setMinimum(0); 
            if ui_progress_bar is not None: ui_progress_bar.setMaximum(self.count_train_imgages); 

            # Train
            for i in range(self.count_train_imgages):
                source_image = in_train_images[i]                           # Исходное изображение 28х28
                source_label = in_train_labels[i]                           # Разметка в унитарном коде (𝐲_𝐢)
                y_true = np.argmax(source_label)                            # Ожидаемый ответ сети
                
                input_layer = source_image.flatten()                        # Переводим пиксели в одну строку 28х28=784 (𝐱_𝐢)
                y_pred = self.out_perceptron(input_layer)                   # Активируем все слои нейронов (вектор выходных значений) (𝐲_pred_i)
                y_result = np.argmax(y_pred)                                # Ответ перцептрона (макс. вероятность)

                ### **Алгоритм обратного распространения ошибки (англ. backpropagation)**
                ## https://machinelearningmastery.ru/implement-backpropagation-algorithm-scratch-python/
                ## https://habr.com/ru/articles/271563/
                ## Вычисляем ошибки для всех слоев персептрона
                ## Стохостический градиентный спуск
                error_2 = y_pred - source_label                                         # Вычисляем ошибку (вектор ошибок) (𝐞rror_i = 𝐲_𝐢 − 𝐲_pred_i)
                error_2_delta = self.l2_output.used_func_activation.backward(error_2)   # Дельта ошибки
                # Дельта на выходном слои 1 
                error_1 = np.dot(error_2_delta, self.l2_output.Weights.T)               # Передаем ошибку на нижний слой
                error_1_delta = self.l1_hidden.used_func_activation.backward(error_1)   # Дельта ошибки

                # Меняем веса по найденным ошибкам
                self.l1_hidden.Weights -= error_1_delta*self.l1_hidden.input.T*in_learning_rate
                self.l2_output.Weights -= error_2_delta*self.l2_output.input.T*in_learning_rate

                # Отправляем значение в ProgressBar
                if ui_progress_bar is not None: ui_progress_bar.setValue(i); 
                if window is not None: window.SystemMassage_TextBrowser_append(f"[{epoch+1}/{in_epochs} эпоха] обучение на изображении {i + 1}/{self.count_train_imgages}, результат {y_result}, ожидалось {y_true}"); 
            
            # Сохраняем веса после обучения по каждой эпохе
            if window is not None: window.weights_history.append([self.l1_hidden.Weights, self.l2_output.Weights]); 
            self.create_csv_weights_perceptron(count=epoch)
        
        # После обучения присваиваем 0% в ProgressBar
        if ui_progress_bar is not None: ui_progress_bar.setValue(0); 

    '''Распознование'''
    def recognition(self, image):
        '''Распознование'''
        x = image.flatten()                               # Выстраиваем пиксели в один ряд 28х28=784
        y_pred = self.out_perceptron(x)                   # Активируем все слои нейронов (вектор выходных значений) (𝐲_pred_i)
        return np.argmax(y_pred)                          # Находим максимальное значение (макс. вероятность)
    
'''Статистика обучения'''
def stat(perceptron_:Perceptron, in_test_images, in_test_labels):
    '''Статистика обучения - Строим матрицу ошибок'''
    def confusion_matrix(y_pred, y_true):
        classes = np.unique(y_true)
        classes.sort()
        conf_m = np.zeros(shape=(len(classes), len(classes)))
        for i in classes:
            for j in classes:
                conf_m[i, j] = np.logical_and((y_pred==i), (y_true==j)).sum()
        return conf_m, classes
    
    def accuracy(y_pred, y_true):
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        return (y_pred==y_true).sum()/y_true.size
    
    y_true = []
    y_pred = []
    for x, y in zip(in_test_images, in_test_labels):
        y_true.append(np.argmax(y))
        y_pred.append(perceptron_.recognition(x)) 
        
    confusion_matrix = confusion_matrix(y_pred,y_true)
    perceptron_accuracy = round(accuracy(y_pred, y_true)*100,5)
    print(f"Accuracy perceptron: {perceptron_accuracy}%")
    df_cm = pd.DataFrame(confusion_matrix[0], range(10), range(10))

    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 8},fmt='g',cmap="Blues") # font size
    plt.show()
    return f"Accuracy perceptron: {perceptron_accuracy}%"

'''Отобразить веса'''
def view_weigts(in_weights):
    weights_T = in_weights.T
    f, ax = plt.subplots(2, 5, figsize=(12,5), gridspec_kw={'wspace':0.05, 'hspace':0.2}, squeeze=True)    
    for index, img in enumerate(weights_T):
        index_cell = index % 5
        index_row = 0 if index < 5 else 1
        ax[index_row, index_cell].axis("off")
        if img.shape[0] == 784: ax[index_row, index_cell].imshow(img.reshape((28,28)), cmap='gray'); 
        else: ax[index_row, index_cell].imshow(img.reshape((5,10)), cmap='gray'); 
        ax[index_row, index_cell].set_title(str(index))
    plt.show()