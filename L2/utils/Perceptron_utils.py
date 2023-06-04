# Распознавание изображений с помощью персептрона Leonov Vladislav 181-311
# ФИО автора: Леонов Владислав Денисович
# E-mail автора: bourhood@gmail.com
# Группа: 224-322
# Университет: Московский Политехнический Университет
# Год разработки: 01.05.2023
# Учебный курс: https://online.mospolytech.ru/course/view.php?id=10055
########################################################################################################################

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

class Perceptron:
    '''Класс перцептрона'''
    def __init__(self, input_size, output_size):
        self.generate_weight(input_size, output_size)
        self.create_csv(self.Weights, "weights")
    
    def generate_weight(self, input_size, output_size):
        # Xavier 
        #stdv = 1/np.sqrt(input_size)
        #self.Weights = np.random.uniform(-stdv, stdv, size=(input_size, output_size))
        #self.bias = np.random.uniform(-stdv, stdv, size=output_size)
        self.Weights = np.random.uniform(-0.15, 0.15, size=(input_size, output_size))
        self.bias = np.random.uniform(-0.15, 0.15, size=output_size)

    def create_csv(self, array, name):
        import os
        folder_ui_path = '_out' if os.path.isdir('_out') is True else 'L2/_out'
        np.savetxt(f"{folder_ui_path}/{name}.csv", array, delimiter=",")

    '''Функции активации'''
    def ReLU(self, x):
        return np.maximum(0,x)

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    '''Формирование весов'''
    def out_neurons(self, input, in_func_activation):
        '''Выходные значения нейронов (вектор выходных значений)'''
        # Перемножаем матрицу входного слоя на матрицу весов (∑weight*x) (dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m]))
        self.output = np.dot(input, self.Weights)
        # Прибавляем к матрице смещения
        self.output += self.bias 
        # Используем функцию активации
        self.output = in_func_activation(self.output)
        return self.output   

    '''Обучение'''
    def train(self, in_train_images, in_train_labels, in_func_activate=ReLU, in_count_use_img=1000, in_epochs=1, in_learning_rate=0.001, ui_progress_bar=None, window=None):
        '''Обучение'''
        self.count_train_imgages = len(in_train_images[:in_count_use_img])    # Пользовательское кол-во изображений для обучения
        
        # Запоминаем веса до обучения
        if window is not None: window.weights_history.append(self.Weights); 

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
                y_pred = self.out_neurons(input_layer, in_func_activate)    # Активируем нейрон (вектор выходных значений) (𝐲_pred_i)
                y_result = np.argmax(y_pred)                                # Ответ перцептрона (макс. вероятность)

                error = source_label-y_pred                                 # Вычисляем ошибку (вектор ошибок) (𝐞rror_i = 𝐲_𝐢 − 𝐲_pred_i)
                input_layer = np.vstack(input_layer)                        # Добавляем размероность, чтобы перемножить матрицы
                𝐃_𝐢 = in_learning_rate*input_layer*error                    # Вычисляем матрицу значений корректировок весов (𝐃_𝐢 = 𝛼*(𝐱_𝐢)^𝑇*𝐞rror_i) (величина корректировки веса)
                self.Weights = self.Weights+𝐃_𝐢                             # Обновляем значения весов (𝐖(𝐧𝐞𝐰) = 𝐖(𝐨𝐥𝐝) + 𝐃_𝐢)

                # Отправляем значение в ProgressBar
                if ui_progress_bar is not None: ui_progress_bar.setValue(i); 
                if window is not None: window.SystemMassage_TextBrowser_append(f"[{epoch+1}/{in_epochs} эпоха] обучение на изображении {i + 1}/{self.count_train_imgages}, результат {y_result}, ожидалось {y_true}"); 

            # Сохраняем веса после обучения по каждой эпохе
            if window is not None: window.weights_history.append(self.Weights); 
            self.create_csv(self.Weights , f"epoch_{epoch}")
        
        # После обучения присваиваем 0% в ProgressBar
        if ui_progress_bar is not None: ui_progress_bar.setValue(0); 

    '''Распознование'''
    def recognition(self, image, in_func_activation):
        '''Распознование'''
        x = image.flatten()                               # Выстраиваем пиксели в один ряд 28х28=784
        y_pred = self.out_neurons(x, in_func_activation)  # Вычисляем выходные значения нейронов (func_activate(X*Weights) 
        return np.argmax(y_pred)                          # Находим максимальное значение (макс. вероятность)
    
'''Статистика обучения'''
def stat(perceptron_:Perceptron, in_test_images, in_test_labels, in_func_activation):
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
        y_pred.append(perceptron_.recognition(x, in_func_activation)) 
        
    confusion_matrix = confusion_matrix(y_pred,y_true)
    perceptron_accuracy = round(accuracy(y_pred, y_true)*100,5)
    print(f"Accuracy perceptron: {perceptron_accuracy}%")
    df_cm = pd.DataFrame(confusion_matrix[0], range(10), range(10))

    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 8},fmt='g',cmap="Blues") # font size
    plt.show()

'''Отобразить веса'''
def view_weigts(in_weights):
    f, ax = plt.subplots(2, 5, figsize=(12,5), gridspec_kw={'wspace':0.05, 'hspace':0.2}, squeeze=True)    
    for index, img in enumerate(in_weights.T):
        index_cell = index % 5
        index_row = 0 if index < 5 else 1
        ax[index_row, index_cell].axis("off")
        ax[index_row, index_cell].imshow(img.reshape((28,28)), cmap='gray')
        ax[index_row, index_cell].set_title(str(index))
    
    '''
    f  = plt.figure(figsize=(20,20))
    for size, i in enumerate(in_weights.T):
        f.add_subplot(5,5, size+1)
        plt.grid(False)
        plt.axis('off')
        plt.imshow(i.reshape((28,28)), cmap='gray')
    '''
    plt.show()