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
        self.create_csv(self.W, "weights")
    
    def generate_weight(self, input_size, output_size):
        # Xavier 
        stdv = 1/np.sqrt(input_size)
        self.W = np.random.uniform(-stdv, stdv, size=(input_size, output_size))
        self.b = np.random.uniform(-stdv, stdv, size=output_size)

    def create_csv(self, array, name):
        import os
        folder_ui_path = '_out' if os.path.isdir('_out') is True else 'L2/_out'
        np.savetxt(f"{folder_ui_path}/{name}.csv", array, delimiter=",")

    '''Функции активации'''
    def ReLU(self, x):
        return np.maximum(0,x)

    def dReLU(self,x):
        return 1 * (x > 0) 

    def softmax(self, z):
        z = z - np.max(z, axis = 1).reshape(z.shape[0],1)
        return np.exp(z) / np.sum(np.exp(z)).reshape(z.shape[0],1)

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    '''Формирование весов'''
    def forward(self, input, in_func_activation):
        # Перемножаем матрицу входного слоя на матрицу весов
        self.output = np.dot(input, self.W)
        self.output += self.b 
        self.output = in_func_activation(self.output)
        return self.output   

    '''Обучение'''
    def train(self, in_train_images, in_train_labels, in_func_activate=ReLU, in_count_use_img=1000, in_epochs=1, in_learning_rate=0.001, ui_progress_bar=None, window_ui=None):
        '''Обучение'''
        self.count_train_imgages = len(in_train_images[:in_count_use_img])    # Пользовательское кол-во изображений для обучения
        
        # Проходимся по эпохам обучения
        for epoch in range(in_epochs):
            # ProgressBar 
            if ui_progress_bar is not None: ui_progress_bar.setValue(0); 
            if ui_progress_bar is not None: ui_progress_bar.setMinimum(0); 
            if ui_progress_bar is not None: ui_progress_bar.setMaximum(self.count_train_imgages); 

            # Train
            for i in range(self.count_train_imgages):
                source_image = in_train_images[i]                       # Исходное изображение 28х28
                input_layer = source_image.flatten()                    # Переводим пиксели в одну строку 28х28=784
                y_pred = self.forward(input_layer, in_func_activate)    # 
                error = in_train_labels[i]-y_pred
                input_layer = np.vstack(input_layer)
                D = in_learning_rate*input_layer*error
                self.W = self.W+D

                # Отправляем значение в ProgressBar
                if ui_progress_bar is not None: ui_progress_bar.setValue(i); 
                if window_ui is not None: window_ui.SystemMassage_TextBrowser_append(f"[{epoch+1}/{in_epochs}] "); 

            self.create_csv(self.W , f"epoch_{epoch}")
        
        # После обучения присваиваем 0% в ProgressBar
        if ui_progress_bar is not None: ui_progress_bar.setValue(0); 

    '''Распознование'''
    def recognition(self, image, in_func_activation):
        '''Распознование'''
        x = image.flatten()   
        y_pred = self.forward(x, in_func_activation)
        return np.argmax(y_pred)    
    
'''Статистика обучения'''
def stat(perceptron_:Perceptron, in_test_images, in_test_labels, in_func_activation):
    '''Статистика обучения'''
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
    print("Accuracy:",accuracy(y_pred, y_true)*100)
    df_cm = pd.DataFrame(confusion_matrix[0], range(10), range(10))

    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 8},fmt='g',cmap="Greens") # font size
    plt.show()

'''Отобразить веса'''
def view_weigts(perceptron_:Perceptron):
    f  = plt.figure(figsize=(20,20))
    for size, i in enumerate(perceptron_.W.T):
        f.add_subplot(5,5, size+1)
        plt.grid(False)
        plt.imshow(i.reshape((28,28)), cmap='gray')
    plt.show()