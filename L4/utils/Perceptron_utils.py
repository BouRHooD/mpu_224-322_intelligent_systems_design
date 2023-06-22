import numpy as np
import utils.Gen_alg_utils as Gen_alg_utils                    

class Perceptron:
    """Класс для перцептрона"""
    def __init__ (self, input_shape, output_shape ,ga_parms,epochs =1):
        # Xavier 
        stdv = 1/np.sqrt(input_shape)
        self.W = np.random.uniform(-stdv, stdv, size=(input_shape, output_shape))
        self.b = np.random.uniform(-stdv, stdv, size=output_shape)
        self.epochs = epochs
        self.activate=0
        self.ga = Gen_alg_utils.GA(*ga_parms)
        self.get_csv(self.W, "start_W")
    
    def get_csv(self, array, name):
        import os
        folder_ui_path = '_out' if os.path.isdir('_out') is True else 'L4/_out'
        np.savetxt(f"{folder_ui_path}/{name}.csv", array, delimiter=",")

    #функции активации 
    def ReLU(self, x):
        return np.maximum(0,x)

    def dReLU(self,x):
        return 1 * (x > 0) 

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    # формирование весов  
    def forward(self, input, activate):
        
        self.output = np.dot(input, self.W)
        self.output += self.b 
        
        return self.output   

    '''Обучение'''
    def train(self, X, Y, count_data = 1000, save_itr=True, window=None):
        '''Обучение'''
        # Создаем начальные веса - индивидумов
        self.ga.createGA()
        # Запоминаем веса до обучения
        #if window is not None: window.weights_history.append([self.l1_hidden.Weights, self.l2_output.Weights]);м

        # Проходимся по эпохам обучения
        for epoch in range(self.epochs):
            # ProgressBar 
            if window.ui.pbLearning is not None: window.ui.pbLearning.setValue(0); 
            if window.ui.pbLearning is not None: window.ui.pbLearning.setMinimum(0); 
            if window.ui.pbLearning is not None: window.ui.pbLearning.setMaximum(count_data); 

            for i in range(len(X[:count_data])):
                self.W = self.ga.TrainGA(X[i].flatten(), Y[i])[0]
                if save_itr:
                    self.get_csv(self.W , f"epoch_{epoch}_{i}")
                
                # Отправляем значение в ProgressBar
                if window.ui.pbLearning is not None: window.ui.pbLearning.setValue(i); 
                if window is not None: window.SystemMassage_TextBrowser_append(f"[{epoch+1}/{self.epochs} эпоха] обучение на изображении {i + 1}/{count_data}");
                print(f"[{epoch+1}/{self.epochs} эпоха] обучение на изображении {i + 1}/{count_data}") 
            self.get_csv(self.W , f"epoch_{epoch}")

    # пердсказание
    def pred(self, image):
        x = image.flatten()   
        y_pred = self.forward(x, self.activate)
        return np.argmax(y_pred) 
    
def stat(perceptron_: Perceptron, train_X, train_Y):
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
    for x, y in zip(train_X,train_Y):
        y_true.append(np.argmax(y))
        y_pred.append(perceptron_.pred(x)) 

    import matplotlib.pyplot as plt
    from PIL import Image
    import pandas as pd
    import seaborn as sn
        
    confusion_matrix = confusion_matrix(y_pred,y_true)
    print("Accuracy:",accuracy(y_pred, y_true)*100)
    df_cm = pd.DataFrame(confusion_matrix[0], range(2), range(2))

    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 8},fmt='g',cmap="Greens") # font size
    plt.show()

    graph = perceptron_.ga.meanFitnessValues
    step = perceptron_.ga.max_generation
    plt.grid(False)
    plt.plot(np.arange(len(graph)), graph, linewidth= 2)
    for i in range(16):    
        plt.axvline(x=i*step, color = 'r', linewidth= 0.5)
    plt.show()

def view_weigts(perceptron_:Perceptron):
    import matplotlib.pyplot as plt
    f  = plt.figure(figsize=(20,20))
    for size, i in enumerate(np.array(perceptron_.W).T):
        f.add_subplot(5,5, size+1)
        plt.grid(False)
        plt.imshow(i.reshape((28,28)), cmap='gray')
    plt.show()