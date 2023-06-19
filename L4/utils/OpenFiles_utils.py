# Распознавание изображений на базе НС обратного распространения Leonov Vladislav 181-311
# ФИО автора: Леонов Владислав Денисович
# E-mail автора: bourhood@gmail.com
# Группа: 224-322
# Университет: Московский Политехнический Университет
# Год разработки: 11.05.2023
# Учебный курс: https://online.mospolytech.ru/course/view.php?id=10055
########################################################################################################################

from PyQt5 import QtCore, QtGui, QtWidgets

def getOpenFilesAndDirs(parent=None, caption='', directory='', 
                        filter='', initialFilter='', options=None):
    '''https://ru.stackoverflow.com/questions/1114200/Выбор-файла-и-или-папки-qfiledialog'''
    def updateText():
        # обновить содержимое виджета редактирования строки выбранными файлами
        selected = []
        for index in view.selectionModel().selectedRows():
            selected.append('"{}"'.format(index.data()))
        lineEdit.setText(' '.join(selected))

    dialog = QtWidgets.QFileDialog(parent, windowTitle=caption)
    dialog.setFileMode(dialog.ExistingFiles)
    if options:
        dialog.setOptions(options)
    dialog.setOption(dialog.DontUseNativeDialog, True)            # !!!
    if directory:
        dialog.setDirectory(directory)
    if filter:
        dialog.setNameFilter(filter)
        if initialFilter:
            dialog.selectNameFilter(initialFilter)

    # по умолчанию, если каталог открыт в режиме списка файлов, 
    # QFileDialog.accept() показывает содержимое этого каталога, 
    # но нам нужно иметь возможность "открывать" и каталоги, как мы можем делать с файлами, 
    # поэтому мы просто переопределяем `accept()` с реализацией QDialog по умолчанию, 
    # которая просто вернет `dialog.selectedFiles()`

    dialog.accept = lambda: QtWidgets.QDialog.accept(dialog)

    # в неродном диалоге есть много представлений элементов, 
    # но те, которые отображают фактическое содержимое, создаются внутри QStackedWidget; 
    # это QTreeView и QListView, и дерево используется только тогда, 
    # когда viewMode установлен на QFileDialog.Details, что не в этом случае.
    
    stackedWidget = dialog.findChild(QtWidgets.QStackedWidget)
    view = stackedWidget.findChild(QtWidgets.QListView)
    view.selectionModel().selectionChanged.connect(updateText)

    lineEdit = dialog.findChild(QtWidgets.QLineEdit)
    # очищаем содержимое строки редактирования всякий раз, когда изменяется текущий каталог
    dialog.directoryEntered.connect(lambda: lineEdit.setText(''))

    dialog.exec_()
    return dialog.selectedFiles()