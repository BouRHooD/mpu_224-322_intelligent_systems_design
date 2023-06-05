# Распознавание изображений на базе НС обратного распространения Leonov Vladislav 181-311
# ФИО автора: Леонов Владислав Денисович
# E-mail автора: bourhood@gmail.com
# Группа: 224-322
# Университет: Московский Политехнический Университет
# Год разработки: 11.05.2023
# Учебный курс: https://online.mospolytech.ru/course/view.php?id=10055
########################################################################################################################

def training_images(in_path):
    '''https://www.appsloveworld.com/python/25/extract-images-from-idx3-ubyte-file-or-gzip-via-python'''
    '''Получаем изображения из архива'''
    try:
        import gzip
        import numpy as np
        with gzip.open(in_path, 'r') as f:
            # first 4 bytes is a magic number
            magic_number = int.from_bytes(f.read(4), 'big')
            # second 4 bytes is the number of images
            image_count = int.from_bytes(f.read(4), 'big')
            # third 4 bytes is the row count
            row_count = int.from_bytes(f.read(4), 'big')
            # fourth 4 bytes is the column count
            column_count = int.from_bytes(f.read(4), 'big')
            # rest is the image pixel data, each pixel is stored as an unsigned byte
            # pixel values are 0 to 255
            image_data = f.read()
            images = np.frombuffer(image_data, dtype=np.uint8).reshape((image_count, row_count, column_count))
            return images, image_count
    except Exception as ex:
        print(ex)
        return None, None
    
def training_labels(in_path):
    '''Получаем разметки из архива'''
    try:
        import gzip
        import numpy as np
        with gzip.open(in_path, 'r') as f:
            # first 4 bytes is a magic number
            magic_number = int.from_bytes(f.read(4), 'big')
            # second 4 bytes is the number of labels
            label_count = int.from_bytes(f.read(4), 'big')
            # rest is the label data, each label is stored as unsigned byte
            # label values are 0 to 9
            label_data = f.read()
            labels = np.frombuffer(label_data, dtype=np.uint8)
            return labels, label_count
    except Exception as ex:
        print(ex)
        return None, None
    
def transformation_zero_one(in_array):
    """Перед подачей данных перцептрону, необходимо, чтобы 𝑥𝑖𝑗 ∈ {0,1} 0 - 0; 255 - 1""" 
    import numpy as np
    in_array = np.round(in_array/255.0).astype(np.uint8)
    return in_array

def transformation_one_hot_encoding(in_array):
    """Метки в унитарный код (англ. one-hot encoding) – двоичный код фиксированной длины, содержащим только одну единицу"""
    import numpy as np
    in_array = np.array(in_array) 
    n = in_array.shape[0]
    categorical = np.zeros((n, 10))
    categorical[np.arange(n), in_array] = 1
    return categorical