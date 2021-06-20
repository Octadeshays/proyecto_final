#Importamos librerias


import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime

#TF
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import model_from_json






########################################################################################################################

#Importamos la data

temp_df = pd.read_csv('/content/drive/MyDrive/Proyecto Final Mecatronica /Datasets/data_temp_with_noise.csv', index_col=['seconds'])
#temp_df.head()

#Definimos tamaño para imprimir los dataframe
plt.rcParams["figure.figsize"] = (20,5)

#Reduzco el data set a un solo nodo
node_to_extract = 1

temp_node_1_df = temp_df['Node %i' % node_to_extract]
temp_node_1_df = temp_node_1_df.to_frame()
temp_node_1_df.rename(columns = {'Node %i' % node_to_extract : 'node'}, inplace = True)

temp_node_1_df.reset_index()

#Graficamos el nodo
#temp_node_1_df.plot()

#Dividimos la data en train, test, validation set: 
#We'll use a (70%, 20%, 10%) split for the training, validation, and test sets. Note the data is not being randomly shuffled before splitting.

column_indices = {name: i for i, name in enumerate(temp_node_1_df.columns)}

n = len(temp_node_1_df)
train_df = temp_node_1_df[0:int(n*0.7)]
val_df = temp_node_1_df[int(n*0.7):int(n*0.9)]
test_df = temp_node_1_df[int(n*0.9):]






########################################################################################################################

#Funcion WindowGenerator

class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
      

    #si la clasificacion es multistep, tenemos que cambiar la cantidad de inputs a la hora de crear la ventana para que funcione
    
    '''
    if label_width > 1:
        input_width = label_width + (input_width - 1)
    '''

    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift
    #self.total_window_size = input_width 

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    
  def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
      labels = tf.stack(
          [labels[:, :, self.column_indices[name]] for name in self.label_columns],
          axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels

  def plot(self, model=None, plot_col='node', max_subplots=3, net_type = None):
    #aux = random.randint(0, (len(self.test_df)-self.total_window_size))  #No vamos a printear en random, sino en un punto determinado para q todos los graficos sean iguales
    aux = 1 #Vamos a printear desde el punto 1
    
    example_window = tf.stack([np.array(test_df[aux:aux+self.total_window_size])]
                          )
    
    example_inputs, example_labels = self.split_window(example_window)
    inputs, labels = example_inputs, example_labels
    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
      plt.subplot(max_n, 1, n+1)
      plt.ylabel(f'{plot_col} [normed]')
      plt.plot(self.input_indices*30, inputs[n, :, plot_col_index],
              label='Inputs', marker='.', zorder=-10)

      if self.label_columns:
        label_col_index = self.label_columns_indices.get(plot_col, None)
      else:
        label_col_index = plot_col_index

      if label_col_index is None:
        continue

      plt.scatter(self.label_indices*30, labels[n, :, label_col_index],
                  edgecolors='k', label='Labels', c='#2ca02c', s=64)
      if model is not None:

        predictions = model(inputs)

        #El 1er if es para el caso de redes donde no hacemos reshape (el output shape tiene 2 dimensiones). El segundo, donde si lo hacemos 
        # (el output shape tiene 3 dimensiones)
        if net_type == 'lstm' or net_type == 'rnn' or net_type == 'gru': 
            plt.scatter(self.label_indices*30, predictions,
                      marker='X', edgecolors='k', label='Predictions',
                      c='#ff7f0e', s=64)

        if net_type == 'cnn' : 
            plt.scatter(self.label_indices*30, predictions[n, :, label_col_index],
                      marker='X', edgecolors='k', label='Predictions',
                      c='#ff7f0e', s=64)

      if n == 0:
        plt.legend()

    plt.xlabel('Time [s]') 

    print('Window plot:')
 
    plt.show()

  def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=32,)

    ds = ds.map(self.split_window)

    return ds  

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

@property#property me permite que agregue getters a la clase usando la misma sintáxis que si estuviera accediendo directamente
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)

@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.test))
    # And cache it for next time
    self._example = result
  return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example



########################################################################################################################

#Definimos una función que hace el training para no tener que estar escribiendoló en cada modelo

def compile_and_fit(model, window, max_epochs = 20, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.RootMeanSquaredError()])

  history = model.fit(window.train, epochs=max_epochs,
                      validation_data=window.val,
                      callbacks=[early_stopping], verbose = 0)
  return history
  
  






########################################################################################################################


#Funciones para obtener el baseline de modelos de un solo output o muchos output

class Baseline(tf.keras.Model):
  def __init__(self, label_index=None):
    super().__init__()
    self.label_index = label_index

  def call(self, inputs):
    if self.label_index is None:
      return inputs
    result = inputs[:, :, self.label_index]
    return result[:, :, tf.newaxis]

class MultiStepLastBaseline(tf.keras.Model):
  def call(self, inputs):
    return tf.tile(inputs[:, -1:, :], [1, label_size, 1])



baseline = Baseline(label_index=column_indices['node'])
baseline.compile(loss=tf.losses.MeanSquaredError(), metrics=[tf.metrics.RootMeanSquaredError()])

multi_step_baseline = MultiStepLastBaseline()
multi_step_baseline.compile(loss=tf.losses.MeanSquaredError(), metrics=[tf.metrics.MeanAbsoluteError()])
  




########################################################################################################################



#Funcion para graficar objetos history
def plot_history(history):

    '''
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    '''

    #Ploteamos rmse de train vs val
    plt.plot(history.history['root_mean_squared_error'])
    plt.plot(history.history['val_root_mean_squared_error'])
    plt.title('model rmse')
    plt.ylabel('rmse')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    #Ploteamos loss de train vs val
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()



########################################################################################################################

#Definimos una funcion para hacer todo esto en las futuras iteraciones

def train_and_test_model(steps_num, labels_num, shifts_num, model, max_epochs, net_type):

    step_window = WindowGenerator(input_width = steps_num, label_width= labels_num , shift = shifts_num, label_columns=['node'])
    #step_window.plot()

    print('Caracteristicas de Window:\n')
    print(step_window)

    #Primero creamos un modelo baseline. Este modelo simplemente predice que el siguiente valor va a ser igual al siguiente.
    #Este modelo absurdo nos da un mínimo de performance para saber que tan bien predice nuestro modelo

    val_performance = {}
    test_performance = {}

    if labels_num == 1:

        val_performance['Baseline'] = baseline.evaluate(step_window.val, verbose=0)
        test_performance['Baseline'] = baseline.evaluate(step_window.test, verbose=0)

    else: #Para prediccion multi step

        label_size = labels_num

        class MultiStepLastBaseline(tf.keras.Model):
            def call(self, inputs):
                return tf.tile(inputs[:, -1:, :], [1, label_size, 1])

        multi_step_baseline = MultiStepLastBaseline()
        multi_step_baseline.compile(loss=tf.losses.MeanSquaredError(), metrics=[tf.metrics.MeanAbsoluteError()])

        val_performance['Baseline'] = multi_step_baseline.evaluate(step_window.val, verbose=0)
        test_performance['Baseline'] = multi_step_baseline.evaluate(step_window.test, verbose=0)

    #Confirmamos tamaño de input y output (las labels y las output tienen que tener la misma longitud!)
    print("\nShape de la red: \n")
    print('Input shape: ', step_window.example[0].shape)
    print('Labels shape:', step_window.example[1].shape)
    print('Output shape:', model(step_window.example[0]).shape)
    print('\n\n')

    #Entrenamos
    history = compile_and_fit(model, step_window, max_epochs, patience=5)

    #Vemos performance comparado al baseline
    val_performance['Model'] = model.evaluate(step_window.val)
    test_performance['Model'] = model.evaluate(step_window.test, verbose=0)

    print('\nPerformance del modelo: ')
    print('Validation:', val_performance)
    print('Test:      ', test_performance)
    print('\n\n')
    
    #Imprimimos prediccion
    #step_window.plot(model)
    step_window.plot(model, net_type = net_type)
    return history, val_performance, test_performance



########################################################################################################################



#Funcion para testear muchas configuraciones para un modelo CNN
def train_and_test_model_multiple_configs(net_type, model, model_name, configs, max_epochs):

    model_name_list = []
    steps_num_list = []
    labels_num_list = []
    shifts_num_list = []
    max_epochs_list = []
    num_epochs_list = []
    loss_list = []
    rmse_list = []
    val_loss_list = []
    val_rmse_list = []

    performance_val_baseline_loss_list = []
    performance_val_baseline_rmse_list = []
    performance_val_model_loss_list = []
    performance_val_model_rmse_list = []
    performance_test_baseline_loss_list = []
    performance_test_baseline_rmse_list = []
    performance_test_model_loss_list = []
    performance_test_model_rmse_list = []

    for config in configs:
        
        print('\n\n\n\n\n' + 63 * '#' + '\n' + 63 * '#')
        print('Configuracion: %s' % str(config))
        print(63 * '#' + '\n' + 63 * '#' + '\n')

        steps_num = config[0]
        labels_num = config[1]
        shifts_num = labels_num #en la forma en que estamos testeando actualmente, shifts_num = labels_num



        #Tenemos que cambiar los parametros de la red en funcion del numero de inputs y outputs

        if net_type == 'cnn':

            model.layers[0].kernel_size = steps_num
            model.layers[-1].units = labels_num
            new_model = model_from_json(model.to_json())
            new_model.add(tf.keras.layers.Reshape([labels_num, 1])) #agregamos capa para hacer reshape

        if net_type =='lstm':
            model.layers[-1].units = labels_num
            new_model = model_from_json(model.to_json())  

        if net_type =='rnn':
            model.layers[-1].units = labels_num
            new_model = model_from_json(model.to_json())
            #new_model.add(tf.keras.layers.Reshape([labels_num, 1])) #agregamos capa para hacer reshape
            
        if net_type == 'gru':
            
            #model.layers[0].kernel_size = steps_num
            model.layers[-1].units = labels_num
            new_model = model_from_json(model.to_json())
            #new_model.add(tf.keras.layers.Reshape([labels_num, 1])) #agregamos capa para hacer reshape

        history, val_performance, test_performance = train_and_test_model(steps_num, labels_num, shifts_num, new_model, max_epochs, net_type)

        #Ploteamos loss de train vs validation
        plot_history(history)

        #Imprimimos summary
        print('\nSummary del modelo:\n')
        new_model.summary()

        #Appendamos listas
        model_name_list.append(model_name)
        steps_num_list.append(steps_num)
        labels_num_list.append(labels_num)
        shifts_num_list.append(shifts_num)
        max_epochs_list.append(max_epochs)

        num_epochs_list.append(len(history.history['val_loss']))

        loss_list.append(history.history['loss'][-1])
        rmse_list.append(history.history['root_mean_squared_error'][-1])
        val_loss_list.append(history.history['val_loss'][-1])
        val_rmse_list.append(history.history['val_root_mean_squared_error'][-1])

        performance_val_baseline_loss_list.append(val_performance['Baseline'][0])
        performance_val_baseline_rmse_list.append(val_performance['Baseline'][1])

        performance_val_model_loss_list.append(val_performance['Model'][0])
        performance_val_model_rmse_list.append(val_performance['Model'][1])

        performance_test_baseline_loss_list.append(test_performance['Baseline'][0])
        performance_test_baseline_rmse_list.append(test_performance['Baseline'][1])

        performance_test_model_loss_list.append(test_performance['Model'][0])
        performance_test_model_rmse_list.append(test_performance['Model'][1])


    results_df = pd.DataFrame(list(zip(model_name_list, steps_num_list, labels_num_list, shifts_num_list, max_epochs_list, 
                                       num_epochs_list, loss_list, rmse_list, val_loss_list, val_rmse_list,
                                       performance_val_baseline_loss_list, performance_val_baseline_rmse_list,
                                       performance_val_model_loss_list, performance_val_model_rmse_list, 
                                       performance_test_baseline_loss_list, performance_test_baseline_rmse_list,
                                       performance_test_model_loss_list, performance_test_model_rmse_list)), 
                              columns =['model_name', 'steps_num', 'labels_num', 'shifts_num', 'max_epochs', 'num_epochs', 'loss', 
                                        'rmse', 'val_loss', 'val_rmse', 
                                        'performance_val_baseline_loss', 'performance_val_baseline_rmse',
                                        'performance_val_model_loss', 'performance_val_model_rmse',
                                        'performance_test_baseline_loss', 'performance_test_baseline_rmse',
                                        'performance_test_model_loss', 'performance_test_model_rmse'])
    

    #Guardamos df en .csv
    now = datetime.now() 
    date_time = now.strftime("%d-%m_%H:%M:%S")
    csv_name = '%s_%s_%s' % (net_type, model_name, date_time) +  '.csv'  
    address = '/content/drive/MyDrive/Proyecto Final Mecatronica /Notebooks/results/' + csv_name

    results_df.to_csv(address, index=False)

    return results_df

