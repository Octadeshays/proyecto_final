#Importamos librerias

import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from varname import varname, nameof


#TF
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import model_from_json



########################################################################################################################

#Importamos el dataframe y pasamos de usar la columna datetime a segundos

#temp_df = pd.read_csv('/content/drive/MyDrive/Proyecto Final Mecatronica /Datasets/data_temp_with_noise.csv', index_col=['seconds'])
temp_df = pd.read_csv('/content/drive/MyDrive/Proyecto Final Mecatronica /Datasets/data_temp_with_noise.csv', index_col=['datetime'])
temp_df.insert(0, 'seconds', range(0, (len(temp_df))*30, 30))
temp_df.set_index( 'seconds', inplace = True )


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

#Dividimos la data en train, test, validation set (70,20,10 %): 

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
      
    '''
    #si la clasificacion es multistep, tenemos que cambiar la cantidad de inputs a la hora de crear la ventana para que funcione
    
    if label_width > 1:
        input_width = label_width + (input_width - 1)
    '''

    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df


    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}


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

    #Al separar la data no se guarda el shape original, por lo que hay que volverla a definir
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
        if net_type == 'lstm' or net_type == 'rnn' or net_type == 'gru' or net_type == 'bi_lstm': 
            plt.scatter(self.label_indices*30, predictions,
                      marker='X', edgecolors='k', label='Predictions',
                      c='#ff7f0e', s=64)

        if net_type == 'cnn': 
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

@property #property me permite que agregue getters a la clase usando la misma sintáxis que si estuviera accediendo directamente
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
    #Ploteamos accuracy de train vs val
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



#Funcion para testear muchas configuraciones para un modelo 
def train_and_test_model_multiple_configs(net_type, model, model_name, configs, max_epochs):

    model_name_list = []
    net_type_list = []
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

        if net_type =='lstm' or net_type =='rnn' or net_type =='bi_lstm' or net_type == 'gru':
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
        net_type_list.append(net_type)
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


    results_df = pd.DataFrame(list(zip(model_name_list, net_type_list, steps_num_list, labels_num_list, shifts_num_list, max_epochs_list, 
                                       num_epochs_list, loss_list, rmse_list, val_loss_list, val_rmse_list,
                                       performance_val_baseline_loss_list, performance_val_baseline_rmse_list,
                                       performance_val_model_loss_list, performance_val_model_rmse_list, 
                                       performance_test_baseline_loss_list, performance_test_baseline_rmse_list,
                                       performance_test_model_loss_list, performance_test_model_rmse_list)), 
                              columns =['model_name', 'net_type', 'steps_num', 'labels_num', 'shifts_num', 'max_epochs', 'num_epochs', 'loss', 
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

    return {'results': results_df, 'model' : new_model}


  
########################################################################################################################


#Funcion para cortar dataset de forma random para simular entrada de datos en orbita. Sube a Drive un .csv de datos de input
def cut_dataset_random(data, period=570, orbit_time=5400, dt=30): #el period esta definido en 570 para conseguir 20 puntos medidos
    data_cut=data.copy()
    data_cut.loc[:,:]=np.nan
    N=data.index.tolist()[-1] #hacemos una lista con el index
    N_orbitas=int(N/orbit_time)

    for i in range(N_orbitas):
        encendido=np.random.randint(i*orbit_time/dt, (i+1)*orbit_time/dt)
        encendido=encendido*dt
        apagado=encendido+period

        data_cut.loc[encendido:apagado, :]=data.loc[encendido:apagado, :]
    return data_cut

  
########################################################################################################################
  
#Funcion para testear desempeño en orbita
def test_model_on_orbit_random(model_to_use, orbits = 5, node_to_extract = 1, max_epochs = 20):

    #Parametros de funcion:
    #orbits = 5
    #node_to_extract = 1

    #Importamos el dataframe y pasamos de usar la columna datetime a segundos

    #temp_df = pd.read_csv('/content/drive/MyDrive/Proyecto Final Mecatronica /Datasets/data_temp_with_noise.csv', index_col=['seconds'])
    temp_df = pd.read_csv('/content/drive/MyDrive/Proyecto Final Mecatronica /Datasets/data_temp_with_noise.csv', index_col=['datetime'])
    temp_df.insert(0, 'seconds', range(0, (len(temp_df))*30, 30))
    temp_df.set_index( 'seconds', inplace = True )

    #Definimos tamaño para imprimir los dataframe
    plt.rcParams["figure.figsize"] = (20,5)

    #Reduzco el data set a un solo nodo
    temp_df_node = pd.DataFrame(temp_df['Node %i' % (node_to_extract)])

    #Generamos el dataframe de data cortada (tomada de orbita)
    temp_df_node_cut=cut_dataset_random(temp_df)

    #Recortamos los datasets a un cierto numero de orbitas
    temp_df_node_less_orbits = temp_df_node[0:180*orbits] #sabiendo que tenemos 180 puntos por orbita
    temp_df_node_cut_less_orbits = temp_df_node_cut[0:180*orbits] #sabiendo que tenemos 180 puntos por orbita
    plt.plot(temp_df_node_less_orbits.loc[:, 'Node %i' % (node_to_extract)], 'r')
    plt.plot(temp_df_node_cut_less_orbits.loc[:, 'Node %i' % (node_to_extract)])

    #Obtenemos los indices de las filas que no tienen Nan
    index_with_values = [index for index, row in temp_df_node_cut_less_orbits.iterrows() if not row.isnull().any()]

    #Inicializamos variables antes de la iteracion
    predict_stop_flag = 0
    current_orbit = 0

    #Inicializamos el dataframe de predicciones total
    index_list = list(temp_df_node_cut_less_orbits.index) 
    total_predictions_df = pd.DataFrame({'seconds': index_list})
    total_predictions_df['Node %i' % (node_to_extract)] = 100000 #ponemos un valor dummy
    total_predictions_df.set_index('seconds', inplace = True)

    while(predict_stop_flag == 0):

        print('\nOrbita actual: ',current_orbit)

        #Conseguimos valores de input de la orbita actual
        input_values_indexes = index_with_values[current_orbit*20: current_orbit*20 + 20]
        
        #Verificamos que sean 20 puntos de input, en caso de no serlo, frenamos la ejecucion
        if len(input_values_indexes) != 20:
            print('ERROR: Los valores de input no son 20. Son: ', len(input_values_indexes))
            predict_stop_flag = 1
        
        #Definimos cuantos puntos de output necesitamos en esta orbita

        #Definimos el primer punto a predecir
        first_predicted_index = input_values_indexes[-1] + 30 #Primer punto a predecir. Son 30 segundos luego de el ultimo valor de la lista

        #Definimos el ultimo punto a predecir
        if current_orbit < orbits - 1:     #este es el caso de si no estamos en la ultima orbita
            last_predicted_index = index_with_values[current_orbit*20 + 20] - 30 #es el proximo valor conocido de la lista, pero con 30 segundos restados

        elif current_orbit == orbits - 1: #Si estamos en la ultima orbita, no tendremos puntos restantes conocidos en la lista. En este caso, tomamos el ultimo punto del dataframe
            last_predicted_index = temp_df_node_cut_less_orbits.index[-1]

        #Obtenemos los puntos de output
        points_to_predict = int((last_predicted_index - first_predicted_index) / 30) #restamos los indices (los segundos) de los puntos a predecir y lo dividimos por el tiempo de muestreo

        configs_list = [[20,points_to_predict]]

        #Entrenamos el modelo

        model_output = train_and_test_model_multiple_configs('cnn', model_to_use, nameof(model_to_use), configs_list, max_epochs)

        #Guardamos los puntos para evaluar 
        input_values = temp_df_node_less_orbits.loc[input_values_indexes,'Node %i' % (node_to_extract)].to_list()

        #Hacemos la prediccion de los input_values 
        input_values_array_reshaped = np.reshape(input_values, (1,20,1)) #el input_size es 20

        model_predictions = model_output['model'].predict(input_values_array_reshaped)
        model_predictions_list = list(np.reshape(model_predictions, (points_to_predict)))

        #Creamos dataframe con resultados predichos
        predicted_indexes = [*range(first_predicted_index, last_predicted_index, 30)]

        index_prediction_dict = dict(zip(predicted_indexes,model_predictions_list)) #creamos diccionario de pares indices-predicciones

        total_predictions_df['Node %i' % (node_to_extract)].update(pd.Series(index_prediction_dict)) #reemplazamos las predicciones en el dataframe

        #Actualizamos la orbita
        current_orbit += 1

        #Si llegamos al limite de orbitas, frenamos
        if current_orbit >= orbits:
            predict_stop_flag = 1

    #Reemplazamos el valor dummy del Dataframe de predicciones por NaNs
    total_predictions_df['Node %i' % (node_to_extract)].replace({100000.000000: np.nan}, inplace=True)

    #Imprimimos al final
    #plt.plot(temp_df_node_less_orbits.loc[:, 'Node %i' % (node_to_extract)], 'r')  #data original
    plt.plot(temp_df_node_cut_less_orbits.loc[:, 'Node %i' % (node_to_extract)]) #data de orbita
    plt.plot(total_predictions_df.loc[:, 'Node %i' % (node_to_extract)], 'y') #prediccion del modelo

    ################################################################
    #Subimos .csv de inputs y outputs a Google Drive
    ################################################################

    #Creamos diccionario con pares 'seconds' - 'fecha de lectura'
    seconds_list = list(temp_df.index)
    datetime_created_at_list = []

    current_datetime = datetime.now()

    for i in range(len(seconds_list)):

        if i == 0: #en la primera iteracion definimos el datetime actual
            datetime_created_at_list.append(current_datetime)
        else:
            datetime_created_at_list.append(datetime_created_at_list[-1] + timedelta(seconds = 30))

    datetime_created_at_list = [date_obj.strftime("%d/%m/%Y %H:%M:%S") for date_obj in datetime_created_at_list] #pasamos tipo de datos de datetime a string

    seconds_datetime_dict = dict(zip(seconds_list,datetime_created_at_list)) #creamos diccionario de pares 'seconds' - 'fecha de lectura'

    #Creamos dataframe de inputs para subir a Drive
    temp_df_node_cut_less_orbits_to_upload = temp_df_node_cut_less_orbits.dropna() #quitamos los NaNs
    temp_df_node_cut_less_orbits_to_upload["datetime_created_at"] = np.nan
    temp_df_node_cut_less_orbits_to_upload['datetime_created_at'].update(pd.Series(seconds_datetime_dict)) #reemplazamos los datetime en el dataframe

    #Creamos dataframe de outputs para subir a Drive
    total_predictions_df_to_upload = total_predictions_df.dropna() #quitamos los NaNs
    total_predictions_df_to_upload["datetime_created_at"] = np.nan
    total_predictions_df_to_upload['datetime_created_at'].update(pd.Series(seconds_datetime_dict)) #reemplazamos los datetime en el dataframe

    #Guardamos csvs en carpetas de Drive
    temp_df_node_cut_less_orbits_to_upload.to_csv('/content/drive/MyDrive/Proyecto Final Mecatronica /Notebooks/input_history/input_{}.csv'.format(current_datetime.strftime("%d_%m_%Y_%H:%M:%S") ))
    total_predictions_df_to_upload.to_csv('/content/drive/MyDrive/Proyecto Final Mecatronica /Notebooks/output_history/output_{}.csv'.format(current_datetime.strftime("%d_%m_%Y_%H:%M:%S") ))


