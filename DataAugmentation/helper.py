import math

import numpy as np
import matplotlib.pyplot as plt

import os
import h5py
import pickle
import tensorflow as tf
from nose.tools import assert_equal
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical


import json

class Helper():
  raise_exception_on_missing_data = True
  
  def __init__(self):
    self.dataset =  "shipsnet.json"
    
    # Data directory
    self.DATA_DIR = "./Data"

    if not os.path.isdir(self.DATA_DIR):
      if self.raise_exception_on_missing_data:
        raise Exception(
          "The required data set is missing: Create a subdirectory named 'Data' in the current directory and place the data file '{f:s}' in it.".format(f=self.dataset)
        )
      else:
        self.DATA_DIR = "../resource/asnlib/publicdata/ships_in_satellite_images/data"



  def getData(self):
    
    data,labels = self.json_to_numpy( os.path.join(self.DATA_DIR, self.dataset) )
    return data, labels

  def showData(self, data, labels, num_cols=5, cmap=None):
    # Plot the first num_rows * num_cols images in X
    (num_rows, num_cols) = ( math.ceil(data.shape[0]/num_cols), num_cols)

    fig = plt.figure(figsize=(10,10))
    # Plot each image
    for i in range(0, data.shape[0]):
        img, img_label = data[i], labels[i]
        ax  = fig.add_subplot(num_rows, num_cols, i+1)
        _ = ax.set_axis_off()
        _ = ax.set_title(img_label)

        _ = plt.imshow(img, cmap=cmap)
    fig.tight_layout()

    return fig

  def modelPath(self, modelName):
      return os.path.join(".", "models", modelName)

  def saveModel(self, model, modelName): 
      model_path = self.modelPath(modelName)
      
      try:
          os.makedirs(model_path)
      except OSError:
          print("Directory {dir:s} already exists, files will be over-written.".format(dir=model_path))
          
      # Save JSON config to disk
      json_config = model.to_json()
      with open(os.path.join(model_path, 'config.json'), 'w') as json_file:
          json_file.write(json_config)
      # Save weights to disk
      model.save_weights(os.path.join(model_path, 'weights.h5'))
      
      print("Model saved in directory {dir:s}; create an archive of this directory and submit with your assignment.".format(dir=model_path))

  def loadModel(self, modelName):
      model_path = self.modelPath(modelName)
      
      # Reload the model from the 2 files we saved
      with open(os.path.join(model_path, 'config.json')) as json_file:
          json_config = json_file.read()
    
      model = tf.keras.models.model_from_json(json_config)
      model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
      model.load_weights(os.path.join(model_path, 'weights.h5'))
      
      return model

  def saveModelNonPortable(self, model, modelName): 
      model_path = self.modelPath(modelName)
      
      try:
          os.makedirs(model_path)
      except OSError:
          print("Directory {dir:s} already exists, files will be over-written.".format(dir=model_path))
          
      model.save( model_path )
      
      print("Model saved in directory {dir:s}; create an archive of this directory and submit with your assignment.".format(dir=model_path))
   
  def loadModelNonPortable(self, modelName):
      model_path = self.modelPath(modelName)
      model = self.load_model( model_path )
      
      # Reload the model 
      return model

  def saveHistory(self, history, model_name):
      history_path = self.modelPath(model_name)

      try:
          os.makedirs(history_path)
      except OSError:
          print("Directory {dir:s} already exists, files will be over-written.".format(dir=history_path))

      # Save JSON config to disk
      with open(os.path.join(history_path, 'history'), 'wb') as f:
          pickle.dump(history.history, f)

  def loadHistory(self, model_name):
      history_path = self.modelPath(model_name)
      
      # Reload the model from the 2 files we saved
      with open(os.path.join(history_path, 'history'), 'rb') as f:
          history = pickle.load(f)
      
      return history

  def MyModel(self, test_dir, model_path):
      # YOU MAY NOT change model after this statement !
      model = self.loadModel(model_path)
      
      # It should run model to create an array of predictions; we initialize it to the empty array for convenience
      predictions = []
      
      # We need to match your array of predictions with the examples you are predicting
      # The array below (ids) should have a one-to-one correspondence and identify the example your are predicting
      # For Bankruptcy: the Id column
      # For Stock prediction: the date on which you are making a prediction
      ids = []
      
      # YOUR CODE GOES HERE
      
      
      return predictions, ids

  def json_to_numpy(self, json_file):
    # Read the JSON file
    f = open(json_file)
    dataset = json.load(f)
    f.close()

    data = np.array(dataset['data']).astype('uint8')
    labels = np.array(dataset['labels']).astype('uint8')

    # Reshape the data
    data = data.reshape([-1, 3, 80, 80]).transpose([0,2,3,1])

    return data, labels




  from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
  modelName = "Ships_in_satellite_images"
  es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=.01, patience=2, verbose=0, mode='auto', baseline=None, restore_best_weights=True)

  callbacks = [ es_callback,
                ModelCheckpoint(filepath=modelName + ".ckpt", monitor='accuracy', save_best_only=True)
                ]   

  max_epochs = 30

  def train(self, model, X, y, model_name, epochs=max_epochs):
    # Describe the model
    model.summary()

    # Compile the model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

    # Fix the validation set (for repeatability, not a great idea, in general)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=42)
    
    print("Train set size: ", X_train.shape[0], ", Validation set size: ", X_valid.shape[0])

    history = model.fit(X_train, y_train, epochs=max_epochs, validation_data=(X_valid, y_valid), callbacks=callbacks)
    fig, axs = plotTrain(history, model_name)

    return history, fig, axs

  def plotTrain(self, history, model_name="???"):
    fig, axs = plt.subplots( 1, 2, figsize=(12, 5) )

    # Plot loss
    axs[0].plot(history.history['loss'])
    axs[0].plot(history.history['val_loss'])
    axs[0].set_title(model_name + " " + 'model loss')
    axs[0].set_ylabel('loss')
    axs[0].set_xlabel('epoch')
    axs[0].legend(['train', 'validation'], loc='upper left')
   
    # Plot accuracy
    axs[1].plot(history.history['accuracy'])
    axs[1].plot(history.history['val_accuracy'])
    axs[1].set_title(model_name + " " +'model accuracy')
    axs[1].set_ylabel('accuracy')
    axs[1].set_xlabel('epoch')
    axs[1].legend(['train', 'validation'], loc='upper left')

    return fig, axs
  def model_interpretation(self, clf):
    dim = round( clf.coef_[0].shape[-1] **0.5)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1,1,1)
    scale = np.abs(clf.coef_[0]).max()
    _= ax.imshow( clf.coef_[0].reshape(dim, dim), interpolation='nearest',
                   cmap="gray",# plt.cm.RdBu, 
                   vmin=-scale, vmax=scale)


    _ = ax.set_xticks(())
    _ = ax.set_yticks(())

    _= fig.suptitle('Parameters')
