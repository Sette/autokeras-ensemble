import autokeras as ak
from keras import backend as K 
from autokeras.utils import types
import datetime
import numpy as np
from autokeras.engine import tuner


from pathlib import Path
from typing import Type
from typing import Tuple
from typing import List
from typing import Optional
from typing import Union
import os
import gc

import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


class AutokerasImageEnsemble():

  """AutoKeras ensemble image classification class.
  # Arguments
        mode: String. Type of variation for generated neural network architectures. 
            By default, the "tuner" is varied.
        ensemble_type. Type of variaton for thecnique to generate ensemble models.
            By default, the "stack" with Logistic Regression is varied. "sum" is another possibility.
        num_classes: Int. Defaults to None. If None, it will be inferred from the
            data.
        multi_label: Boolean. Defaults to False.
        loss: A Keras loss function. Defaults to use 'binary_crossentropy' or
            'categorical_crossentropy' based on the number of classes.
        metrics: A list of Keras metrics. Defaults to use 'accuracy'.
        project_name: String. The name of the AutoModel.
            Defaults to 'image_classifier'.
        max_trials: Int. The maximum number of different Keras Models to try.
            The search may finish before reaching the max_trials. Defaults to 100.
        directory: String. The path to a directory for storing the search outputs.
            Defaults to None, which would create a folder with the name of the
            AutoModel in the current directory.
        objective: String. Name of model metric to minimize
            or maximize, e.g. 'val_accuracy'. Defaults to 'val_loss'.
        overwrite: Boolean. Defaults to `False`. If `False`, reloads an existing
            project of the same name if one is found. Otherwise, overwrites the
            project.
        seed: Int. Random seed.

  """

  def __init__(self,
              mode = "tuner",
              ensemble_type = "stack",
              num_classes: Optional[int] = None,
              multi_label: bool = False,
              loss: types.LossType = None,
              metrics: Optional[types.MetricsType] = None,
              project_name: str = "image_classifier",
              max_trials: int = 100,
              directory: Union[str, Path, None] = None,
              objective: str = "val_loss",
              tuner: Union[str, Type[tuner.AutoTuner]] = None,
              overwrite: bool = False,
              seed: Optional[int] = None):
    
    self.mode = mode
    self.ensemble_type = ensemble_type
    self.num_classes = num_classes
    self.multi_label = multi_label
    self.loss = loss
    self.metrics = metrics
    self.project_name = project_name
    self.max_trials = max_trials
    self.directory = directory
    self.objective = objective
    self.tuner = tuner
    self.overwrite = overwrite
    self.seed = seed
    self.tuner_list = ['greedy', 'bayesian', 'hyperband','random']




  def fit(self,
        x: Optional[types.DatasetType] = None,
        y: Optional[types.DatasetType] = None,
        epochs: Optional[int] = None,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
        validation_split: Optional[float] = 0.2,
        validation_data: Union[
            tf.data.Dataset, Tuple[types.DatasetType, types.DatasetType], None
        ] = None):
      """Search for the best model and hyperparameters for any tuner.

        It will search for the best model based on the performances on
        validation data.

        # Arguments
            x: numpy.ndarray or tensorflow.Dataset. Training data x. The shape of
                the data should be (samples, width, height)
                or (samples, width, height, channels).
            y: numpy.ndarray or tensorflow.Dataset. Training data y. It can be raw
                labels, one-hot encoded if more than two classes, or binary encoded
                for binary classification.
            epochs: Int. The number of epochs to train each model during the search.
                If unspecified, by default we train for a maximum of 1000 epochs,
                but we stop training if the validation loss stops improving for 10
                epochs (unless you specified an EarlyStopping callback as part of
                the callbacks argument, in which case the EarlyStopping callback you
                specified will determine early stopping).
            callbacks: List of Keras callbacks to apply during training and
                validation.
            validation_split: Float between 0 and 1. Defaults to 0.2.
                Fraction of the training data to be used as validation data.
                The model will set apart this fraction of the training data,
                will not train on it, and will evaluate
                the loss and any model metrics
                on this data at the end of each epoch.
                The validation data is selected from the last samples
                in the `x` and `y` data provided, before shuffling. This argument is
                not supported when `x` is a dataset.
                The best model found would be fit on the entire dataset including the
                validation data.
            validation_data: Data on which to evaluate the loss and any model metrics
                at the end of each epoch. The model will not be trained on this data.
                `validation_data` will override `validation_split`. The type of the
                validation data should be the same as the training data.
                The best model found would be fit on the training dataset without the
                validation data.

          """


      for tuner_name in self.tuner_list:
          # Initialize the image classifier.
          clf = ak.ImageClassifier(
              num_classes=self.num_classes,
              multi_label= self.multi_label,
              loss= self.loss,
              metrics=self.metrics,
              project_name= self.project_name+"_"+tuner_name,
              max_trials= self.max_trials,
              directory= self.directory + "_" + tuner_name,
              objective= self.objective,
              tuner= tuner_name,
              overwrite= self.overwrite,
              seed= self.seed
          )
          
          clf.fit(
              x= x, 
              y = y, 
              epochs=epochs,
              validation_split=validation_split,
              validation_data = validation_data
              )
          
          model = clf.export_model()
          
          try:
              model.save(self.directory + "_" + tuner_name+ "_tf" , save_format="tf")
          except:
              model.save(self.directory + "_" + tuner_name+ ".h5")

          del model
          K.clear_session()
          gc.collect()
 
  def load_models(self):
    models = dict.fromkeys(self.tuner_list)
    for tuner_name in self.tuner_list:
      yield load_model(self.directory + "_" + tuner_name+ "_tf", custom_objects=ak.CUSTOM_OBJECTS)

  def stack_dataset(self,x):
    # create dataset using ensemble
    stackX = None
    for model in self.load_models():
      # make prediction
      yhat = model.predict(x, verbose=0)
      # stack predictions into [rows, members, probabilities]
      if stackX is None:
        stackX = yhat
      else:
        stackX = np.dstack((stackX, yhat))
      
    # flatten predictions to [rows, members x probabilities]
    return stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))

  def __fit_stack(self,models,x,y):
    # fit standalone model
    model = LogisticRegression()
    stackX = self.stack_dataset(x)
    model.fit(stackX, y)
    return model



  def fit_ensemble(self,x,y):
    """Fit the ensemble for the given data.

      # Arguments
          x: Any allowed types according to the input node. Testing data.
          y: Any allowed types according to the head. Testing targets.

      # Returns
          Ensemble accuracy if ensemble_type is "sum" and model if ensemble_type is "stack".
    """
    try:
      models = self.load_models()
    except:
      print("Models not found. Try use fit() method first.")

    if self.ensemble_type == "stack":
      return self.__fit_stack(models,x,y)
    elif self.ensemble_type == "sum":
      return self.__evaluate_sum(x,y)


  def __evaluate_sum(self,x,y):
    """Evaluate the ensemble for the given data.

      # Arguments
          x: Any allowed types according to the input node. Testing data.
          y: Any allowed types according to the head. Testing targets.

      # Returns
          Ensemble accuracy.
    """
    models = self.load_models()
    # make predictions
    yhats = [model.predict(x) for model in models]
    yhats = np.array(yhats)
    # sum across ensembles
    summed = np.sum(yhats, axis=0)
    # argmax across classes
    yhat = np.argmax(summed, axis=1)
    K.clear_session()
    gc.collect()
    return accuracy_score(pred, y)

  def evaluate(self,x,y):
    """Evaluate the ensemble for the given data.

      # Arguments
          x: Any allowed types according to the input node. Testing data.
          y: Any allowed types according to the head. Testing targets.

      # Returns
          Ensemble accuracy.
    """
    models = self.load_models()
    if self.ensemble_type == "stack":
      model = self.__fit_stack(models,x,y)
      # make a prediction
      stackX = self.stack_dataset(x)
      yhat = model.predict(stackX)
      K.clear_session()
      gc.collect()
      return accuracy_score(y, yhat)

    elif self.ensemble_type == "sum":
      return self.__evaluate_sum(x,y)

  def compare(self,x,y):
    """Compare acc for the given data.

      # Arguments
          x: Any allowed types according to the input node. Testing data.
          y: Any allowed types according to the head. Testing targets.

      # Returns
          Dict for accuracy.
    """
    
    models = self.load_models()
    i=0
    # make predictions
    result = {}
    yhats = []
    for model in models:
      yhat = model.predict(x)
      yhats.append(yhat)
      # argmax across classes
      pred = np.argmax(yhat, axis=1)
      result.update({self.tuner_list[i]:accuracy_score(pred, y)})
      i+=1

    if self.ensemble_type == "stack":
      model = self.__fit_stack(models,x,y)
      stackX = self.stack_dataset(x)
      # make a prediction
      yhat = model.predict(stackX)
      acc = accuracy_score(y, yhat)

    elif self.ensemble_type == "sum":
      acc = self.__evaluate_sum(x,y)

    result.update({"ensemble":acc})
    K.clear_session()
    gc.collect()
    return result



