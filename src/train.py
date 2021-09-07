from nltk.sem import evaluate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('punkt')

from sklearn.linear_model import LogisticRegression
from sklearn import naive_bayes
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from sklearn.svm import SVC

import config
import argparse
import model_util

def clean_text(text:str):

  """
  util function to clean the text
  ------------------------------
  Parameters-
  text : input text to preprocess

  Return-
  text : cleaned text
  """

  # lower case the text
  text = str(text).lower()
  # remove twitter user names
  text = re.sub('@[\w]+','',text)
  # remove hyperlinks
  text = re.sub('https?://\S+|www\.\S+', '', text)
  text = re.sub('<.,*?>+', '', text)
  text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

  return text

def labelencoder():

  """
  Creates and returns label encoders for each task.

  Returns:
          task_labels (dict): label encoder dict of the given input task
  """
  
  task_a_labels = {'NAG':0,'OAG':1,'CAG':2}
  
  task_b_labels = {'NGEN':0,'GEN':1}

  return task_a_labels,task_b_labels


def BaselineEvaluateTest(task,model,vec,eval):
  
  """
  Evaluate the models on Dev and Test sets and prints the classification report
  
  Parameters: 
          task (str) : Type of Task
                      - A
                      - B
          eval (str) : evaluation set
                      - 'dev' (default)
                      - 'test'
          model  : trained model
          vec : tokenizer
  
  """
  task_a_labels,task_b_labels  = labelencoder()

  if task.upper() == "A":
    col = 'Sub-task A'
  elif task.upper() == "B":
    col = 'Sub-task B'
  else:
    raise ValueError("Task not found")

  if eval == 'dev':
    path_to_data =config.VALIDATION_FILE
  elif eval == "test":
    path_to_data = config.TEST_FILE
    path_to_labels = '../input/gold/trac2_hin_gold_' + task.lower() + '.csv' 

    test_labels = pd.read_csv(path_to_labels)
    test_labels = test_labels.replace({'Sub-task A':task_a_labels,'Sub-task B':task_b_labels})
  else:
    raise ValueError("eval type not found")
  
  eval_data = pd.read_csv(path_to_data)
  eval_data = eval_data.replace({'Sub-task A':task_a_labels,'Sub-task B':task_b_labels})
  
  x_eval = vec.transform(eval_data['Text'])
  eval_preds = model.predict(x_eval)
  
  if eval=="test":
    print("Test set Evaluation report")
    eval_report = metrics.classification_report(test_labels[col],eval_preds)
  else:
    print("Dev set Evaluation report")
    eval_report = metrics.classification_report(eval_data[col],eval_preds)
  print(eval_report)

def run(task,Model,vec,eval):

  """
  Train the model on the given task ,using the given vec as the input
  ------------------------------------------------------------------

  Parameters:

  task : 'a' - Task A (Aggressive text detection)
         'b' - Task B (Gendered text detection)
  
  """
  model_dict = {"LR":"Logistic Regression","NB":"Naive Bayes","SVC": "SVM Classifier"}
  vec_dict = {"CV":"CountVectorizer","TFIDF":"TfidfVectorizer"}
  task_dict = {"A": "Aggresive Text Classification","B":"Gendered Text Classification"}

  data = pd.read_csv(config.TRAIN_FILE)
  data.columns = ['ID','Text','aggressive','gendered']

  task_a_labels,task_b_labels  = labelencoder()

  data = data.replace({'aggressive':task_a_labels,'gendered':task_b_labels})
  
  if task == "a":
    target = 'aggressive'
  elif task == "b":
    target = 'gendered'
  else:
    raise ValueError("Task not found")

  # random state for reproducibility 
  x_train,x_test,y_train,y_test = train_test_split(data['Text'],data[target],stratify=data[target],random_state=21)  

  vect = model_util.vecs[vec]
  vect.fit(x_train)

  x_train_cv = vect.transform(x_train)
  x_test_cv = vect.transform(x_test)

  model = model_util.models[Model]

  model.fit(x_train_cv,y_train)

  preds = model.predict(x_test_cv)
 
  report = metrics.classification_report(y_test,preds)
  print("\n")
  print(model_dict[Model.upper()] + " with " + vec_dict[vec.upper()])
  print("\n")
  print("Training Report")
  print(report)
  
  if eval:
    BaselineEvaluateTest(task,model,vect,eval)

if __name__ == "__main__":
  
  parser = argparse.ArgumentParser()
  
  parser.add_argument(
    "--task",
    type=str
  )
  parser.add_argument(
    "--model",
    type=str
  )
  parser.add_argument(
    "--vectorizer",
    type=str
  )

  parser.add_argument(
    "--eval",
    type=str
  )
  
  args = parser.parse_args()

  run(args.task,args.model,args.vectorizer,args.eval)