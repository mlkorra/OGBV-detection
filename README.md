# OGBV-detection

## Folder Structure
    
    ├── input                             # Contains data 
         |── train                        # Train and Dev data
         |── test                         # Test data
    ├── models                            # Model pkl files  
    ├── notebooks                         # Experiment and Analysis notebooks 
         |── EDA.ipynb                    # Preprocessing and EDA of data
         |── Modelling Experiments.ipynb  # Modeling experiments on 
         |── Model Result Analysis.ipynb  # Error Analysis 
         |── Model  Interpretation.ipynb  # Explainability of Models
         |── Fine tuning Bert.ipynb       # Fine tuning BERT
    ├── src                               # Source files
    ├── output                            # Prediction files
    |── requirements.txt
    └── README.md                   
## Problem Statement

Build a gender based violence classifier based on the TRAC2020 data

## Run the Baselines
```
cd src
```
```python

python train.py --task [a,b] --model [lr,nb] --vect [cv,tfidf] --eval [test,dev]

usage: train.py [-h] [--task TASK] [--model MODEL] [--vectorizer VECTORIZER]
                [--eval EVAL]

optional arguments:
  -h, --help            show this help message and exit
  --task TASK           Provide the Task [a,b]
  --model MODEL         select Model [lr - Logistic Regression,nb - Naive
                        Bayes]
  --vectorizer VECTORIZER
                        select vectorizer [cv - CountVectorizer,tfidf -
                        TfidfVectorizer]
  --eval EVAL           evaluate on either [test,dev] set

```

## Task A - Aggresive Text Classification

F1 (weighted scores) 

|Model | dev | test|
|----|---|---|
|Logistic Regression + CV |0.66|0.69|
|SVC|0.68 |0.74 |

## Task B - Gendered Text Classification

F1 (weighted scores)
|Model|dev |test|
|--|--|--|
|LR + CV|0.85|0.75|
|BERT|0.86|0.80|


## Future Work

- Preprocess Devnagri text and check the results
- Finetuning models such as XLM-R,m-BERT for performance on cross-lingual data
