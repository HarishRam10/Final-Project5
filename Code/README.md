# Code Description

## run_glue.py
run_glue.py referenced from the huggingface Github transformers repository that allowed us to run each glue task. Once this file is downloaded on your local machine, the command: 

![image](https://user-images.githubusercontent.com/54903276/145453035-c8e0903d-2804-4fad-9a9c-933d5ce64eaa.png) 

This allowed our group to manipulate the parameters, run the GLUE tasks to output the corresponding metrics and make predictions for each task. This code uses the Trainer class at the base of model building; however, the train, evaluate and prediction functions are custom made.

Reference: https://github.com/huggingface/transformers/tree/master/examples/pytorch/text-classification


## Trainer.py

Trainer.py is another python file that we used to run the GLUE Benchmark tasks. This was also referenced from the huggingface Github notebooks repository. This file has similar functionality to run_glue.py; however, uses the Trainer class to make train, evaluate and make predictions on the model. This code was more simple which allowed to experiment with several transformer models. In this file, the user can define the GLUE task by changing the string variable "task" and initializing the "tokenizer" and "model" variable with the desired model. Once that is done, a Trainer object is instantiated which will run the model. 

https://github.com/huggingface/transformers/tree/master/notebooks

## download_glue_data.py

download_glue_data.py is a python file that can be opened up any Python-usable IDE. This file downloads all the datasets from the 9 tasks on the GLUE Benchmark website. 

## Discriminator.py

Discriminator.py is python file that can be opened up in any Python-usable IDE. This file contains a model class where we modified the transformer models that we focused on, but did not get better results. 

## NLP - Ensemble.py

NLP - Ensemble.pynb is a Jupyter Notebook that can be opened in Google CoLab as well. This file contains the Logistic Regression, Random Forest Classification and Linear Regression models to generate ensemble results. In order to run this file, please keep the csv files in the Data folder in this same folder as NLP - Ensemble.py. If opening in Jupyter Notebook, comment out the first cell block and the first line in the second cell block that defines "abspath_curr." After that, remove "abspath_curr" from the following pd.read_csv() lines.
