#Open Dataset
import pandas as pd

#Import Data - Can be found in main Data folder in Github
cola = pd.read_csv('combined_cola.csv',header=0)
sst2 = pd.read_csv('NLP - Ensemble Data - combined_sst2.csv',header=0)
wnli = pd.read_csv('NLP - Ensemble Data - combined_wnli.csv',header=0)
rte = pd.read_csv('NLP - Ensemble Data - combined_rte.csv',header=0)
qnli = pd.read_csv('NLP - Ensemble Data - combined_qnli.csv',header=0)
wnli = pd.read_csv('NLP - Ensemble Data - combined_wnli.csv',header=0)
mrpc = pd.read_csv('NLP - Ensemble Data - combined_mrpc.csv',header=0)
stsb = pd.read_csv('NLP - Ensemble Data - combined_stsb.csv',header=0)
qqp = pd.read_csv('NLP - Ensemble Data - combined_qqp.csv',header=0)
mnli = pd.read_csv('NLP - Ensemble Data - combined_mnli.csv',header=0)
print(cola.head())


#Ensemble Functions

# Import sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LinearRegression


# logistic regression function to train and make predictions
def LR(data):
    X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1),
                                                        data['target'], test_size=0.20, random_state=2000)
    model = LogisticRegression(solver='liblinear', random_state=0).fit(X_train, y_train)
    pred = model.predict(X_test)
    return pred, y_test


# RandomForestClassifer function to train and make predictions
def RFC(data):
    X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1),
                                                        data['target'], test_size=0.20, random_state=2000)
    clf = RandomForestClassifier(max_depth=2, random_state=0).fit(X_train, y_train)
    pred_clf = clf.predict(X_test)
    return pred_clf, y_test


def LinReg(data):
    X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1),
                                                        data['target'], test_size=0.20, random_state=2000)
    model = LinearRegression().fit(X_train, y_train)
    pred = model.predict(X_test)
    return pred, y_test


#CoLA Classification Report using Logistic Regression
lr_cola_pred, lr_cola_y_test = LR(cola)
print(classification_report(lr_cola_y_test,lr_cola_pred))

#CoLA Mathew's Correlation Coefficient using Logistic Regression Predictions
print(f"Using Logistic Regression Ensemble: \n\nMatthew's Correlation Coefficient: {matthews_corrcoef(lr_cola_pred, lr_cola_y_test)}")

#CoLA Classification Report using RandomForestClassifier
rfc_cola_pred, rfc_cola_y_test = RFC(cola)
print(classification_report(rfc_cola_y_test, rfc_cola_pred))

#CoLA Mathew's Correlation Coefficient using RandomForestClassifier Predictions
print(f"Using RandomForestClassifer Ensemble: \n\nMatthew's Correlation Coefficient: {matthews_corrcoef(rfc_cola_pred, rfc_cola_y_test)}")

#SST2 Classification Report using Logistic Regression
lr_sst2_pred, lr_sst2_y_test = LR(sst2)
print(classification_report(lr_sst2_y_test, lr_sst2_pred))

#SST2 Accuracy Score using Logistic Regression Predictions
print(f"Using Logistic Regression Ensemble: \n\nAccuracy Score: {accuracy_score(lr_sst2_y_test, lr_sst2_pred)}")

#SST2 Classification Report using RandomForestClassificatier Predictions
rfc_sst2_pred, rfc_sst2_y_test = RFC(sst2)
print(classification_report(rfc_sst2_y_test, rfc_sst2_pred))

#SST-2 Accuracy Score using Random Forest Classifier
print(f"Using RandomForestClassifier Ensemble: \n\nAccuracy Score: {accuracy_score(rfc_sst2_y_test, rfc_sst2_pred)}")

#MRPC Classification Report using Logistic Regression
lr_mrpc_pred, lr_mrpc_y_test = LR(mrpc)
print(classification_report(lr_mrpc_y_test, lr_mrpc_pred))

#MRPC Accuracy/F1 Score using Logistic Regression Predictions
lr_mrpc_acc = accuracy_score(lr_mrpc_y_test, lr_mrpc_pred)
lr_mrpc_f1 = f1_score(lr_mrpc_y_test, lr_mrpc_pred)
print(f"Using Logistic Regression Predictions: \n\nAccuracy Score: {lr_mrpc_acc} \nF1 Score: {lr_mrpc_f1}")

#MRPC Classification Report using RandomForestClassificatier Predictions
rfc_mrpc_pred, rfc_mrpc_y_test = RFC(mrpc)
print(classification_report(rfc_mrpc_y_test, rfc_mrpc_pred))

#MRPC Accuracy/F1 Score using RandomForestClassifier Predictions
rfc_mrpc_acc = accuracy_score(rfc_mrpc_y_test, rfc_mrpc_pred)
rfc_mrpc_f1_score = f1_score(rfc_mrpc_y_test, rfc_mrpc_pred)
print(f"Using RandomForestClassifier Predictions: \n\nAccuracy Score: {rfc_mrpc_acc} \nF1 Score: {rfc_mrpc_f1_score}")

from scipy import stats

#STSB Pearson-Spearman Correlation Coefficient using Linear Regression
linreg_stsb_pred, linreg_stsb_y_test = LinReg(stsb)
stats.spearmanr(linreg_stsb_pred, linreg_stsb_y_test)

#STSB Pearson-Spearman Correlation Coefficient using Linear Regression
linreg_stsb_pred, linreg_stsb_y_test = LinReg(stsb)
stats.pearsonr(linreg_stsb_pred, linreg_stsb_y_test)

#QQP Classification Report using Logistic Regression
lr_qqp_pred, lr_qqp_y_test = LR(qqp)
print(classification_report(lr_qqp_y_test, lr_qqp_pred))

#QQP Accuracy/F1 Score using Logistic Regression Predictions
lr_qqp_acc = accuracy_score(lr_qqp_y_test, lr_qqp_pred)
lr_qqp_f1 = f1_score(lr_qqp_y_test, lr_qqp_pred)
print(f"Using Logistic Regression Predictions: \n\nAccuracy Score: {lr_qqp_acc} \nF1 Score: {lr_qqp_f1}")

#QQP Classification Report using RandomForestClassificatier Predictions
rfc_qqp_pred, rfc_qqp_y_test = RFC(qqp)
print(classification_report(rfc_qqp_y_test, rfc_qqp_pred))

#QQP Accuracy/F1 Score using RandomForestClassifier Predictions
rfc_qqp_acc = accuracy_score(rfc_qqp_y_test, rfc_qqp_pred)
rfc_qqp_f1_score = f1_score(rfc_qqp_y_test, rfc_qqp_pred)
print(f"Using RandomForestClassifier Predictions: \n\nAccuracy Score: {rfc_qqp_acc} \nF1 Score: {rfc_qqp_f1_score}")

#MNLI Classification Report using Logistic Regression
lr_mnli_pred, lr_mnli_y_test = LR(mnli)
print(classification_report(lr_mnli_y_test, lr_mnli_pred))

#MNLI Accuracy Score using Logistic Regression
lr_mnli_acc = accuracy_score(lr_mnli_y_test, lr_mnli_pred)
print(f"Logistic Regression Ensemble: \n\nAccuracy Score: {lr_mnli_acc}")

#MNLI Classification Report RandomForestClassificatier Predictions
rfc_mnli_pred, rfc_mnli_y_test = RFC(mnli)
print(classification_report(rfc_mnli_y_test, rfc_mnli_pred))

#MNLI Accuracy Score using Random Forest Classification
rfc_mnli_acc = accuracy_score(rfc_mnli_y_test, rfc_mnli_pred)
print(f"Random Forest Classification Ensemble: \n\nAccuracy Score: {rfc_mnli_acc}")

#QNLI Classification Report using Logistic Regression
lr_qnli_pred, lr_qnli_y_test = LR(qnli)
print(classification_report(lr_qnli_y_test, lr_qnli_pred))

#QNLI Accuracy Score using Logistic Regression
lr_qnli_acc = accuracy_score(lr_qnli_y_test, lr_qnli_pred)
print(f"Logistic Regression Ensemble: \n\nAccuracy Score: {lr_qnli_acc}")

#QNLI Classification Report RandomForestClassificatier Predictions
rfc_qnli_pred, rfc_qnli_y_test = RFC(qnli)
print(classification_report(rfc_qnli_y_test, rfc_qnli_pred))

#QNLI Accuracy Score using Random Forest Classification
rfc_qnli_acc = accuracy_score(rfc_qnli_y_test, rfc_qnli_pred)
print(f"Random Forest Classification Ensemble: \n\nAccuracy Score: {rfc_qnli_acc}")

#RTE Classification Report using Logistic Regression
lr_rte_pred, lr_rte_y_test = LR(rte)
print(classification_report(lr_rte_y_test, lr_rte_pred))

#RTE Accuracy Score using Logistic Regression
print(f"Logistic Regression Ensemble: \n\nAccuracy Score: {accuracy_score(lr_rte_y_test, lr_rte_pred)} ")

#RTE Classification Report using RandomForestClassificatier Predictions
rfc_rte_pred, rfc_rte_y_test = RFC(rte)
print(classification_report(rfc_rte_y_test, rfc_rte_pred))

#RTE Accuracy Score using RandomForestClassifier
print(f"Random Forest Classifier Ensemble: \n\nAccuracy Score: {accuracy_score(rfc_rte_y_test, rfc_rte_pred)} ")

#WNLI Classification Report using Logistic Regression
lr_wnli_pred, lr_wnli_y_test = LR(wnli)
print(classification_report(lr_wnli_y_test, lr_wnli_pred))

#WNLI Accuracy Score
wnli_lr_accuracy = accuracy_score(lr_wnli_y_test, lr_wnli_pred)
print(f"Accuracy Score: {wnli_lr_accuracy}")

#WNLI Classification Report using RandomForestClassificatier Predictions
rfc_wnli_pred, rfc_wnli_y_test = RFC(wnli)
print(classification_report(rfc_wnli_y_test, rfc_wnli_pred))

#WNLI Accuracy Score
wnli_rfc_accuracy = accuracy_score(rfc_wnli_y_test, rfc_wnli_pred)
print(f"Accuracy Score: {wnli_rfc_accuracy}")