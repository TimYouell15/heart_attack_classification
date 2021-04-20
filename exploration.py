# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 20:29:29 2021

@author: YOUELLT
"""

import os
import sys
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


file_dir = os.getcwd() + r'\Documents\Coding\heart_attack_classification'
sys.path.append(file_dir)

df_csv = os.path.join(file_dir, "data/heart.csv")
df = pd.read_csv(df_csv)

df.columns.tolist()

new_cols = {'cp': 'chest_pain', 'trtbps': 'rest_blood_pressure',
            'chol': 'cholestrol', 'fbs': 'fasting_blood_sugar',
            'restecg': 'resting_ecg', 'thalachh': 'max_heart_rate',
            'exng': 'exer_induced_angina', 'oldpeak': 'old_peak',
            'slp': 'st_slope', 'caa': 'num_major_vessels',
            'thall': 'thalassemia', 'output': 'heart_attack'}
df.rename(columns=new_cols, inplace=True)

'''
age : Age of the patient
sex : Sex of the patient
chest_pain : chest pain (scored)
rest_blood_pressure : resting blood pressure (in mm Hg)
cholestrol : cholestoral in mg/dl
fasting_blood_sugar : fasting blood sugar
resting_ecg : resting electrocardiographic results
max_heart_rate : Maximum heart rate
exer_induced_angina : exercise induced angina (chest pain) (yes or no)
old_peak : Previous peak
st_slope : Slope
num_major_vessels : number of major vessels
thalassemia : Thalassemia is an inherited blood disorder in which the body
              makes an abnormal form or inadequate amount of hemoglobin.
heart_attack : 0 = less chance of heart attack 1 = more chance of heart attack
'''

print("The shape of the dataset is: ", df.shape)

print(df.isnull().sum())

df.info()

df_describe = df.describe()


# quick barplot of target
objects = ('1', '0')
y_pos = np.arange(len(objects))
heart_attack = list(df['heart_attack'].value_counts())
plt.bar(y_pos, heart_attack, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Count')
plt.title('Heart Attack Counts')
plt.show()

# pearson correlation plot
corrPearson = df.corr(method="pearson")
figure = plt.figure(figsize=(10, 8))
sns.heatmap(corrPearson, annot=True, cmap='RdYlGn', vmin=-1, vmax=+1)
plt.title("PEARSON")
plt.xlabel("COLUMNS")
plt.ylabel("COLUMNS")
plt.show()

# histograms of features
df.hist(figsize=(20, 10))
plt.show()


def boxplot(feature):
    sns.boxplot(y=feature, x="heart_attack", data=df)
    plt.show()


boxplot('rest_blood_pressure')
boxplot('cholestrol')
boxplot('thalassemia')
boxplot('old_peak')
boxplot('age')


def barplot(feature):
    sns.barplot(x=feature, y="heart_attack", data=df)
    plt.show()


barplot('sex')
barplot('chest_pain')
barplot('fasting_blood_sugar')
barplot('resting_ecg')
barplot('exer_induced_angina')
barplot('st_slope')
barplot('thalassemia')


sns.regplot(x=df['age'], y=df['heart_attack'])
sns.regplot(x=df['chest_pain'], y=df['heart_attack'])
sns.regplot(x=df['cholestrol'], y=df['heart_attack'])



