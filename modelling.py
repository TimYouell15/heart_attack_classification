# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 21:06:13 2021

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

