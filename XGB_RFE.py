import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# load data
data = pd.read_csv('~/Desktop/Senior/专业综合实践/data_clean.csv')

X = data.drop(columns=['EUA']) # features
y = data['EUA'] # target

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
import numpy as np


# XGBRegressor
xgb_model = XGBRegressor(
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100,
    booster='gbtree',
    random_state=918
)
fold = 4

# RFE CV for feature selection
rfecv = RFECV(
    estimator=xgb_model,
    step=1,
    cv=fold,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    min_features_to_select=2
)

rfecv.fit(X_train, y_train)


# get the mean score for each number of features
mean_test_score = rfecv.cv_results_['mean_test_score']

# get the scores for each fold
cv_folds_scores = [rfecv.cv_results_[f'split{i}_test_score'] for i in range(fold)]  

# draw the scores for each fold
plt.figure(figsize=(8, 6), dpi=200)
for i, cv_score in enumerate(cv_folds_scores):
    plt.plot(range(1, len(cv_score) + 1), cv_score, label=f'CV{i+1}', linestyle='--', marker='o')

# draw the average score
plt.plot(range(1, len(mean_test_score) + 1), mean_test_score, color='black', marker='*', label='Average', linewidth=2)

plt.title('RFECV for XGB', fontsize=16)
plt.xlabel('Number of features selected', fontsize=12)
plt.ylabel('Cross validation score (negative MSE)', fontsize=12)
plt.legend(loc='lower right')
plt.savefig('Senior/专业综合实践/RFECV_XGB.png')

# get the selected features using scorer function
rfecv.n_features_, rfecv.support_, rfecv.ranking_ = rfecv.n_features_, rfecv.support_, rfecv.ranking_
print(f'Number of features selected: {rfecv.n_features_}')
print(f'Selected features: {X.columns[rfecv.support_]}')
print(f'Feature ranking: {rfecv.ranking_}')
