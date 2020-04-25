# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import re
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from datetime import datetime
from fbprophet import Prophet

### 気温データ読み込み
y19531962 = pd.read_csv("data/y19531962.csv",encoding="SHIFT-JIS")
y19631972 = pd.read_csv("data/y19631972.csv",encoding="SHIFT-JIS")
y19731982 = pd.read_csv("data/y19731982.csv",encoding="SHIFT-JIS")
y19831992 = pd.read_csv("data/y19831992.csv",encoding="SHIFT-JIS")
y19932002 = pd.read_csv("data/y19932002.csv",encoding="SHIFT-JIS")
y20032012 = pd.read_csv("data/y20032012.csv",encoding="SHIFT-JIS")
y20132019 = pd.read_csv("data/y20132019.csv",encoding="SHIFT-JIS")

### 開花情報の取得
flower = pd.read_csv("data/train2.csv",encoding="SHIFT-JIS",usecols=[u'徳島','year'])
flower = flower.rename(columns={u'徳島':'tokushima'})

### 学習データの作成
train = pd.concat([y19531962,y19631972,y19731982,y19831992,y19932002,y20032012,y20132019],ignore_index=True)
train = train.rename(columns={u'年月日':'date',u'平均気温(℃)':'tmp',u'最高気温(℃)':'max',u'最低気温(℃)':'min'})

### 年月日に変換
for j, columns in enumerate(flower.columns) :
    if columns != 'year' :
        for i, (md, year) in enumerate(zip(flower[columns],flower['year'])) :
            if isinstance(md,(int,float,str)) is False :
                md = md.encode('utf-8')
            if pd.isnull(md) is False :
                month = re.sub(r'月.*','',md)
                day = re.sub(r'日.*','',re.sub(r'.*月','',md))
                flower.iat[i,j] = str(year) + "/" + str(month) + "/" + str(day)

### 開花判定
train['open'] = int(0)
for i, date in enumerate(train['date']) :
    flag = False
    for open in flower['tokushima'] :
        if date == open :
            flag = True
    if flag is True :
        train.iat[(i,train.columns.get_loc('open'))] = int(1)
    else :
        train.iat[(i,train.columns.get_loc('open'))] = int(0)

## Category Encorder
# for column in ['date']:
#     le = LabelEncoder()
#     le.fit(train[column])
#     train[column] = le.transform(train[column])

### datetime型に変更、その他変更
train['date'] = pd.to_datetime(train['date'], format='%Y/%m/%d')
train['date'] = train['date'].index.get_values()
# train['tmp'] = train['tmp'].astype(float)
# train['max'] = train['max'].astype(float)
# train['min'] = train['min'].astype(float)
# train['open'] = train['open'].astype(int)

### OneHot Encording
# oh_area = pd.get_dummies(train.area)
# train.drop(['area'], axis=1, inplace=True)
# train = pd.concat([train,oh_area], axis=1)
# _, i = np.unique(train.columns, return_index=True)
# train = train.iloc[:, i]

# ### 履歴をシフト
# for i in range(1,6):
#     train['sft%s'%i] = train['tokushima'].shift(i)
#
# ### 1階微分
# train['drv1'] = train['sft1'].diff(1)
#
# ### 2階微分
# train['drv2'] = train['sft1'].diff(1).diff(1)
#
# ### 移動平均値
# train['mean'] = train['sft1'].rolling(6).mean()

### NaNの削除
train = train.dropna()

### int型に変換
# for columns in train.columns :
#     if columns is not 'mean' :
#         train[columns] = train[columns].astype(int)

X = train.drop(['open','min','max'], axis=1)
y = train['open']
print('X shape: {}, y shape: {}'.format(X.shape, y.shape))
print(X.dtypes)


### データセットの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
# X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

print("LinearRegression")
model = LinearRegression()
model.fit(X_train,y_train)
print(model.score(X_train,y_train))

print("LogisticRegression")
model = LogisticRegression()
model.fit(X_train,y_train)
print(model.score(X_train,y_train))

print("SVM")
model = SVC()
model.fit(X_train, y_train)
predicted = model.predict(X_test)
print(metrics.accuracy_score(y_test, predicted))
# print("SVM(GridSearch)")
# best_score = 0
# for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
#     for C in [0.001, 0.01, 0.1, 1, 10, 100]:
#         # print(str(gamma) + "," + str(C))
#         svm = SVC(gamma=gamma, C=C)
#         svm.fit(X_train, y_train.values.ravel())
#         score = svm.score(X_test, y_test)
#         if score > best_score:
#             best_score = score
#             best_parameters = {'C':C, 'gamma':gamma}
# print("Best score: " + str(best_score))
# print("Best parameters: " + str(best_parameters))

print("RandomForest")
model = RandomForest(n_estimators=100).fit(X_train, y_train)
print(model.score(X_test, y_test))

print("LightGBM")
lgb_train = lgb.Dataset(X_train, y_train)
lgb_test  = lgb.Dataset(X_test, y_test, reference=lgb_train)
params = {
        'objective': 'binary',
        'metric': 'auc'
}
evaluation_results = {}
model = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                valid_sets=lgb_test,
                valid_names='test',
                evals_result=evaluation_results,
                early_stopping_rounds=10
                )
y_pred = model.predict(X_test, num_iteration=model.best_iteration)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
print(metrics.auc(fpr, tpr))

# print("prophet")
# train.rename(columns={'date':'ds','open':'y'}, inplace=True)
# train['y'] = train['y'].astype(int)
# print(train)
# # train['origin'] = train['y']
# # train['y'] = np.log(train['y'])
# model = Prophet()
# model.fit(train)
# future_data = model.make_future_dataframe(periods=60, freq='m')
# forecast_data = model.predict(future_data)
# model.plot(forecast_data)
# model.plot_components(forecast_data)
# plt.show()
