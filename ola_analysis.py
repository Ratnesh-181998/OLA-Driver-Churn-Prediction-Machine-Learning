"""
OLA Driver Churn Analysis - Ensemble Learning
Extracted from Jupyter Notebook
"""


# ===== Code Cell 1 =====
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib import figure

import statsmodels.api as sm
from scipy.stats import norm
from scipy.stats import t

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# ===== Code Cell 2 =====
ola = pd.read_csv("ola_driver_scaler.csv")

# ===== Code Cell 3 =====
ola.head(5)

# ===== Code Cell 4 =====
df = ola.copy()

# ===== Code Cell 5 =====
(df.isna().sum()/len(df))*100

# ===== Code Cell 6 =====
df.head(10)

# ===== Code Cell 7 =====
df.shape

# ===== Code Cell 8 =====
df["Driver_ID"].nunique()  # 2381 drivers data. 


# ===== Code Cell 9 =====
df.drop(["Unnamed: 0"],axis = 1 , inplace=True)

# ===== Code Cell 10 =====
df["Gender"].replace({0.0:"Male",1.0:"Female"},inplace=True)

# ===== Code Cell 11 =====
df[df["Driver_ID"]==25]

# ===== Code Cell 12 =====
agg_df = df.groupby(["Driver_ID"]).aggregate({'MMM-YY':len,
                                     "Age":max,
                                     
                                     "City":np.unique,
                                    "Education_Level":max,
                                     "Income":np.mean,
                                     "Dateofjoining":np.unique, 
#                                     "LastWorkingDate":last_value,
                                     "Joining Designation":np.unique,
                                     "Grade": np.mean,
                                    "Total Business Value":sum, 
                                     "Quarterly Rating":np.mean
                                     
                                    })

# ===== Code Cell 13 =====
agg_df = agg_df.reset_index()


# ===== Code Cell 14 =====
final_data = agg_df.rename(columns={"MMM-YY":"No_of_Records",
                      "Dateofjoining":"Date_of_joining",
                      "Joining Designation":"Joining_Designation",
                      "Total Business Value" : "Total_Business_Value",
                      "Quarterly Rating":"Quarterly_Rating"})

# ===== Code Cell 15 =====
final_data

# ===== Code Cell 16 =====
final_data = pd.merge(left = df.groupby(["Driver_ID"])["LastWorkingDate"].unique().apply(lambda x:x[-1]),
        right = final_data,
         on = "Driver_ID",
         how="outer"
    )

# ===== Code Cell 17 =====
final_data = pd.merge(left = df.groupby(["Driver_ID"])["Gender"].unique().apply(lambda x:x[-1]),
        right = final_data,
         on = "Driver_ID",
         how="outer"
    )

# ===== Code Cell 18 =====
data = final_data.copy()

# ===== Code Cell 19 =====
data["Gender"].value_counts()

# ===== Code Cell 20 =====
pd.Series(np.where(data["LastWorkingDate"].isna(),0,1)).value_counts()

# ===== Code Cell 21 =====
data["Churn"] = data["LastWorkingDate"].fillna(0)

# ===== Code Cell 22 =====
def apply_0_1(y):
    if y == 0:
        return 0
    if y != 0:
        return 1
    

# ===== Code Cell 23 =====
data["Churn"] = data["Churn"].apply(apply_0_1)

# ===== Code Cell 24 =====
data["Churn"].value_counts()

# ===== Code Cell 25 =====
data["Churn"].value_counts(normalize=True)*100

# ===== Code Cell 26 =====
# data["Total_Business_Value"] = data["Total_Business_Value"].replace({0:np.nan})

# ===== Code Cell 27 =====
data["Date_of_joining"] = pd.to_datetime(data["Date_of_joining"])
data["LastWorkingDate"] = pd.to_datetime(data["LastWorkingDate"])

# ===== Code Cell 28 =====
data["joining_Year"] = data["Date_of_joining"].dt.year


# ===== Code Cell 29 =====
#  data["joining_month"] = data["Date_of_joining"].dt.month

# ===== Code Cell 30 =====
(data.isna().sum()/len(data))*100

# ===== Code Cell 31 =====
data["Churn"].value_counts(normalize=True)*100

# ===== Code Cell 32 =====
def app_rating_inc(y):
    
    if len(y)>=2:
        for i in range(len(y)):
            if y[-1]>y[-2]:
                return 1
            else:
                return 0
    else:
        return 0

# ===== Code Cell 33 =====
Quarterly_Rating_increased = df.groupby("Driver_ID")["Quarterly Rating"].unique().apply(app_rating_inc)

# ===== Code Cell 34 =====
data = pd.merge(left = Quarterly_Rating_increased,
        right = data,
         on = "Driver_ID",
         how="outer"
    )

# ===== Code Cell 35 =====
# df.groupby("Driver_ID")["Quarterly Rating"].unique().apply(app_rating_inc)

# ===== Code Cell 36 =====
data["Quarterly_Rating_increased"] = data["Quarterly Rating"]

# ===== Code Cell 37 =====
data.drop(["Quarterly Rating"],axis=1,inplace=True)

# ===== Code Cell 38 =====
def app_income_inc(y):
    
    if len(y)>=2:
        for i in range(len(y)):
            if y[-1]>y[-2]:
                return 1
            else:
                return 0
    else:
        return 0

# ===== Code Cell 39 =====
# df.groupby("Driver_ID")["Income"].unique().apply(app_income_inc).rename("Increased_Income")

# ===== Code Cell 40 =====
data = pd.merge(left = df.groupby("Driver_ID")["Income"].unique().apply(app_income_inc).rename("Increased_Income"),
        right = data,
         on = "Driver_ID",
         how="outer"
    )

# ===== Code Cell 41 =====
data

# ===== Code Cell 42 =====
Mdata = data.copy()

# ===== Code Cell 43 =====
Mdata["Gender"].replace({"Male":0,
                       "Female":1},inplace=True)

# ===== Code Cell 44 =====
Mdata.drop(["Driver_ID"],axis = 1, inplace=True)

# ===== Code Cell 45 =====
Mdata.isna().sum()

# ===== Code Cell 46 =====
Mdata

# ===== Code Cell 47 =====
pd.to_datetime("2021-06-01")

# ===== Code Cell 48 =====
Mdata["LastWorkingDate"] = Mdata["LastWorkingDate"].fillna(pd.to_datetime("2021-06-01"))

# ===== Code Cell 49 =====
(Mdata["LastWorkingDate"] - Mdata["Date_of_joining"])

# ===== Code Cell 50 =====
Mdata["Driver_tenure_days"] = (Mdata["LastWorkingDate"] - Mdata["Date_of_joining"])

# ===== Code Cell 51 =====
Mdata["Driver_tenure_days"] = Mdata["Driver_tenure_days"].dt.days

# ===== Code Cell 52 =====
Mdata.drop(["LastWorkingDate","Date_of_joining"],inplace=True,axis = 1)

# ===== Code Cell 53 =====
Mdata.drop(["Driver_tenure_days"],inplace=True,axis = 1)

# ===== Code Cell 54 =====
Mdata

# ===== Code Cell 55 =====
Mdata.columns

# ===== Code Cell 56 =====
Mdata["Grade"] = np.round(Mdata["Grade"])

# ===== Code Cell 57 =====
Mdata["Quarterly_Rating"]= Mdata["Quarterly_Rating"].round()

# ===== Code Cell 58 =====
categorical_features = ['Increased_Income', 'Gender','City','Education_Level',
                   'Joining_Designation','Grade','Quarterly_Rating','Quarterly_Rating_increased',"joining_Year"]

for col in categorical_features:
    pd.crosstab(index = Mdata[col],
               columns = Mdata["Churn"],
               normalize="columns").plot(kind = "bar")
    plt.show()

# ===== Code Cell 59 =====
Mdata.isna().sum()

# ===== Code Cell 60 =====
from sklearn.impute import SimpleImputer

# ===== Code Cell 61 =====
imputer = SimpleImputer(strategy='most_frequent')

# ===== Code Cell 62 =====
Mdata["Gender"] = imputer.fit_transform(X=Mdata["Gender"].values.reshape(-1,1),y=Mdata["Churn"].values.reshape(-1,1))

# ===== Code Cell 63 =====
Mdata["Gender"].value_counts(dropna=False)

# ===== Code Cell 64 =====
Mdata.isna().sum()

# ===== Code Cell 65 =====
from category_encoders import TargetEncoder
TE = TargetEncoder()

# ===== Code Cell 66 =====
Mdata["City"] = TE.fit_transform(X = Mdata["City"],y = Mdata["Churn"])



# ===== Code Cell 67 =====
Mdata["joining_Year"] = TE.fit_transform(X = Mdata["joining_Year"],y = Mdata["Churn"])


# ===== Code Cell 68 =====
Mdata

# ===== Code Cell 69 =====
# Mdata.drop(["No_of_Records"], axis = 1 , inplace= True)

# ===== Code Cell 70 =====
plt.figure(figsize=(15, 15))
sns.heatmap(Mdata.corr(),annot=True, cmap="RdYlGn", annot_kws={"size":10})

# ===== Code Cell 71 =====
X = Mdata.drop(["Churn"],axis = 1)
y = Mdata["Churn"]

# ===== Code Cell 72 =====
import numpy as np
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)



# ===== Code Cell 73 =====
X = pd.DataFrame(imputer.fit_transform(X),columns=X.columns)

# ===== Code Cell 74 =====
X

# ===== Code Cell 75 =====
X.describe()

# ===== Code Cell 76 =====
from sklearn.model_selection import train_test_split

X_train , X_test, y_train ,y_test = train_test_split(X,y,
                                                    random_state=5,
                                                    test_size=0.2)

# ===== Code Cell 77 =====
y.value_counts()

# ===== Code Cell 78 =====
765 + 1616

# ===== Code Cell 79 =====
from sklearn.preprocessing import StandardScaler

# ===== Code Cell 80 =====
scaler = StandardScaler()

# ===== Code Cell 81 =====
scaler.fit(X_train)


# ===== Code Cell 82 =====
X_train = scaler.transform(X_train) 
X_test =  scaler.transform(X_test)

# ===== Code Cell 83 =====
from sklearn.ensemble import RandomForestClassifier

# ===== Code Cell 84 =====
RF = RandomForestClassifier(n_estimators=100,
    criterion='entropy',
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features='sqrt',
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    bootstrap=True,
    oob_score=False,
    n_jobs=None,
    random_state=None,
    verbose=0,
    warm_start=False,
    class_weight="balanced",
    ccp_alpha=0.0085,
    max_samples=None,)

# ===== Code Cell 85 =====
RF.fit(X_train,y_train)

# ===== Code Cell 86 =====
RF.score(X_train,y_train),RF.score(X_test,y_test)

# ===== Code Cell 87 =====
RF.feature_importances_

# ===== Code Cell 88 =====
X.columns

# ===== Code Cell 89 =====
pd.DataFrame(data=RF.feature_importances_,
            index=X.columns).plot(kind="bar")

# ===== Code Cell 90 =====
from sklearn.metrics import f1_score , precision_score, recall_score,confusion_matrix



# ===== Code Cell 91 =====
confusion_matrix(y_test,RF.predict(X_test) )

# ===== Code Cell 92 =====
confusion_matrix(y_train,RF.predict(X_train) )

# ===== Code Cell 93 =====
f1_score(y_test,RF.predict(X_test)),f1_score(y_train,RF.predict(X_train))

# ===== Code Cell 94 =====
precision_score(y_test,RF.predict(X_test)),precision_score(y_train,RF.predict(X_train))

# ===== Code Cell 95 =====
recall_score(y_test,RF.predict(X_test)),recall_score(y_train,RF.predict(X_train))

# ===== Code Cell 96 =====
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

parameters = {"max_depth":[7,10,15],
             "n_estimators":[100,200,300,400],
             "max_features":[4,7,10],
             "ccp_alpha":[0.0005,0.00075,0.001]}

RFC = RandomForestClassifier()
grid_search = GridSearchCV(
    estimator = RFC,
    param_grid = parameters,
    scoring = "accuracy",
    n_jobs = -1,
    refit=True,                   # need not to train again after grid search
    cv=3,
    pre_dispatch='2*n_jobs',
    return_train_score=False)


# ===== Code Cell 97 =====
grid_search.fit(X_train,y_train.values.ravel())


# ===== Code Cell 98 =====
grid_search.best_estimator_

# ===== Code Cell 99 =====
grid_search.best_score_

# ===== Code Cell 100 =====
grid_search.best_params_

# ===== Code Cell 101 =====
from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=100,
    criterion='entropy',
    max_depth=7,
    min_samples_split=2,
    min_samples_leaf=1,
    
    class_weight="balanced",
    ccp_alpha=0.0001,
    max_samples=None)

# ===== Code Cell 102 =====
RF.fit(X_train , y_train)

# ===== Code Cell 103 =====
RF.score(X_train,y_train),RF.score(X_test,y_test)

# ===== Code Cell 104 =====
y_test_pred = RF.predict(X_test)
y_train_pred = RF.predict(X_train)

# ===== Code Cell 105 =====
f1_score(y_test,y_test_pred),f1_score(y_train,y_train_pred)


# ===== Code Cell 106 =====
precision_score(y_test,y_test_pred),precision_score(y_train,y_train_pred)


# ===== Code Cell 107 =====
recall_score(y_test,y_test_pred),recall_score(y_train,y_train_pred)

# ===== Code Cell 108 =====
from sklearn.tree import DecisionTreeClassifier

# ===== Code Cell 109 =====
from sklearn.ensemble import BaggingClassifier

# ===== Code Cell 110 =====
bagging_classifier_model = BaggingClassifier(base_estimator=  DecisionTreeClassifier(max_depth=7,
                                                                                     class_weight="balanced"),
                                            n_estimators=50,
                                            max_samples=1.0,
                                            max_features=1.0,
                                            bootstrap=True,
                                            bootstrap_features=False,
                                            oob_score=False,
                                            warm_start=False,
                                            n_jobs=None,
                                            random_state=None,
                                            verbose=0,)

# ===== Code Cell 111 =====
bagging_classifier_model.fit(X_train,y_train)

# ===== Code Cell 112 =====

from sklearn.metrics import f1_score , precision_score, recall_score,confusion_matrix


# ===== Code Cell 113 =====
y_test_pred = bagging_classifier_model.predict(X_test)
y_train_pred = bagging_classifier_model.predict(X_train)

# ===== Code Cell 114 =====

confusion_matrix(y_test,y_test_pred)

# ===== Code Cell 115 =====

confusion_matrix(y_train,y_train_pred)

# ===== Code Cell 116 =====
f1_score(y_test,y_test_pred),f1_score(y_train,y_train_pred)


# ===== Code Cell 117 =====
precision_score(y_test,y_test_pred),precision_score(y_train,y_train_pred)


# ===== Code Cell 118 =====

recall_score(y_test,y_test_pred),recall_score(y_train,y_train_pred)

# ===== Code Cell 119 =====
bagging_classifier_model.score(X_test,y_test)

# ===== Code Cell 120 =====
bagging_classifier_model.score(X_train,y_train)

# ===== Code Cell 121 =====
# !pip install xgboost

# ===== Code Cell 122 =====
from xgboost import XGBClassifier

# ===== Code Cell 123 =====
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

parameters = {"max_depth":[2,4,6,10],
             "n_estimators":[100,200,300,400]    }


grid_search = GridSearchCV(
    estimator = XGBClassifier(),
    param_grid = parameters,
    scoring = "accuracy",
    n_jobs = -1,
    refit=True,                   # need not to train again after grid search
    cv=3,
    pre_dispatch='2*n_jobs',
    return_train_score=False)


grid_search.fit(X_train,y_train.values.ravel())

grid_search.best_estimator_

grid_search.best_score_

grid_search.best_params_

# ===== Code Cell 124 =====
xgb = XGBClassifier(n_estimators=100,
                   max_depth = 2)
xgb.fit(X_train, y_train)

# ===== Code Cell 125 =====
y_test_pred = xgb.predict(X_test)
y_train_pred = xgb.predict(X_train)

# ===== Code Cell 126 =====
confusion_matrix(y_test,y_test_pred)

# ===== Code Cell 127 =====
confusion_matrix(y_train,y_train_pred)

# ===== Code Cell 128 =====
xgb.score(X_train,y_train),xgb.score(X_test,y_test)

# ===== Code Cell 129 =====
f1_score(y_test,y_test_pred),f1_score(y_train,y_train_pred)


# ===== Code Cell 130 =====

recall_score(y_test,y_test_pred),recall_score(y_train,y_train_pred)

# ===== Code Cell 131 =====
precision_score(y_test,y_test_pred),precision_score(y_train,y_train_pred)


# ===== Code Cell 132 =====
xgb.feature_importances_

# ===== Code Cell 133 =====
pd.DataFrame(data=xgb.feature_importances_,
            index=X.columns).plot(kind="bar")

# ===== Code Cell 134 =====
def GradientBoostingClassifier(X, y):
    from sklearn.ensemble import  GradientBoostingClassifier
    from sklearn.metrics import f1_score, accuracy_score , roc_auc_score,auc,recall_score,precision_score
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=1)
    
    lr = GradientBoostingClassifier()
    scaler = StandardScaler()
    scaler.fit(X_train) 
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    prob = lr.predict_proba(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print('Train Score : ', lr.score(X_train, y_train), '\n')
    print('Test Score : ', lr.score(X_test, y_test), '\n')
    print('Accuracy Score : ', accuracy_score(y_test, y_pred), '\n')
    print(cm, "---> confusion Matrix ", '\n')
    print("ROC-AUC score  test dataset:  ", roc_auc_score(y_test, prob[:, 1]),'\n')
    print("precision score  test dataset:  ", precision_score(y_test, y_pred),'\n')
    print("Recall score  test dataset:  ", recall_score(y_test, y_pred), '\n')
    print("f1 score  test dataset :  ", f1_score(y_test, y_pred), '\n')
    return (prob[:,1], y_test)

# ===== Code Cell 135 =====
probs , y_test = GradientBoostingClassifier(X,y)

# ===== Code Cell 136 =====
def plot_pre_curve(y_test,probs):
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    plt.plot([0, 1], [0.5, 0.5], linestyle='--')
    # plot the precision-recall curve for the model
    plt.plot(recall, precision, marker='.')
    plt.title("Precision Recall curve")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the plot
    plt.show()
    
def plot_roc(y_test,prob):
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    # plot no skill
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.')
    plt.title("ROC curve")
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    # show the plot
    plt.show()

# ===== Code Cell 137 =====
plot_roc(y_test , probs)

# ===== Code Cell 138 =====
plot_pre_curve(y_test , probs)
