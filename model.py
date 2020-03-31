import numpy as np
import scipy as sp
import pandas as pd
import sklearn

data = pd.read_csv("credit_card_details.csv", encoding = 'utf-8') 
record = pd.read_csv("credit_record.csv", encoding = 'utf-8')
data.dropna(), record.dropna()
data.rename(columns={'CODE_GENDER':'Gender','FLAG_OWN_CAR':'Car','FLAG_OWN_REALTY':'Reality',
                         'CNT_CHILDREN':'ChldNo','AMT_INCOME_TOTAL':'inc',
                         'NAME_EDUCATION_TYPE':'edutp','NAME_FAMILY_STATUS':'famtp',
                        'NAME_HOUSING_TYPE':'houtp','FLAG_EMAIL':'email',
                         'NAME_INCOME_TYPE':'inctp','FLAG_WORK_PHONE':'wkphone',
                         'FLAG_PHONE':'phone','CNT_FAM_MEMBERS':'famsize',
                        'OCCUPATION_TYPE':'occyp'
                        },inplace=True)


record['STATUS'] = record['STATUS'].replace(['X','C'],[1,0])
data.dropna()
record.head()
record['STATUS'] = record['STATUS'].replace(['1','2','3','4','5'],[1.2,2.8,4.8,7.2,10.0]) ##penalising users for late deposition using i*(i+0.2*i) for due month 'i'

r1=record[['ID','STATUS']]
r2=record[['ID','MONTHS_BALANCE']]

r_1=r1.groupby("ID").mean()
r_1['STATUS'][r_1['STATUS'] <= 1]=1
r_1['STATUS'][r_1['STATUS'] > 1]=0
r_2=r2.groupby("ID").min()

new_record=pd.merge(r_1,r_2,how="left",on="ID")
new_data=pd.merge(data,new_record,how="inner",on="ID")


## forming function that calculate Infromation gain of each feature
##which help in knowing the favourable variable that can be used for modelling
def info_gain(df, feature, target, pr=False):
    lst = []
    df[feature] = df[feature].fillna("NULL")

    for i in range(df[feature].nunique()):
        val = list(df[feature].unique())[i]
        lst.append([feature,
                    val, 
                    df[feature][df[feature] == val].count(),
                    df[feature][(df[feature] == val) & (df[target] == 1)].count(),
                    df[feature][(df[feature] == val) & (df[target] == 0)].count()])
    data1 = pd.DataFrame(lst, columns=['Variable', 'Value', 'All', 'Good', 'Bad'])
    data1['prob_good'] = data1['Good'] / data1['All']
    data1['prob_bad'] = data1['Bad'] / data1['All']
    data1['prob_variable'] = data1['All'] / data1['All'].sum()
    data1['entropy_value'] = -(data1['prob_good']*np.log2(data1['prob_good'])+data1['prob_bad']*np.log2(data1['prob_bad']))*data1['prob_variable']
    data1['IG']=1-data1['entropy_value'].sum()
    Entropy_table= print("The Entropy table for " + feature + " is:"),print(data1), print('\n Information Gain of ' + feature + ' is:')
    IG_value=data1['IG'].mean()
    return Entropy_table, IG_value


##forming function to create dummy value of each feature
##in term of zero and one for each feature that is splitted in more than two leaf
def convert_dummy(df, feature,rank=0):
    pos = pd.get_dummies(df[feature], prefix=feature)
    mode = df[feature].value_counts().index[rank]
    biggest = feature + '_' + str(mode)
    pos.drop([biggest],axis=1,inplace=True)
    df.drop([feature],axis=1,inplace=True)
    df=df.join(pos)
    return df


#forming function to form different bins/feature on the basis of regularisation of continuous data
## like age, income which is divided in bins (as like histogram division)
def get_category(df, col, binsnum, labels, qcut = False):
    if qcut:
        localdf = pd.qcut(df[col], q = binsnum, labels = labels) # quantile cut
    else:
        localdf = pd.cut(df[col], bins = binsnum, labels = labels) # equal-length cut
        
    localdf = pd.DataFrame(localdf)
    name = 'gp' + '_' + col
    localdf[name] = localdf[col]
    df = df.join(localdf[name])
    df[name] = df[name].astype(object)
    return df


##Features extraction of all:
IG_TABLE={'Variable':'IG'}

#1. Gender
new_data['Gender'] = new_data['Gender'].replace(['F','M'],[0,1])
Entropy_table, IG_value= info_gain(new_data,'Gender','STATUS')
print(IG_value)
IG_TABLE['Gender']=IG_value

#2. Reality
new_data['Reality'] = new_data['Reality'].replace(['N','Y'],[0,1])
Entropy_table, IG_value= info_gain(new_data,'Reality','STATUS')
print(IG_value)
IG_TABLE['Reality']=IG_value

#3. Car
new_data['Car'] = new_data['Car'].replace(['N','Y'],[0,1])
Entropy_table, IG_value= info_gain(new_data,'Car','STATUS')
print(IG_value)
IG_TABLE['Car']=IG_value

#4. Phone
Entropy_table, IG_value= info_gain(new_data,'phone','STATUS')
print(IG_value)
IG_TABLE['Phone']=IG_value

#5. Email
Entropy_table, IG_value= info_gain(new_data,'email','STATUS')
print(IG_value)
IG_TABLE['Email']=IG_value

#6. Work Phone
Entropy_table, IG_value= info_gain(new_data,'wkphone','STATUS')
print(IG_value)
IG_TABLE['Work Phone']=IG_value

#7. Children
new_data['ChldNo'][new_data['ChldNo']>=2]='More than 2'
print((new_data['ChldNo'].value_counts()))
Entropy_table, IG_value= info_gain(new_data,'ChldNo','STATUS')
print(IG_value)
IG_TABLE['Child Number']=IG_value
new_data = convert_dummy(new_data,'ChldNo')

#8. Annual Income
new_data['inc'] = new_data['inc']/10000 
print(new_data['inc'].value_counts(bins=10,sort=False))
new_data = get_category(new_data,'inc', 3, ["low","medium", "high"], qcut = True)
Entropy_table, IG_value= info_gain(new_data,'gp_inc','STATUS')
print(IG_value)
IG_TABLE['Annual Income']=IG_value
new_data = convert_dummy(new_data,'gp_inc')

#9. Age
new_data['Age']=-(new_data['DAYS_BIRTH'])//365	
print(new_data['Age'].value_counts(bins=10,normalize=True,sort=False))
new_data = get_category(new_data,'Age',5, ["lowest","low","medium","high","highest"])
Entropy_table, IG_value= info_gain(new_data,'gp_Age','STATUS')
print(IG_value)
IG_TABLE['Age']=IG_value
new_data = convert_dummy(new_data,'gp_Age')


#10. Family Size
new_data['famsize'].value_counts(sort=False)
new_data['famsize'][new_data['famsize']>=2]='More than 2'
Entropy_table, IG_value= info_gain(new_data,'famsize','STATUS')
print(IG_value)
IG_TABLE['famsize']=IG_value
new_data = convert_dummy(new_data,'famsize')


#11. Income type
new_data['inctp'].value_counts(sort=False)
new_data['inctp'][new_data['inctp']=='Pensioner']='State servant'
new_data['inctp'][new_data['inctp']=='Student']='State servant'
Entropy_table, IG_value= info_gain(new_data,'inctp','STATUS')
print(IG_value)
IG_TABLE['income type']=IG_value
new_data = convert_dummy(new_data,'inctp')

#12. House type
Entropy_table, IG_value= info_gain(new_data,'houtp','STATUS')
print(IG_value)
IG_TABLE['house type']=IG_value
new_data = convert_dummy(new_data,'houtp')

#13. Education type
new_data['edutp'].value_counts(sort=False)
new_data['edutp'][new_data['edutp']=='Academic degree']='Higher education'
Entropy_table, IG_value= info_gain(new_data,'edutp','STATUS')
print(IG_value)
IG_TABLE['Education type']=IG_value
new_data = convert_dummy(new_data,'edutp')

#14. Family type
new_data['famtp'].value_counts(sort=False)
Entropy_table, IG_value= info_gain(new_data,'famtp','STATUS')
print(IG_value)
IG_TABLE['Family type']=IG_value
new_data = convert_dummy(new_data,'famtp')

new_data.dropna()
print(pd.Series(IG_TABLE))
print(new_data.columns)

##MODELS
y = new_data[['STATUS']]
x = new_data[[ 'Car', 'Reality', 'inc', 'Age', 'DAYS_EMPLOYED',
        'wkphone', 'phone', 'email', 'inctp_Commercial associate', 'inctp_State servant',
       'houtp_Co-op apartment', 'houtp_Municipal apartment',
       'houtp_Office apartment', 'houtp_Rented apartment',
       'houtp_With parents', 'edutp_Higher education',
       'edutp_Incomplete higher', 'edutp_Lower secondary']]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y)

##SVM
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=50)
model.fit(x_train, y_train.values.ravel())
y_pred=model.predict(x_test)
from sklearn import metrics
print('Accuracy of Random Forest is:', metrics.accuracy_score(y_pred,y_test))

##Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train.values.ravel())
y_pred=model.predict(x_test)
from sklearn import metrics
print('Accuracy of Logistic Regression is:', metrics.accuracy_score(y_pred,y_test))

##SVM
from sklearn.svm import SVC
model= SVC()
model.fit(x_train, y_train.values.ravel())
y_pred=model.predict(x_test)
from sklearn import metrics
print('Accuracy of SVM is:', metrics.accuracy_score(y_pred,y_test))