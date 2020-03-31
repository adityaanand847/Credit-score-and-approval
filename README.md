# credit-card-score-and-approval
Prediction of credit score and approval using several (Random Forest, SVM and Logistic Regression) machine learning algorithm and comparing them to get best accuracy score.

This is a Machine learning project on prediction of credit card approval using some essential features of the users.
There are two csv file in data.rar, namely: credit_card_details and credit_record in which the first one contains the user information
and later one contains the time when user issued the card and status of his credit account.

The columns of the credit_card_details is described below:

Feature name 	Explanation 	Remarks
ID 	==client number 	
'CODE_GENDER' ==	gender 	
'FLAGOWNCAR' 	==Is there a car 	
'FLAGOWNREALTY' ==	Is there a property 	
'CNT_CHILDREN'== 	Number of children 	
'AMTINCOMETOTAL' ==	Annual income 	
'NAMEINCOMETYPE' ==	Income category 	
'NAMEEDUCATIONTYPE' ==	education level 	
'NAMEFAMILYSTATUS' ==	Marital status 	
'NAMEHOUSINGTYPE' ==	Way of living 	
'DAYS_BIRTH' ==	birthday 	
'DAYS_EMPLOYED' ==	Start date 	
'FLAG_MOBIL' 	==Is there a mobile phone 	
'FLAGWORKPHONE' ==	Is there a work phone 	
'FLAG_PHONE' ==	Is there a phone 	
'FLAG_EMAIL' ==	Is there an email 	
'OCCUPATION_TYPE' ==	Occupation 	
'CNTFAMMEMBERS' ==	Family size

The columns of credt_record is described below:

ID == client number 	
MONTHS_BALANCE ==	record month ==	The month of the extracted data is the starting point, backwards, 0 is the current month, -1 is the previous month, and so on
STATUS ==	0: 1-29 days past due 1: 30-59 days past due 2: 60-89 days overdue 3: 90-119 days overdue 4: 120-149 days overdue 5: Overdue or bad debts, write-offs for more than 150 days C: paid off that month X: No loan for the month

Steps to run the code:
1. unpack the data.rar in a folder, say 'data'.
2. Then copy the model.py in the same 'data' folder.
3. Run the model.py on machine.

Requirement:
Python3, Anaconda, basic libraries like Numpy, Pandas and Sklearn
