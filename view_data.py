from Functions import *
warnings.filterwarnings("ignore")

'''
# number of folds
n_splits = 5

# 1 - WA_Fn-UseC_-Telco-Customer-Churn #########################
#     - Du doan hanh vi de giu chan khach hang                 #
#     - Nhung khach hang bo di - goi la Churn                  #
#     - Khach hang su dung cac dich vu nhu: phone, internet,v.v#
################################################################
#---------------------------- Read file

filename = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
churn_data = pd.read_csv(filename, header=0, engine='python')
remove(filename, '\/:*?"<>|') # remove special characters

##--------------------------- View data info
print('-------------Dataset - Infomation-------------')
print(churn_data.info())
print('-------------Dataset - Description-------------')
print(churn_data.describe())
print('-------------Dataset - Head-------------')
print(churn_data.head())

#############################
###   DATA MANIPULATION   ###
#############################

# 1- Replacing spaces with null values in Total charges column
churn_data['TotalCharges'] = churn_data['TotalCharges'].replace(" ",np.nan)

# 2 - Dropping null values from Total charges column which contain .15% missing data
churn_data = churn_data[churn_data['TotalCharges'].notnull()]
churn_data = churn_data.reset_index()[churn_data.columns]

# 3 - Convert to float type
churn_data["TotalCharges"] = churn_data["TotalCharges"].astype(float)

# 4 - Replace 'No internet service' to No for the following columns
replace_cols = ['OnlineSecurity',
                'OnlineBackup',
                'DeviceProtection',
                'TechSupport',
                'StreamingTV',
                'StreamingMovies']

for i in replace_cols:
    churn_data[i].replace({'No internet service': 'No'}, inplace=True)

# 5 - Replace values
churn_data['SeniorCitizen'] = churn_data['SeniorCitizen'].replace({1:"Yes",0:"No"})
#churn_data['SeniorCitizen'].replace({0: "No", 1: "Yes"}, inplace=True)
churn_data['MultipleLines'].replace({'No phone service': 'No'}, inplace=True)
churn_data['InternetService'].replace({'DSL':'Yes','Fiber optic':'Yes'}, inplace = True)

detail_data(churn_data,'telcom')



# 6 - Tenure to categorical column
def tenure_lab(telcom):
    if telcom["tenure"] <= 12:
        return "Tenure_0-12"
    elif (telcom["tenure"] > 12) & (telcom["tenure"] <= 24):
        return "Tenure_12-24"
    elif (telcom["tenure"] > 24) & (telcom["tenure"] <= 48):
        return "Tenure_24-48"
    elif (telcom["tenure"] > 48) & (telcom["tenure"] <= 60):
        return "Tenure_48-60"
    elif telcom["tenure"] > 60:
        return "Tenure_gt_60"



churn_data["cat_tenure"] = churn_data.apply(lambda churn_data:tenure_lab(churn_data), axis = 1)



# 7 - Separating churn and non churn customers
churn = churn_data[churn_data['Churn'] == 'Yes']
not_churn = churn_data[churn_data['Churn'] == 'No']

# 8 - Separating categorical and numerical columns
Id_col = ['customerID']
target_col = ['Churn']
cat_cols = churn_data.nunique()[churn_data.nunique() < 6].keys().tolist()
cat_cols = [x for x in cat_cols if x not in target_col]

print("cat_cols: \n",cat_cols)
num_cols = [x for x in churn_data.columns if x not in cat_cols + target_col + Id_col]
print("num_cols: \n",num_cols)

print(churn_data.head())
'''