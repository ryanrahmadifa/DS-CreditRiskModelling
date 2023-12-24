# Import dependencies
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
import itertools
import statsmodels.api as sm

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.utils import resample
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score,confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


# Read the csv dataset into a pandas dataframe
df = pd.read_csv('loan_data_2007_2014.csv')

# Check out how many columns and rows are in the dataframe, and the column info
df.shape

df.info()

# Take a look at the descriptive statistics
df.describe()

# Check for duplicates in the 'id' column
duplicates = df[df.duplicated('id', keep=False)]

# Display the duplicate rows
print(duplicates)

# Delete columns that ONLY has NaN values AND no other value

df_rev = df.dropna(axis=1, how='all')

df_rev.info()

# Select the rows where 'delinq_2yrs' column has missing values
missing_delinq = df[df['delinq_2yrs'].isnull()]

# Display the rows with missing values in 'delinq_2yrs' column
print(missing_delinq)

# Update the DataFrame to contain data that only has values in the delinq_2yrs column
df_rev = df_rev.dropna(subset=['delinq_2yrs'])
print(df_rev)

df_rev.info()

# Get the columns with NaN values and their count
na_columns = df_rev.columns[df_rev.isna().any()]
na_counts = df_rev[na_columns].isna().sum()

# Get the data types of the columns
column_dtypes = df_rev[na_columns].dtypes

# Display the columns with NaN values, their counts, and data types
print("Columns with NaN values and their counts:")
for column in na_columns:
    count = na_counts[column]
    dtype = column_dtypes[column]
    print(f"{column} ({dtype}): {count}")

# Fill NaN values for columns with 0

df_rev['tot_coll_amt'].fillna(0, inplace=True)
df_rev['tot_cur_bal'].fillna(0, inplace=True)
df_rev['total_rev_hi_lim'].fillna(0, inplace=True)
df_rev['mths_since_last_delinq'].fillna(0, inplace=True)
df_rev['mths_since_last_record'].fillna(0, inplace=True)
df_rev['collections_12_mths_ex_med'].fillna(0, inplace=True)
df_rev['mths_since_last_major_derog'].fillna(0, inplace=True)

# Replace NA values in 'next_pymnt_d' with values from 'last_pymnt_d'
df_rev['next_pymnt_d'].fillna(df_rev['last_pymnt_d'], inplace=True)

# Replace NA values in 'last_pymnt_d' with values from 'next_pymnt_d'
df_rev['last_pymnt_d'].fillna(df_rev['next_pymnt_d'], inplace=True)

# Replace NA values in 'revol_util' with values from 'revol_bal'
df_rev['revol_util'].fillna(df_rev['revol_bal'], inplace=True)

# Replace NA values in 'last_credit_pull_d' with values from 'last_pymnt_d'
df_rev['last_credit_pull_d'].fillna(df_rev['last_pymnt_d'], inplace=True)

# Replace NA values in 'title' with values from 'purpose'
df_rev['title'].fillna(df_rev['purpose'], inplace=True)

# Replace NA values in 'emp_title' with object 'none'
df_rev['emp_title'].fillna('none', inplace=True)

# Replace NA values in 'emp_length' with object 'none'
df_rev['emp_length'].fillna('none', inplace=True)

# Desc column does not have any purpose, so we can drop it
df_rev = df_rev.drop(['desc', 'Unnamed: 0'], axis=1)

print(df_rev[df_rev['last_pymnt_d'].isna()]['loan_status'].value_counts())

# Replace NA values in 'last_credit_pull_d' with values from 'last_credit_pull_d'
df_rev['last_pymnt_d'].fillna(df_rev['last_credit_pull_d'], inplace=True)

# Replace NA values in 'next_pymnt_d' with values from 'last_pymnt_d'
df_rev['next_pymnt_d'].fillna(df_rev['last_pymnt_d'], inplace=True)

# Get the list of values that are in the loan_status column
print(df_rev['loan_status'].unique())

# loan_status values will be converted to Label
label_categories = [
    (0, ['Fully Paid', 'Does not meet the credit policy. Status:Fully Paid', 'Current']),
    (1, ['Late (31-120 days)', 'Late (16-30 days)', 'In Grace Period',
         'Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off'])
]

# function to apply the transformation
def classify_label(text):
    for category, matches in label_categories:
        if any(match in text for match in matches):
            return category
    return None

df_rev.loc[:, 'label'] = df_rev['loan_status'].apply(classify_label)

# Print the unique values of every column to analyze deeper

pd.Series({col:df_rev[col].unique() for col in df_rev})

# Convert date columns with object data type into datetime data type

# Convert 'last_pymnt_d' to datetime format
df_rev['last_pymnt_d'] = pd.to_datetime(df_rev['last_pymnt_d'], format='%b-%y').dt.date

# Convert 'issue_d' to datetime format
df_rev['issue_d'] = pd.to_datetime(df_rev['issue_d'], format='%b-%y').dt.date

# Columns policy_code and application_type only has 1 unique value, we can drop them
df_rev = df_rev.drop(['policy_code','application_type'],axis = 1)

# Drop columns that has no relevancy in exploring the data
df_rev = df_rev.drop(['url','id','member_id','title','earliest_cr_line','last_credit_pull_d','next_pymnt_d','emp_title'],axis = 1)

# Ordinal encoding for the columns
def OrdinalEncoder1(text):
    if text == "F":
        return 1
    if text == "E":
        return 2
    elif text == "D":
        return 3
    elif text == "C":
        return 4
    elif text == "B":
        return 5
    elif text == "A":
        return 6
    else:
        return 0


def OrdinalEncoder2(text):
    if text == "< 1 year":
        return 1
    elif text == "1 year":
        return 2
    elif text == "2 years":
        return 3
    elif text == "3 years":
        return 4
    elif text == "4 years":
        return 5
    elif text == "5 years":
        return 6
    elif text == "6 years":
        return 7
    elif text == "7 years":
        return 8
    elif text == "8 years":
        return 9
    elif text == "9 years":
        return 10
    elif text == "10 years":
        return 11
    elif text == "10+ years":
        return 12
    else:
        return 0

def OrdinalEncoder3(text):
    if text == "RENT":
        return 1
    elif text == "MORTGAGE":
        return 2
    elif text == "OWN":
        return 3
    else:
        return 0

def OrdinalEncoder4(text):
    if text == "Verified":
        return 1
    if text == "Source Verified":
        return 2
    else:
        return 0

def OrdinalEncoder5(text):
    if text == "y":
        return 1
    else:
        return 0

def OrdinalEncoder6(text):
    if "60" in text:
        return 1
    else:
        return 0

def OrdinalEncoder7(text):
    if text == "f":
        return 0
    else:
        return 1

df_rev["grade"] = df_rev["grade"].apply(OrdinalEncoder1)
df_rev["emp_length"] = df_rev["emp_length"].apply(OrdinalEncoder2)
df_rev["home_ownership"] = df_rev["home_ownership"].apply(OrdinalEncoder3)
df_rev["verification_status"] = df_rev["verification_status"].apply(OrdinalEncoder4)
df_rev["pymnt_plan"] = df_rev["pymnt_plan"].apply(OrdinalEncoder5)
df_rev["term"] = df_rev["term"].apply(OrdinalEncoder6)
df_rev["initial_list_status"] = df_rev["initial_list_status"].apply(OrdinalEncoder7)

fig, ax = plt.subplots(1,2,figsize=(15,5))
sns.countplot(data=df_rev, x='label', hue='addr_state', ax=ax[0]).set_title("x")
sns.countplot(data=df_rev, x='label', hue='purpose', ax=ax[1]).set_title("x")

fig, ax = plt.subplots(1,2,figsize=(15,5))
sns.countplot(data=df_rev, x='label', hue='loan_status', ax=ax[0]).set_title("x")
sns.countplot(data=df_rev, x='label', hue='sub_grade', ax=ax[1]).set_title("x")



# Group the DataFrame by 'addr_state' and calculate the counts of label 0 and label 1
grouped = df_rev.groupby('addr_state')['label'].value_counts().unstack(fill_value=0)

# Calculate the total count of labels (0+1) in each state
grouped['total_count'] = grouped.sum(axis=1)

# Perform linear regression analysis
X = grouped['total_count']
y = grouped[1]  # Label 1 count

# Add a constant to the X variable
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X)
results = model.fit()

# Print the regression results
print(results.summary())

# Drop the columns
df_rev = df_rev.drop(['addr_state','zip_code','purpose','loan_status','sub_grade'],axis = 1)

# Group the DataFrame by 'issue_d' and count the occurrences of label 1 for each date
grouped = df_rev[df_rev['label'] == 1].groupby('issue_d').size()

# Plot the line chart
plt.plot(grouped.index, grouped.values, marker='o', linestyle='-', markersize=4)
plt.xlabel('Issue Date')
plt.ylabel('Count of Label 1')
plt.title('Count of Label 1 for Each Date')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Select a subset of variables for the heatmap
subset_vars = df[['loan_amnt', 'funded_amnt', 'int_rate', 'installment','annual_inc','pub_rec']]
# Calculate the correlation matrix
corr_matrix = subset_vars.corr()

# Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")

# Display the heatmap
plt.show()

corr = df_rev[['loan_amnt', 'int_rate', 'grade', 'emp_length', 'home_ownership', 'annual_inc','verification_status','label',
                    'pymnt_plan','dti','delinq_2yrs','mths_since_last_delinq','mths_since_last_record','pub_rec']].corr()
sns.set(rc={'figure.figsize':(11,7)})
sns.heatmap(corr,linewidths=.5, annot=True, cmap="YlGnBu",mask=np.triu(np.ones_like(corr, dtype=np.bool_)))\
    .set_title("Pearson Correlations Heatmap");

# Count the values to show how many data we have for each label
print(df_rev[['label']].value_counts())

# Separate the majority and minority classes
majority_class = df_rev[df_rev['label'] == 0]
minority_class = df_rev[df_rev['label'] == 1]

# Undersample the majority class
undersampled_majority = resample(majority_class,
                                replace=False,  # Sampling without replacement
                                n_samples=len(minority_class),  # Match the number of samples in the minority class
                                random_state=42)  # Set a random seed for reproducibility

# Combine the undersampled majority class with the minority class
df_undersampled = pd.concat([undersampled_majority, minority_class])

# Shuffle the DataFrame
df_undersampled = df_undersampled.sample(frac=1, random_state=42)

# Print the value counts of the undersampled data
print(df_undersampled['label'].value_counts())

# Declare feature vector and target variable
X = df_undersampled.drop(['label','issue_d','last_pymnt_d','total_pymnt_inv','out_prncp_inv'], axis = 1)
y = df_undersampled['label']

# Split the data into a training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Create an instance of SelectKBest with chi2 scoring function and k=30
selector = SelectKBest(chi2, k=30).fit(X_train, y_train)

# Get the chi-square scores and p-values
scores = selector.scores_
p_values = selector.pvalues_

# Calculate the proportion of explained data for each feature
explained_data = scores / sum(scores)

# Create a DataFrame to store the scores and p-values
feature_scores = pd.DataFrame({'Feature': X.columns, 'Score': scores, 'p-value': p_values})

# Sort the DataFrame by descending order of scores
feature_scores = feature_scores.sort_values('Score', ascending=False)

# Print the ordered feature scores
print(feature_scores)

X_train_2 = selector.transform(X_train)
X_test_2 = selector.transform(X_test)

# Random Forest Classifier
# Random forest classifier with n_estimators=10 (default)
clf_rf_2 = RandomForestClassifier()
clr_rf_2 = clf_rf_2.fit(X_train_2,y_train)

acc_rf = accuracy_score(y_test,clf_rf_2.predict(X_test_2))
recall_rf = recall_score(y_test,clf_rf_2.predict(X_test_2))
pre_rf = precision_score(y_test,clf_rf_2.predict(X_test_2))
f1_rf = f1_score(y_test,clf_rf_2.predict(X_test_2))
print('Accuracy is: ',acc_rf)
print('Recall is: ',recall_rf)
print('F1 Score is: ',f1_rf)
print('Precision is: ')
cm_1 = confusion_matrix(y_test,clf_rf_2.predict(X_test_2))
sns.heatmap(cm_1,annot=True,fmt="d")

## 3.2 K-Nearest Neighbors

inertias = []

for i in range(2, 16):
    kmeans = KMeans(n_clusters=i, n_init=10, random_state=0).fit(X_train_2)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.title('Inertias vs. N_Clusters')
plt.plot(np.arange(2, 16), inertias, marker='o', lw=2)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

max_score = 0
max_k = 0
for k in range(1, 50):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train_2,y_train)
    score = f1_score(y_test, neigh.predict(X_test_2),average='micro')
    if score > max_score:
        max_k = k
        max_score = score

print('If we use K-Nearest Neighbors Classification, then the value of K is',str(max_k),' to get the best prediction, then the average accuracy is ', max_score)

# Train the model with the optimal value of k
neigh = KNeighborsClassifier(n_neighbors=max_k)
neigh.fit(X_train_2, y_train)

acc_knn = accuracy_score(y_test,neigh.predict(X_test_2))
recall_knn = recall_score(y_test,neigh.predict(X_test_2))
pre_knn = precision_score(y_test,neigh.predict(X_test_2))
f1_knn = f1_score(y_test,neigh.predict(X_test_2))
print('Accuracy is: ',acc_knn)
print('Recall is: ',recall_knn)
print('Precision is: ',pre_knn)
print('F1 Score is: ',f1_knn)
cm_1 = confusion_matrix(y_test,neigh.predict(X_test_2))
sns.heatmap(cm_1,annot=True,fmt="d")

## 3.3 XGBoost

# Model Training
RANDOM_STATE = 42
LR = 0.01
TEST_SIZE = 0.33
MAX_DEPTH = 0
NTHREAD = 2
EVAL_METRIC = 'logloss'
BOOSTER = 'gbtree'
VERBOSITY = 1

xgboost = XGBClassifier(
                        random_state=RANDOM_STATE,
                        learning_rate=LR,
                        booster=BOOSTER,
                        nthread=NTHREAD,
                        eval_metric=EVAL_METRIC,
                        verbosity=VERBOSITY
                        )

start = time.time() # Time before training

# Fit the model with the training data
xgboost.fit(X_train_2, y_train)

end = time.time() # Time after training

# Compute how much time the model need to train
print(f'Training took {round(end-start,2)} seconds to be completed!')

# predict the target on the train dataset
predict_train = xgboost.predict(X_train_2)
print('\nTarget on train data',predict_train)

# Accuracy Score on train dataset
accuracy_train = accuracy_score(y_train,predict_train)
print('\naccuracy_score on train dataset : ', accuracy_train)

# predict the target on the test dataset
xgb_predict_test = xgboost.predict(X_test_2)
print('\nTarget on test data',xgb_predict_test)

# Accuracy Score on test dataset
acc_xgb = accuracy_score(y_test,xgb_predict_test)
print('\naccuracy_score on test dataset : ', acc_xgb)

recall_xgb = recall_score(y_test,xgb_predict_test)
pre_xgb = precision_score(y_test,xgb_predict_test)
f1_xgb = f1_score(y_test,xgb_predict_test)
print(f'The accuracy in the test set was {acc_xgb}, the recall was {recall_xgb}, the precision was {pre_xgb} and the f1 score was {f1_xgb}')

cm_3 = confusion_matrix(y_test, xgb_predict_test)
cm_plot = sns.heatmap(cm_3,
                      annot=True,
                      cmap='Blues',
                      fmt='d');
cm_plot.set_xlabel('Predicted Values')
cm_plot.set_ylabel('Actual Values')
cm_plot.set_title('Confusion Matrix', size=16)

## 3.4 Naive Bayes

# instantiate the model
gnb = GaussianNB()

# fit the model
gnb.fit(X_train_2, y_train)

y_pred = gnb.predict(X_test_2)

y_pred_train = gnb.predict(X_train_2)

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))

# print the scores on training and test set

print('Training set score: {:.4f}'.format(gnb.score(X_train_2, y_train)))

print('Test set score: {:.4f}'.format(gnb.score(X_test_2, y_test)))

acc_gnb = accuracy_score(y_test,gnb.predict(X_test_2))
recall_gnb = recall_score(y_test,gnb.predict(X_test_2))
pre_gnb = recall_score(y_test,gnb.predict(X_test_2))
f1_gnb = f1_score(y_test,gnb.predict(X_test_2))
print('Accuracy is: ',acc_gnb)
print('Recall is: ',recall_gnb)
print('Precision is: ',pre_gnb)
print('F1 Score is: ',f1_gnb)
cm_4 = confusion_matrix(y_test,gnb.predict(X_test_2))
sns.heatmap(cm_1,annot=True,fmt="d")

#Artificial Neural Network
# Define a batch size
bs = 128
# Both x_train and y_train can be combined in a single TensorDataset, which will be easier to iterate over and slice

# Convert X_train_2 and y_train to PyTorch tensors
X_train_2_tensor = torch.tensor(X_train_2, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32).unsqueeze(1)

train_ds = TensorDataset(X_train_2_tensor, y_train_tensor)

# Pytorch’s DataLoader is responsible for managing batches.
# You can create a DataLoader from any Dataset. DataLoader makes it easier to iterate over batches
train_dl = DataLoader(train_ds, batch_size=bs)

#For the validation/test dataset
# Convert X_test_2 and y_test to PyTorch tensors
X_test_2_tensor = torch.tensor(X_test_2, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.float32).unsqueeze(1)

test_ds = TensorDataset(X_test_2_tensor, y_test_tensor)
test_loader = DataLoader(test_ds, batch_size=32)

n_input_dim = X_train_2_tensor.shape[1]

#Layer size
n_hidden1 = 256
n_hidden2 = 512  # Number of hidden nodes
n_hidden3 = 256
n_hidden4 = 128
n_hidden5 = 64
n_hidden6 = 32
n_output =  1   # Number of output nodes = for binary classifier
patience = 5  # Number of epochs with no improvement before reducing LR


class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.layer_1 = nn.Linear(n_input_dim, n_hidden1)
        self.layer_2 = nn.Linear(n_hidden1, n_hidden2)
        self.layer_3 = nn.Linear(n_hidden2, n_hidden3)
        self.layer_4 = nn.Linear(n_hidden3, n_hidden4)
        self.layer_5 = nn.Linear(n_hidden4, n_hidden5)
        self.layer_6 = nn.Linear(n_hidden5, n_hidden6)
        self.layer_out = nn.Linear(n_hidden6, n_output)


        self.relu = nn.ReLU()
        self.sigmoid =  nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(n_hidden1)
        self.batchnorm2 = nn.BatchNorm1d(n_hidden2)
        self.batchnorm3 = nn.BatchNorm1d(n_hidden3)
        self.batchnorm4 = nn.BatchNorm1d(n_hidden4)
        self.batchnorm5 = nn.BatchNorm1d(n_hidden5)
        self.batchnorm6 = nn.BatchNorm1d(n_hidden6)


    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.relu(self.layer_3(x))
        x = self.batchnorm3(x)
        x = self.relu(self.layer_4(x))
        x = self.batchnorm4(x)
        x = self.relu(self.layer_5(x))
        x = self.batchnorm5(x)
        x = self.relu(self.layer_6(x))
        x = self.batchnorm6(x)
        x = self.dropout(x)
        x = self.sigmoid(self.layer_out(x))

        return x


ann = ANN()
print(ann)

# Loss Computation
loss_func = nn.BCELoss()
# Optimizer
learning_rate = 0.001
optimizer = torch.optim.Adam(ann.parameters(), lr=learning_rate)

epochs = 250

ann.train()
train_loss = []
for epoch in range(epochs):

    #Within each epoch run the subsets of data = batch sizes.
    for xb, yb in train_dl:
        y_pred = ann(xb)            # Forward Propagation
        loss = loss_func(y_pred, yb)  # Loss Computation
        optimizer.zero_grad()         # Clearing all previous gradients, setting to zero
        loss.backward()               # Back Propagation
        optimizer.step()              # Updating the parameters

    print("Loss in iteration "+str(epoch)+" is: "+str(loss.item()))
    train_loss.append(loss.item())
print('Last iteration loss value: '+str(loss.item()))

# Plot loss overtime

plt.plot(train_loss)
plt.show()

# Make predictions on the test set
with torch.no_grad():
    outputs = ann(X_test_2_tensor)
    predicted_labels = (outputs >= 0.5).float()

y_pred_list = []
ann.eval()
#Since we don't need model to back propagate the gradients in test set we use torch.no_grad()
# reduces memory usage and speeds up computation
with torch.no_grad():
    for xb_test,yb_test  in test_loader:
        y_test_pred = ann(xb_test)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list.append(y_pred_tag.detach().numpy())

#Takes arrays and makes them list of list for each batch
y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
#flattens the lists in sequence
ytest_pred = list(itertools.chain.from_iterable(y_pred_list))

y_true_test = y_test.values.ravel()
conf_matrix = confusion_matrix(y_true_test ,ytest_pred)
acc_ann = accuracy_score(y_true_test,ytest_pred)
recall_ann = recall_score(y_true_test,ytest_pred)
pre_ann = precision_score(y_true_test,ytest_pred)
f1_ann = f1_score(y_true_test,ytest_pred)
print("Confusion Matrix of the Test Set")
print("-----------")
print(conf_matrix)
print("Accuracy: "+str(acc_ann))
print("Recall: "+str(recall_ann))
print('Recall is: ',str(pre_ann))
print("F1 Score: "+str(f1_ann))

# Model comparisons
print('Random Forest Classifier ','\n','Accuracy: ',acc_rf, 'Recall: ',recall_rf, 'Precision: ', pre_rf, 'F1 Score: ',f1_rf)
print('K-Nearest Neighbors ','\n','Accuracy: ',acc_knn,'Recall: ',recall_knn,'Precision: ', pre_knn, 'F1 Score: ',f1_knn)
print('XGBoost ','\n','Accuracy: ',acc_xgb,'Recall: ',recall_xgb,'Precision: ', pre_xgb,'F1 Score:',f1_xgb)
print('Naive Bayes ','\n','Accuracy: ',acc_gnb,'Recall: ',recall_gnb,'Precision: ', pre_gnb,'F1 Score: ',f1_gnb)
print('Artificial Neural Network','\n','Accuracy: ',acc_ann,'Recall: ',recall_ann,'Precision: ', pre_ann,'F1 Score: ',f1_ann)