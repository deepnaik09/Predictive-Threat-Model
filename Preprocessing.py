import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno #helps visualize missing data in a dataset
from sklearn.model_selection import train_test_split , KFold , cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier , plot_tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


file_path = "dataSet/newDataset.csv"
mergedData = pd.read_csv(file_path)
print([col for col in mergedData.columns if mergedData[col].isnull().sum()>0])
# msno.matrix(mergedData)  #to check missing/duplicates value
# plt.figure(figsize=(15,9))
# plt.show()

label_encoder = LabelEncoder()
for col in ["transaction_type"]:
    mergedData[col] = label_encoder.fit_transform(mergedData[col])


#handling missing values
print(mergedData.isnull().sum())
mergedData = mergedData.drop_duplicates()
mergedData.columns = mergedData.columns.str.strip()
mergedData = mergedData.fillna(mergedData.mean(numeric_only=True))
print("missing value after: ")
print([col for col in mergedData.columns if mergedData[col].isnull().sum()>0])


mergedData.to_csv("dataSet/preprocessed_loan_datase1.csv", index=False)

msno.matrix(mergedData)  #to check missing/duplicates value
plt.figure(figsize=(15,9))
plt.show()

sns.pairplot(mergedData, vars=['income_annum', 'loan_amount', 'cibil_score'], hue="loan_status")
plt.show()



sns.barplot(mergedData, x ="loan_to_income_ratio" , y="loan_status", hue="loan_status")
plt.show()
     #check for outliers
sns.boxplot(mergedData["income_annum"])
plt.show()
sns.boxplot(mergedData["loan_amount"])
plt.show()
    
    
    # represents percentage of loan approval or rejction
plt.figure(figsize=(6,6))
mergedData["loan_status"].value_counts().plot.pie(autopct="%1.1f%%", colors=["lightcoral", "lightblue"], startangle=90, wedgeprops={"edgecolor":"black"})
plt.title("Loan Status Distribution")
plt.ylabel("")  
plt.show()
# 
  #relationship between cibil score and loan status 
correlation = mergedData["rental_agreement"].corr(mergedData["loan_approved"])
print(f"Correlation between rental_agreement and loan_approved: {correlation:.2f}")

correlation = mergedData["avg_monthly_balance"].corr(mergedData["loan_approved"])
print(f"Correlation between avg_monthly_balance and loan_approved: {correlation:.2f}")

#Splitting Data (80-20)

X = mergedData[['rental_agreement', 'transaction_type', 'avg_monthly_balance', 'gst_filing_status','overdraft_frequency','salary_deposits','guarantor_exists','upi_transaction_count','age','transaction_type']]  # Select key features
y = mergedData["loan_approved"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training Data: {X_train.shape}, Testing Data: {X_test.shape}")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

#comparing which attributes highly affects loan approval : Feature Importance
feature_importances = dt_model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

#Linear Regression



#Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': log_reg.coef_[0]})
print(feature_importance.sort_values(by="Importance", ascending=False))

#Decision Tree
tree_clf = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)
print(" Decision Tree Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))
plt.figure(figsize=(15,8))
plot_tree(tree_clf, feature_names=X.columns, class_names=['Rejected', 'Approved'], filled=True, rounded=True)
plt.show()

#random forest : Train and Evaluate
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf_clf.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x="Importance", y="Feature", data=feature_importances, color="green")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance from Random Forest")
plt.show()

#SVM Model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_svm)
print("SVM Model Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred_svm))

