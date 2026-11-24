"""
Module - 06

Assignment: Exploratory Data Analysis (EDA) on the Titanic Dataset
This assignment will guide you through a comprehensive Exploratory Data Analysis (EDA) of the famous Titanic disaster dataset. The goal is to understand the dataset's structure, identify patterns, and uncover insights related to passenger survival.

Dataset
The dataset includes information such as:
PassengerId: Unique ID for each passenger.
Survived: Survival (0 = No, 1 = Yes) - Target Variable.
Pclass: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd).
Name: Passenger name.
Sex: Passenger sex.
Age: Age in years.
SibSp: # of siblings/spouses aboard the Titanic.
Parch: # of parents/children aboard the Titanic.
Ticket: Ticket number.
Fare: Passenger fare.
Cabin: Cabin number.
Embarked: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

The dataset can be found here:
https://github.com/datasciencedojo/datasets/blob/master/titanic.csv

To download this data using terminal, run the following command:
!wget https://raw.githubusercontent.com/datasciencedojo/datasets/refs/heads/master/titanic.csv  



Task 1: Data Loading and Initial Inspection
Load the dataset (e.g., using pandas in Python).
Display the first 5 rows to get a quick look at the data structure.
Check the data types of all columns (.info()).
Get descriptive statistics of the numerical columns (.describe()).
Identify the total number of missing values for each column.


Task 2: Handling Missing Values
Missing data can skew analysis. For this task, you will address the missing values in three key columns: Age, Cabin, and Embarked.
Cabin:
Calculate the percentage of missing values in the Cabin column.
Decision: Do we need to drop the Cabin column entirely from the analysis? Briefly justify your reason.
Embarked:
Find the most frequent port of embarkation (the mode).Impute the missing values in the Embarked column with the mode.
Age:
Impute the missing values in the Age column with the median age. 



Task 3: Univariate Analysis 
Univariate analysis looks at a single variable at a time.
Survival Rate:
Calculate the overall survival rate (percentage of passengers who survived).
Visualize the distribution of the Survived variable (e.g., a count plot/bar chart).
Passenger Class (Pclass):
Visualize the distribution of Pclass using a count plot.
Identify which class had the most passengers.
Age Distribution:
Plot a histogram of the Age column to visualize its distribution.



Task 4: Bivariate and Multivariate Analysis 
Bivariate analysis explores the relationship between two variables, specifically focusing on how other features relate to the target variable (Survived).
Survival by Sex:
Create a crosstabulation (or a grouped count) to show the count of survivors and non-survivors grouped by Sex.
Visualize this relationship using a stacked bar chart or a grouped count plot.
Question: Which gender had a significantly higher survival rate?
Survival by Class (Pclass):
Calculate and visualize the survival rate for each Pclass.
Question: Is there a clear correlation between ticket class and survival probability?
Survival by Age:
Plot and compare the age distribution of survivors vs. non-survivors.
Observation: What does the plot suggest about the survival chances of children and the elderly?
Survival by Port of Embarkation (Embarked):
Calculate and visualize the survival rate for each Embarked port.



Task 5: Conclusion and Insights 
Based on your EDA, write a brief (3-5 sentences) conclusion summarizing the most significant factors affecting survival on the Titanic. Highlight the top 2-3 features that appear to be the strongest predictors of survival.
"""




               ###Solve###

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set plot style
sns.set_theme(style="whitegrid")

# ==========================================
# TASK 1: Data Loading and Initial Inspection
# ==========================================
df = pd.read_csv("titanic.csv")

print("--- First 5 Rows ---")
print(df.head())

print("\n--- Dataset Info ---")
df.info()

print("\n--- Descriptive Statistics ---")
print(df.describe())

print("\n--- Missing Values Count ---")
print(df.isnull().sum())

# ==========================================
# TASK 2: Handling Missing Values
# ==========================================
# 1. Cabin
cabin_missing_pct = (df['Cabin'].isnull().sum() / len(df)) * 100
print(f"\nPercentage of missing values in Cabin: {cabin_missing_pct:.2f}%")
df.drop(columns=['Cabin'], inplace=True)

# 2. Embarked
embarked_mode = df['Embarked'].mode()[0]
print(f"Most frequent port (Mode): {embarked_mode}")
df['Embarked'] = df['Embarked'].fillna(embarked_mode)

# 3. Age
age_median = df['Age'].median()
print(f"Median Age: {age_median}")
df['Age'] = df['Age'].fillna(age_median)

print("\n--- Missing Values After Cleaning ---")
print(df.isnull().sum())

# ==========================================
# TASK 3: Univariate Analysis
# ==========================================
# 1. Overall Survival Rate
survival_rate = df['Survived'].mean() * 100
print(f"\nOverall Survival Rate: {survival_rate:.2f}%")

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Survived', palette='Set2', hue='Survived', legend=False)
plt.title("Survival Distribution (0 = Died, 1 = Survived)")
plt.xlabel("Survived")
plt.ylabel("Passenger Count")
plt.show()

# 2. Passenger Class (Pclass)
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Pclass', palette='Set1', hue='Pclass', legend=False)
plt.title("Passenger Class Distribution")
plt.xlabel("Ticket Class (1 = 1st, 2 = 2nd, 3 = 3rd)")
plt.ylabel("Passenger Count")
plt.show()

# 3. Age Distribution
plt.figure(figsize=(8, 4))
sns.histplot(df['Age'], bins=30, kde=True, color='skyblue')
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# ==========================================
# TASK 4: Bivariate and Multivariate Analysis
# ==========================================
# 1. Survival by Sex
sex_crosstab = pd.crosstab(df['Sex'], df['Survived'])
print("\n--- Survival Crosstab by Sex ---")
print(sex_crosstab)

sex_crosstab.plot(kind='bar', stacked=True, figsize=(6, 4), color=['salmon', 'lightgreen'])
plt.title("Survival Count by Sex")
plt.xlabel("Sex")
plt.ylabel("Passenger Count")
plt.xticks(rotation=0)
plt.legend(["Did Not Survive (0)", "Survived (1)"])
plt.show()

# 2. Survival by Class (Pclass)
pclass_survival = df.groupby('Pclass')['Survived'].mean() * 100
print("\n--- Survival Rate (%) by Pclass ---")
print(pclass_survival)

plt.figure(figsize=(6, 4))
sns.barplot(data=df, x='Pclass', y='Survived', palette='viridis', hue='Pclass', legend=False)
plt.title("Survival Rate by Ticket Class")
plt.xlabel("Ticket Class")
plt.ylabel("Survival Probability")
plt.show()

# 3. Survival by Age
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='Age', hue='Survived', kde=True, element="step", palette={0: 'red', 1: 'green'})
plt.title("Age Distribution: Survivors vs. Non-Survivors")
plt.xlabel("Age")
plt.ylabel("Passenger Count")
plt.legend(title="Survived", labels=["Yes (1)", "No (0)"])
plt.show()

# 4. Survival by Port of Embarkation (Embarked)
embarked_survival = df.groupby('Embarked')['Survived'].mean() * 100
print("\n--- Survival Rate (%) by Embarked Port ---")
print(embarked_survival)

plt.figure(figsize=(6, 4))
sns.barplot(data=df, x='Embarked', y='Survived', palette='coolwarm', hue='Embarked', legend=False)
plt.title("Survival Rate by Port of Embarkation")
plt.xlabel("Port (C = Cherbourg, Q = Queenstown, S = Southampton)")
plt.ylabel("Survival Probability")
plt.show()
