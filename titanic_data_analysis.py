import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Enable inline plotting for matplotlib
%matplotlib inline

# Load Titanic dataset
df = pd.read_csv('titanic.csv')

# Basic data exploration
print(df.to_string())
print(df.shape)
print(df.head(0))
print(df.describe(include='all'))
print(df.isna().sum())
print(df['Age'].describe())

# Unique values in the 'Age' column
for col in df:
    uni = df['Age'].unique()
print(uni)

# Mean age
print(df['Age'].mean())
agemean = df['Age'].mean()

# Filling missing values in 'Age' with various methods
newdf = df.copy()
newdf['new_age_mean'] = df['Age'].fillna(agemean)
newdf['new_age_ffill'] = df['Age'].fillna(method='ffill')
newdf['new_age_bfill'] = df['Age'].fillna(method='bfill')

# Display the first few rows
print(newdf.head())

# Check for null values
print(newdf['new_age_ffill'].isnull().sum())
print(newdf['Age'].isnull().sum())

# Age distribution plots
sns.displot(x='Age', data=newdf)
plt.show()

sns.displot(x='Age', data=newdf, kind='kde')
plt.show()

sns.displot(x='Age', data=newdf, kind='ecdf', rug=True)
plt.show()

# Distribution plots for different age filling methods
sns.displot(x='new_age_mean', data=newdf)
plt.show()

sns.displot(x='new_age_ffill', data=newdf)
plt.show()

sns.displot(x='new_age_bfill', data=newdf)
plt.show()

# Filling missing 'Age' values with forward fill for further analysis
df['fixed_age'] = df['Age'].fillna(method='ffill')
print(df.head(0))
print(df['fixed_age'].isnull().sum())

# Bi-variate plots
sns.displot(x='Age', data=df, y='new_age_mean')
plt.show()

sns.catplot(x='Fare', y='new_age_mean', hue="Survived", col='Sex', data=df)

# Box plots for 'Fare' by 'Pclass' and 'Sex'
sns.catplot(x='Pclass', y='Fare', hue="Sex", kind='box', data=df)

# Bar plots for survival by class and sex
sns.catplot(y='Survived', x='Pclass', hue="Sex", kind="bar", data=df)
sns.barplot(x="Pclass", y="Survived", hue="Sex", data=newdf[newdf["new_age_mean"] <= 14.0])
sns.barplot(x="Pclass", y="Survived", hue="Sex", data=newdf[newdf["new_age_mean"] > 14.0])

# Count plot for passengers by class and sex
sns.countplot(x="Pclass", hue="Sex", data=newdf)

# Analysis of cabin data
newdf['Cabin'].isnull().sum()

# Cabin analysis by class
for i in range(1, 4):
    df_Class = newdf[newdf["Pclass"] == i]
    print(pd.isnull(df_Class["Cabin"]).value_counts())

# Extracting deck level from 'Cabin' and creating a new column 'Level'
df_nona = newdf[newdf["Cabin"].isna() == False]
Level = [index[0] for index in df_nona["Cabin"]]
df_nona["Level"] = Level

# Bar plots for fare and survival by level
df_Class1_nona = df_nona[df_nona["Pclass"] == 1]
sns.barplot(x="Level", y="Fare", data=df_Class1_nona).set_title('Fare vs. Level for Class 1')
plt.show()

sns.barplot(x='Level', y='Survived', data=df_nona)
plt.show()

sns.barplot(x="Level", y="Survived", hue="Sex", data=df_nona)
plt.show()
