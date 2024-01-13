import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datetime import datetime

# Load and preview the dataset
df = pd.read_csv('september2018.csv', delimiter=';', parse_dates=['DateTime'])
df.head()

# Plot the top 10 active users
plt.figure(1, figsize=(20, 10))
top_10_people = df.groupby('UserID').count().sort_values(by='Tweet', ascending=False)[:10].index
partial_df = df[df.UserID.isin(top_10_people)]
sns.countplot(x='UserID', data=partial_df)
print('Unique Users:', len(df['UserID'].drop_duplicates()))

# Daily tweet count
plt.figure(2, figsize=(20, 10))
plt.xticks(rotation=90)
df['Date'] = df['DateTime'].apply(lambda dt: dt.date())
sns.countplot(x='Date', data=df)

# Analysis for a specific date
date_string = "2018-09-09"
election_date = datetime.strptime(date_string, '%Y-%m-%d').date()
df1 = df[df['Date'] == election_date]
df1.drop('Date', axis=1, inplace=True)
df1['Hour'] = df1['DateTime'].apply(lambda dt: dt.hour)
print(df1.shape)
plt.figure(3)
sns.countplot(x='Hour', data=df1)
plt.show()

# Hashtag analysis
df1["NumTags"] = df1['Tweet'].str.count('#')
top_tags = df1[df1['NumTags'] > 4]
print(top_tags['UserID'].value_counts())
top_tags = top_tags['UserID'].head(10)
df2 = df1[(df1['UserID'].isin(top_tags)) & (df1['NumTags'] > 4)]
df2["Tags"] = df2.Tweet.apply(lambda x: [w for w in x.split() if "#" in w])
print_data = pd.Series(df2["Tags"].sum()).value_counts().head(20).reset_index(name="Tags")
plt.figure(4, figsize=(20, 10))
sns.barplot(x='index', y='Tags', data=print_data)
plt.xlabel('Tags')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()

# Analysis of mentions
df["talksto"] = df.Tweet.apply(lambda x: [w for w in x.split() if "@" in w])
df_most = pd.DataFrame(df.groupby("UserID").talksto.apply(lambda x: Counter(x.sum()).most_common(1)))
df_most["Num"] = df_most.talksto.apply(lambda x: x[0][1] if len(x) > 0 else 0)
df_most.sort_values("Num", ascending=False, inplace=True)
print(df_most.head(10))
print(len(df_most[df_most["Num"]==1])/len(df_most) * 100, '%')
print(df.groupby("UserID").talksto.apply(lambda x: len(set(x.sum()))).sort_values(ascending=False).head(10))

# Specific user tweet analysis
print(df[df.UserID == 'Guds_barn']['Tweet'].value_counts().index[0])
print(df[df.UserID == 'gunillaoberg67']['Tweet'].value_counts()[:50])
