#Import a CSV file into a Pandas DataFrame

import pandas as pd
data = pd.read_csv("BankData.csv")
print(data.head())
print(data.shape)

#Visualize Matplotlib

import matplotlib as plt
import matplotlib.pyplot as plt


data["Customer_Age"].hist(bins=30)
plt.show()
print(data)


avg_CustomerAge_by_Gender = data.groupby("Gender")["Customer_Age"].mean()
print(avg_CustomerAge_by_Gender)
avg_CustomerAge_by_Gender.plot(kind="bar", title="Mean Customer Age by Gender")
plt.show()

#Sorting, Indexing and Grouping
print(data.sort_index(level='Customer_Age', ascending=False))
data_sorted = data.head(10)
print(data_sorted.sort_values("CLIENTNUM"))
data_index = data.set_index("CLIENTNUM")
print(data_index)

data_grouped = data_sorted.groupby(["CLIENTNUM"])
print(data_grouped)

data_groupedby = data.groupby('Gender')['Customer_Age'].count()
print(data_groupedby)

#Replace missing values or dropping duplicates
print(data)
data.isna()
data.isna().any()
data.isna().sum()

import matplotlib.pyplot as plt
data.isna().sum().plot(kind="bar")
plt.show()

data.fillna(0)

data.drop_duplicates()

#Dictionary or lists

dict_of_lists = {"CLIENTNUM": ["818770008", "768805383"],
                 "Attrition_Flag": ["Existing Customer", "Existing Customer"],
                 "Customer_Age": [45, 49],
                 "Gender": ["M", "F"],
                 "Dependent_count": [3, 5]}

new_dict = pd.DataFrame(dict_of_lists)
print(new_dict)
print(new_dict.sort_index(level="CLIENTNUM", ascending=False))
print(sorted(new_dict["CLIENTNUM"]))
















