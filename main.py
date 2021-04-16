# Import a CSV file into a Pandas DataFrame#########

import pandas as pd

data = pd.read_csv("BankData.csv")
print(data.head())
print(data.shape)

# Visualize Matplotlib########

import matplotlib.pyplot as plt

data["Customer_Age"].hist(bins=30)
plt.show()
print(data)

#Insights from visualization
avg_CustomerAge_by_Gender = data.groupby("Gender")["Customer_Age"].mean()
print(avg_CustomerAge_by_Gender)
avg_CustomerAge_by_Gender.plot(kind="bar", title="Mean Customer Age by Gender")
plt.show()

# Sorting, Indexing and Grouping###########
print(data.sort_index(level='Customer_Age', ascending=False))
data_sorted = data.head(10)
print(data_sorted.sort_values("CLIENTNUM"))
data_index = data.set_index("CLIENTNUM")
print(data_index)

data_grouped = data_sorted.groupby(["CLIENTNUM"])
print(data_grouped)

data_groupedby = data.groupby('Gender')['Customer_Age'].count()
print(data_groupedby)


#Generate Valuable Insights from visualization###


import matplotlib.pyplot as plt
import pandas as pd


fig, ax = plt.subplots()

def SHOW():  #Create function to  reusable code
    plt.show()

data = pd.read_csv("Train.csv")

data_ID = data["ID"]
data_COST = data["Cost_of_the_Product"]
print(data_ID.head(3)), print(data_COST.head(3))

ax.plot(data.head(3)["ID"], data.head(3)["Cost_of_the_Product"], marker="o", linestyle="--",color="g")
ax.set_title("ID CUSTOMER BY COST OF THE PRODUCT")
plt.grid(True)
plt.xlabel('ID')
plt.ylabel('Cost')
SHOW()


ncalls = data["Customer_care_calls"].value_counts() #counting values for number of calls


print(ncalls)

data3 =  data.iloc[0:3, 0:2]
print(data3)


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


data = pd.read_csv("Train.csv")

data['Mode_of_Shipment'].value_counts()[:3].plot(kind='barh',title="Mode of Shipment")
plt.grid(True)
plt.show()

ship = data['Mode_of_Shipment'].value_counts()
print(ship)

# Replace missing values or dropping duplicates########
print(data)
data.isna()
data.isna().any()
data.isna().sum()
data  = data.replace(0)

import matplotlib.pyplot as plt

data.isna().sum().plot(kind="bar")
plt.show()

data.fillna(0)

data.drop_duplicates()

# Dictionary or lists######

dict_of_lists = {"CLIENTNUM": ["818770008", "768805383"],
                 "Attrition_Flag": ["Existing Customer", "Existing Customer"],
                 "Customer_Age": [45, 49],
                 "Gender": ["M", "F"],
                 "Dependent_count": [3, 5]}

new_dict = pd.DataFrame(dict_of_lists)

def print_function_new_dict():
    print(new_dict)

print_function_new_dict() # Use Function to create reusable code

print(new_dict.sort_index(level="CLIENTNUM", ascending=False))
print(sorted(new_dict["CLIENTNUM"]))

### Looping and Iterrows ###########

import pandas as pd

dataloop = pd.read_csv('BankData.csv', index_col=0)

for lab, row in dataloop.iterrows():
    dataloop.loc[lab, "EDUCATION LEVEL"] = row["Education_Level"].upper()


def print_function():
    print(dataloop)


print_function()  # Use function to create reusable code

import pandas as pd

dataloop2 = pd.read_csv("BankData.csv", index_col=0)
for val in dataloop2:
    print(val)

# Visualize Seaborn

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_seaborn = pd.read_csv("BankData.csv")
sns.countplot(x="Education_Level", data=data_seaborn)
plt.title("Customers Education Level")
plt.show()

#Insight from visualization
sns.scatterplot(x="Customer_Age", y="Dependent_count", data=data_seaborn)
plt.show()

#Insight from visualization
sns.catplot(x="Education_Level", y ="Gender", hue="Gender" ,data=data_seaborn.head(100))
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Insight from visualization
data3 = pd.read_csv("Train.csv")
sns.regplot(x=data3.head(50)["ID"],y=data3.head(50)["Cost_of_the_Product"])
plt.title("Cost of Product by ID Customer")
plt.show()



import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv("Train.csv")
sns.catplot(x= "Customer_care_calls", data= data.head(5),
                col="ID", kind='box')
plt.show()

# Merge DataFrames######

dataframe1 = pd.DataFrame({'country': ['US', 'CAN', 'UK'],
                           'GDP in Bi': ['700', '490', '506']})

dataframe2 = pd.DataFrame({'Country': ['US', 'CAN', 'UK'],
                           'Population in Mi': ['330', '200', '60']})

dataframe_merged = dataframe1.merge(dataframe2, left_on='country', right_on='Country')
print(dataframe_merged)

# Merge CSV Files
import pandas as pd
import matplotlib.pyplot as plt

csv1 = pd.read_csv("BankData.csv")

csv2 = pd.read_csv("Train.csv")

csvtogether = csv1.merge(csv2)
with pd.option_context('display.max_rows', 3, 'display.max_columns', 3):
    print(csvtogether.head())
plt.show()


# Using Numpy

import pandas as pd
import matplotlib as plt
import numpy as np



data = pd.read_csv("Train.csv")
np_vals = data.values
np_DATA = np.array(np_vals)
print(np_DATA)

#Calculate MEAN on cost of the product  using NUMPY
mean = np.mean(np_DATA[:,5])
print(mean)

#Calculate MEDIAN on Weight in grams  using NUMPY
median = np.median(np_DATA[:,10])
print(median)

#Append new data using NUMPY APPEND
import numpy

appenddata = numpy.array([np_vals])

data_appended = numpy.append (appenddata, ["To be examined"])

print(data_appended)
#version 160421