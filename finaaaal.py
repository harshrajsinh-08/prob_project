# %% [markdown]
# ### Content

# %%

#### 0   | Check and Clean Dataframe
#### 1   | Comment on numerical summary for Quantitative variable as well as Categorical variables
#### 2   | Observe distribution of Amount
#### 3.1 | Comment on count distribution of City (With Visualization)
#### 3.2 | Relation of city with card type /gender wise / exp type distribution
#### 3.3 | Relation of city with amount
#### 4.1 | Comment on count distribution of 3 categories ( Card Type , Exp Type , Gender )
#### 4.2 | Maximimum and minimum contribution to amount for ( Card Type , Exp Type , Gender )
#### 4.3 | 5 point summary wrt amount for each subcategories of ( Card Type , Exp Type , Gender )
#### 5.1 | Comment on count distribution of Gender among (Exp Type , Card Type )
#### 5.2 | Observe Relationship between card type and exp type (Vice Versa)
#### Conclusion

# %%
import numpy as np # linear algebra
import pandas as pd # data processing , reading the csv file (pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import scipy.stats as stats
from scipy.stats import ttest_ind
import norm

# %%
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns',100)          # Set now of columns visible to 100

# %%
df=pd.read_csv("/Users/harsh/PROB ML/final/Credit card transactions - India - Simple.csv")

# %%
print(df.columns)

# %% [markdown]
# ### 0 | Check and Clean Dataframe

# %%
# Check First five 
df.head() # data.head(10)
#df.tail()

# %%
df.info()  # As the count of non null equals to rangeindex , we can conclude no null values

# %%
df.drop(columns='index',inplace=True) # Removing the irrelevent Column
df_clean = df.dropna(subset=['Amount'])

# %%
df=df.replace(', India','', regex=True) # Since India is common

# Convert 'Amount' column to numeric
df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

# Convert 'Gender' column to string
df['Gender'] = df['Gender'].astype(str)


# %% [markdown]
# ### 1 | Comment on numerical summary for Quantitative variable as well as Categorical variables

# %%
df.describe(include='O') # To describe stats for categorical column

# %% [markdown]
# ***
# 1. On looking the summary for Categorical variables We got to know about the following points : - 
# ***
#    - Bengaluru is most frequent in transactions. 
#    - Silver is the most used card Category type.
#    - Most of the transactions are done in food Category. 
#    - Female Does the most no of transactions.               
# ***

# %% [markdown]
# ### 2 | Observe distribution of Amount

# %%
df['Amount'].describe() # To describe stats for quantitative column

# %%
print("Skewness for amount " + str(df["Amount"].skew())+"\n")

plt.subplots(1,2,figsize=(26,5))

plt.subplot(1,2,1)
sns.histplot(x=df["Amount"])
plt.title("Distribution of Amounts in Transactions")


plt.subplot(1,2,2)
sns.boxplot(x=df["Amount"])
plt.title("Distribution of Amounts in Transactions")

plt.show()

# %% [markdown]
# 2. On looking the Distribution of Amount
#     
#     *******
#     * Using Describe Summary 
#          * we can see amount ranging from 1005 to 998077
#              - 25% of transactions amount lies between 1005 to 77k. 
#              - Middle 50 % of transactions lies between 77k to 228k. 
#              - 25 % of transaction lies between 228k to 998k.     
#     
#     *******
#     
#     * Using skewness method 
#         * we got the skewnss for the amount which was 1.7 and anything greater then 1 is highly skewed.
#         
#     *******
#     * Using Histplot 
#         * we observed It followed a uniform distribution with high peaks initially starting from thousand to around 3 lakh.
#         * Followed by few no of transactions upto 10 lakh (negligible compared to above).
#         * Since it does not follow Normal distribution so iqr method gives the proper view of outliers.
#         
#     *******    
#     * Using Boxplot
#         * we can observe the above same stats as describe (approx)
#         * we can see the presence of outlier shown as points after whisker end (which extended eitherways to 1.5 times the iqr from q3,q1 by default).
#     *******
#         
#         

# %% [markdown]
# ### 3.1 | Comment on count distribution of City (With Visualization)

# %%
city=pd.DataFrame({'count' : df.groupby("City").size()}).reset_index()

print("\n")
print("Top 3 City  ->  "+str(city.nlargest(3,'count')['City'].tolist()))
print("\n")
print("Last 3 City ->  " + str(city.nsmallest(3,'count')['City'].tolist()))
print("\n")

# %%
#pd.DataFrame({'count' : df.groupby(  "City").size()}).reset_index()  # to get the result as df 
city_cropped=city.copy()
city_cropped.loc[~(city_cropped.City.isin(city_cropped.nlargest(5,'count')['City'].tolist())),'City']='Other'
city_cropped=city_cropped.groupby("City").sum().reset_index()

#For Ordering
set_ord=city.nlargest(5,'count')['City'].tolist()
set_ord.append("Other")

plt.subplots(1,2,figsize=(19,4))

plt.subplot(1,2,1)
sns.barplot(x=city_cropped['City'], y=city_cropped['count'],order=set_ord)

plt.subplot(1,2,2)
plt.pie(x=city_cropped["count"],labels=city_cropped["City"],autopct="%0.2f%%")

plt.show()


# %% [markdown]
# 3.1 
# 
# On Observing the city column 
# 
# ***
#    * Group by 
#         * we get the top 3 city as per no of transactions using nlargest fucntion.
#         * we got the bottom 3 city as per no of transactions using nsmallest function.
# ***   
#    * Barplot
#         * Bengaluru has the highest peak with the most no of transactions
#         * Peak between top 4 does not have sharp deviation but moving to 5 we see a sharp drop in peak  
#         * this gives fair idea about the importance of top 4 but it will be better to observe proportion
# ***       
#    * Pieplot
#     
#        * This gives us the idea that first 5 out of 986 countries contributes over 56.8% of the transactions.
#        * other 981 contributes to around 43.2% of the transactions
# ***
# ##### " Since there were many catergories so Before visualizing we grouped  all city's other then top 5 into 'Others' so we now have 6 categories in city column "
# ***
# 
#        
#        

# %% [markdown]
# ### 3.2 |   Relation of city with card type /gender wise / exp type  distribution

# %%
# Without using added mask column for city 
#pd.DataFrame({'count' : df.groupby(  "City").size()}).reset_index()  # to get the result as df 

city_gender_cropped=pd.DataFrame({'count' : df.groupby(["City",'Gender']).size()}).reset_index() 
city_gender_cropped.loc[~(city_gender_cropped.City.isin(city_cropped.nlargest(6,'count')['City'].tolist())),'City']='Other'
city_gender_cropped=city_gender_cropped.groupby(['City','Gender']).sum().reset_index()
city_gender_cropped

# %%
df['City']=np.where(df.City.isin(city.nlargest(5,'count')['City'].tolist()), df.City, 'Other')

# %%
# With Mask Column
df.groupby(["City",'Card Type']).size()

# %%
df.groupby(["City",'Exp Type']).size()

# %%
#most_city=pd.DataFrame({'count' : df.groupby("City").size()}).reset_index().nlargest(5,'count')['City']
#sns.countplot(x=df.loc[df['City'].isin(most_city.to_list())]['City'] , hue=df['Gender'])

plt.subplots(1,3,figsize=(26,6))

plt.subplot(1,3,1)
sns.countplot(x=df["City"],hue=df['Gender'],order=set_ord,palette=sns.color_palette("RdBu"))
plt.xticks(rotation=45)
plt.title("Distribution of Gender in transactions")

plt.subplot(1,3,2)
sns.countplot(x=df["City"],hue=df['Exp Type'],order=set_ord,palette=sns.color_palette("RdBu"))
plt.xticks(rotation=45)
plt.title("Distribution of Expense Category in transactions")

plt.subplot(1,3,3)
sns.countplot(x=df["City"],hue=df['Card Type'],order=set_ord,palette=sns.color_palette("RdBu"))
plt.xticks(rotation=45)
plt.title("Distribution of Card Type in transactions")


plt.show()


# %% [markdown]
# 3.2
# ***
# ##### Distribution of gender in top 5 city
# 
#   * Using Countplot and Group (Group by city followed by Gender) 
#     
#        * In top 4 city ['Bengaluru', 'Greater Mumbai', 'Ahmedabad','Delhi'] females Dominates no of transactions 
#        * The diff in no of transactions for female over man ranges from 200 to 400 in each of the city 
#        * Male dominates over Hyderabad with minor difference of 6  
#        * Male dominates over "Other" (rest of city combined ) with few no's around to 30-40 
#         
# ***        
# ##### Distribution of Card Type in top 5 city
# 
#    *  Using Countplot and Group (Group by city followed by Card)
#         
#        * Silver is the most used card type in top 4 city(with most trxns) ['Bengaluru', 'Greater Mumbai', 'Ahmedabad','Delhi']
#        * Platinum card user does the most no of transactions in hyderabad 
#        * While in rest of the city "Other" signature is the most and platinum the least used card
#             
# ***            
# ##### Distribution of Exp Type in top 5 city
# 
#    *  Using Countplot and Group (Group by city followed by Card)
#         
#        * Food is most frequent expense category in "OTHERS" and 3 out of top 4 city(with most trxns) ['Bengaluru', 'Ahmedabad','Delhi']
#        * In Hyderabad Bills and Grocery exceeds food category with minor difference of 4
#        * In Greater Mumbai fuel category has the most frequent transactions
#        * There is no transactions done in top 4 city for travel category.
# ***

# %% [markdown]
# ### 3.3 | Relation of city with amount

# %%
df.groupby('City').describe()

# %%
total_amt_city=df.groupby('City').sum('Amount')[:-1].reset_index()
total_amt_city

# %%
df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

# Group by 'City' and calculate skewness for each city group
city_skewness = df.groupby('City')['Amount'].skew()

# Print skewness for each city
print("Skewness for each city:")
print(city_skewness)

# %%
plt.subplots(1,3,figsize=(26,5))
plt.subplot(1,3,1)
sns.boxplot(data=df,x='City' ,y='Amount',palette=sns.color_palette("RdBu"))
plt.title("Distribution of Amount City wise")

plt.subplot(1,3,2)
sns.barplot(x=df["City"],y=df["Amount"],palette=sns.color_palette("RdBu"))
plt.title("Mean Distribution of Amount City wise")

plt.subplot(1,3,3)
sns.barplot(x=total_amt_city["City"], y=total_amt_city['Amount'],palette=sns.color_palette("RdBu"))
plt.title("City Wise Total Amount")

plt.show()

# %% [markdown]
# 3.3
# ***
# * Box plot Graph - Understanding Distribution of Amount City wise
#     * Min amount and Median amount for all city category lies close to each other 
#     * Top 4 city has presence of outlier which is the amount greater then (1.5 * iqr + q3)
# ***
# * Bar Plot - Understanding Mean City wise
#     * Mean amount for transaction lies around 150k (approx)
#     * Hyderabad has the least mean in top 5 cities
#     
# ***       
# * Bar Plot - Understanding Total Amount City wise
#     * Mumbai has the highest total sum of amount followed by the bengaluru
# ***    
#     

# %% [markdown]
# ### 4.1 | Comment on count distribution of 3 categories ( Card Type , Exp Type ,  Gender )

# %%
# For Individual 
# Gender distribution
# plt.figure(figsize=(4,3)) # (width,height)
# sns.countplot(x=df["Gender"])
# plt.title("Distribution of Gender in transactions")
# plt.show()


# For Multiple

plt.subplots(2,3,figsize=(24,12))

plt.subplot(2,3,1)
sns.countplot(x=df["Gender"],palette=sns.color_palette("RdBu"))
plt.title("Distribution of Gender in transactions")

plt.subplot(2,3,2)
sns.countplot(x=df["Exp Type"],palette=sns.color_palette("RdBu"))
plt.title("Distribution of Expense Category in transactions")

plt.subplot(2,3,3)
sns.countplot(x=df["Card Type"],palette=sns.color_palette("RdBu"))
plt.title("Distribution of Card Type in transactions")

plt.subplot(2,3,4)
plt.pie(x=df["Gender"].value_counts(),labels=df["Gender"].value_counts().index,autopct="%0.2f%%")

plt.subplot(2,3,5)
plt.pie(x=df["Exp Type"].value_counts(),labels=df["Exp Type"].value_counts().index,autopct="%0.2f%%")

plt.subplot(2,3,6)
plt.pie(x=df["Card Type"].value_counts(),labels=df["Card Type"].value_counts().index,autopct="%0.2f%%")
plt.show()

# %% [markdown]
# 4.1
# ***
# * Graph - Distribution of Gender in transactions
#     * It tells the no of transactions done by female is 5 % more then the male
# ***
# * Graph - Distribution of Expense Category in transactions
#     * We have 6 different categories
#     * Food has the highest 20.97%
#     * We can see most of Exp type categories (5 out of 6) has transaction share between 18.25 to 20.97 
#     * We have (travel) 1 out of 6 category which has the least share with 2.83 %
# ***       
# * Graph - Distribution of Card Type in transactions
#     * We have 4 different card type categories
#     * All 4 transaction share lie between 24.44 % to 26.26 %
#     * Silver is the most used card type
# ***    
#     

# %% [markdown]
# ### 4.2 | Maximimum and minimum contribution to amount for ( Card Type , Exp Type ,  Gender )

# %%
import matplotlib.pyplot as plt
df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

# Convert 'Gender' column to string
df['Gender'] = df['Gender'].astype(str)

# Plot the total sum of amounts grouped by gender
plt.plot(df.groupby('Gender')['Amount'].sum())
plt.xlabel('Gender')
plt.ylabel('Total Amount')
plt.title('Total Amount by Gender')
plt.show()

# %% [markdown]
# 4.2
# ***
# * Graph - Distribution of Gender in Total Amount
#     * Female has the highest total amount then male
# ***
# * Graph - Distribution of Expense Category in Total Amount
#     * Bills has the highest total amount and travel has the least
# ***       
# * Graph - Distribution of Card Type in Total Amount
#     * Silver contributes highest to the total amount and gold has the least contribution
# ***    
#     

# %% [markdown]
# ### 4.3 | 5 point summary wrt amount for each subcategories of ( Card Type , Exp Type ,  Gender )

# %%
df.groupby('Gender').describe()

# %%
df.groupby('Card Type').describe()

# %%
df.groupby('Exp Type').describe()

# %%
plt.subplots(2,3,figsize=(24,10))

plt.subplot(2,3,1)
sns.barplot(x=df["Gender"],y=df["Amount"],palette=sns.color_palette("RdBu"))
plt.title("Distribution of Gender in transactions")

plt.subplot(2,3,2)
sns.barplot(x=df["Exp Type"],y=df["Amount"],palette=sns.color_palette("RdBu"))
plt.title("Distribution of Expense Category in transactions")

plt.subplot(2,3,3)
sns.barplot(x=df["Card Type"],y=df["Amount"],palette=sns.color_palette("RdBu"))
plt.title("Distribution of Card Type in transactions")


plt.subplot(2,3,4)
sns.boxplot(data=df,x='Gender' ,y='Amount',palette=sns.color_palette("RdBu"))

plt.subplot(2,3,5)
sns.boxplot(data=df,x='Exp Type' ,y='Amount',palette=sns.color_palette("RdBu"))

plt.subplot(2,3,6)
sns.boxplot(data=df,x='Card Type' ,y='Amount',palette=sns.color_palette("RdBu"))

plt.show()



# %%
#df.groupby('Gender').describe()
print("\n******** Skewness for Gender Type ********\n")
skewness = df.groupby('Gender')['Amount'].skew()
print(skewness)
print("\n******** Skewness for Exp Type ********\n")
# Select numeric columns
numeric_columns = df.select_dtypes(include=np.number).columns

# Calculate skewness for numeric columns grouped by 'Exp Type'
skewness_exp_type = df.groupby('Exp Type')[numeric_columns].skew()
print(skewness_exp_type)

print("\n******** Skewness for Card Type ********\n")

# Select numeric columns
numeric_columns = df.select_dtypes(include=np.number).columns

# Calculate skewness for numeric columns grouped by 'Card Type'
skewness_card_type = df.groupby('Card Type')[numeric_columns].skew()
print(skewness_card_type)


# %% [markdown]
# 4.3
# ***
# * Comment on Median and Mean of Gender Type 
#     * Median is almost close.
#     * Mean for female seems to be high which is influenced by presence of transactions having high amount value.
# ***
# * Comment on Median and Mean of Card Type 
#    * Median is almost close for all subcategories in Card Type
#    * Mean for Bills seems to be high which is influenced by presence of transactions having high amount value.
# ***    
# * Comment on Median and Mean of Exp Type 
#    * Median is almost close for all subcategories in Card Type
#    * Mean is almost close for all subcategories in Card Type 
# ***
# * Does any of the sub category among category Gender Type has skewness 
#    * Looking at the box plot as well as skewness value we can say female transactions are highly skewed
# ***
# * Does any of the sub category among category Card Type has skewness  
#   * Looking at the box plot as well as skewness value we can say Bill Category for transactions are highly skewed
# ***
# * Does any of the sub category among category Exp Type has skewness
#   * All sub category has skewness 
# ***

# %% [markdown]
# ### 5.1 | Comment on count distribution of Gender among (Exp Type ,  Card Type )

# %%
df.groupby(["Exp Type","Gender"]).size()

# %%
df.groupby(["Card Type","Gender"]).size()

# %%
plt.subplots(1,2,figsize=(26,5))
plt.subplot(1,2,1)
sns.countplot(x=df["Exp Type"],hue=df["Gender"],palette=sns.color_palette("RdBu"))
plt.title("Distribution of Gender in transactions over Exp Type")

plt.subplot(1,2,2)
sns.countplot(x=df["Card Type"],hue=df["Gender"],palette=sns.color_palette("RdBu"))
plt.title("Distribution of Gender in transactions over over Card Type")
plt.show()

# %% [markdown]
# 5.1
# ***
# * Graph - Distribution of Gender in transactions over Exp Type
#     * Female has high no transactions in almost all exp category except fuel
# ***
# * Graph - Distribution of Gender in transactions over over Card Type
#     * Female has high no transactions in all Card type
# ***       

# %% [markdown]
# ### 5.2 | Observe Relationship between card type and exp type (Vice Versa)

# %%
df.groupby(["Exp Type","Card Type"]).size()

# %%
df.groupby(["Card Type","Exp Type"]).size()

# %%
plt.subplots(2,1,figsize=(26,12))

plt.subplot(2,1,1)
sns.countplot(x=df["Exp Type"],hue=df["Card Type"],palette=sns.color_palette("RdBu"))
plt.title("Distribution of Card Type in transactions over Exp Type")

plt.subplot(2,1,2)
sns.countplot(x=df["Card Type"],hue=df["Exp Type"],palette=sns.color_palette("RdBu"))
plt.title("Distribution of Exp Type in transactions over Card Type")
plt.show()

# %% [markdown]
# ## Conclusion 

# %% [markdown]
# ***
# * 3 City having highest no of transactions ?
#     * Top 3 City  ->  ['Bengaluru', 'Greater Mumbai', 'Ahmedabad']
# ***
# * 3 City having lowest no of transactions ?
#     * Last 3 City ->  ['Alirajpur', 'Bagaha', 'Changanassery']
# ***
# * Which card type is highly and least used ?
#     * Silver is the highest used card & Gold is the least    
# ***        
# * In which expense category customer does most and least number of transactions  happens ?
#     * Food has the highest no of transactions 
#     * Travel has the lowest no of transactions
# ***
# * Who does the most no of transactions (Males or Females) ?
#     * Females has the most no of transactions 
# ***
# * Which card type has the highest & least contribution to total amount ?
#     * Silver has the highest overall amount while gold is least 
# ***    
# * Which Exp Type has the highest & least contribution to total amount ?
#     * Bills has the highest overall amount while travel is least
# ***    
# * Which Gender Type has the highest & least contribution to total amount ?
#     * Female has the highest overall amount then men
# ***
# * Any subcategory in Exp type where no of transactions dominates for men
#     * Only in Fuel subcategory
# ***    
# * Any subcategory in Card type where no of transactions dominates for men
#     * No , All subcategories dominated by womens
# ***
# * Which Card type dominates most of the subcategories of Exp type
#     * Silver card holder dominates in all sub categories except Travel.
#     * Gold card holder has most transactions in travel sector
# ***    
# * Which exp type dominates most of the subcategories of Card Type
#     * Food is the most frequent category in all card types & travel being the least 
#     
# ***
# 


#mean and sd of transctn
mean_amount = df['Amount'].mean()
sdamt = df['Amount'].std()

# range for calculating probability
lower_bound = mean_amount - sdamt  # Lower bound of one standard deviation below the mean
upper_bound = mean_amount + sdamt  # Upper bound of one standard deviation above the mean

#range mai aane vaale transactions ka probability
from scipy.stats import norm
# Calculate the probability of transaction amounts falling within the defined range
probability_within_range = norm.cdf(upper_bound, loc=mean_amount, scale=sdamt) - norm.cdf(lower_bound, loc=mean_amount, scale=sdamt)

print("Probability of transaction amount being within one standard deviation of the mean:", probability_within_range)

# Calculate the expected value (mean) and variance of transaction amounts
mean_amount = df['Amount'].mean()
variance_amount = df['Amount'].var()

print("Expected Value (Mean) of Transaction Amounts:", mean_amount)
print("Variance of Transaction Amounts:", variance_amount)

# Separate transaction amounts for male and female customers
male_amounts = df[df['Gender'] == 'M']['Amount']
female_amounts = df[df['Gender'] == 'F']['Amount']

# Perform two-sample t-test
t_statistic, p_value = ttest_ind(male_amounts, female_amounts, equal_var=False)

# Define significance level
alpha = 0.05

# Print results
print("Two-Sample T-Test Results:")
print("T-Statistic:", t_statistic)
print("P-Value:", p_value)

# Interpret the results
if p_value < alpha:
    print("The difference in transaction amounts between male and female customers is statistically significant.")
else:
    print("There is no statistically significant difference in transaction amounts between male and female customers.")

