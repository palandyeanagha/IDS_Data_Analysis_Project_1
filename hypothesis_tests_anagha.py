#Import Library
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, norm


#set directory
os.chdir("D:/Anagha/Masters/NYU GSAS/Semester 1/IDS/Data Analysis Project 1")

pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Adjust width to fit the screen
pd.set_option('display.max_colwidth', None)  # Show full content in each cell

#config
alpha = 0.005

#read movieReplicationSet (1).csv
df = pd.read_csv("movieReplicationSet.csv") 
columns = list(df.columns)
print("How many cols in total?:\n", len(columns))

## Info()
print("\n Information df: \n",df.info(verbose=True))
## describe()
print("\n Describe df: \n",df.describe())


### Q3
# Is enjoyment of ‘Shrek (2001)’ gendered, i.e. do male and female viewers rate it differently?
#H0 : There is no difference in rating for Shrek 2001 based on gender (male and female viewers)
#H1 : There is a difference in rating for Shrek 2001 based on gender (male and female viewers)

#data for this test
test3_df = pd.DataFrame(df[["Shrek (2001)","Gender identity (1 = female; 2 = male; 3 = self-described)"]])
print("Before data cleaning:", test3_df.shape)


#drop rows where both ratings and gender columns are empty
test3_df.dropna(subset=['Shrek (2001)', 'Gender identity (1 = female; 2 = male; 3 = self-described)'], how='all', inplace = True)
print("After data cleaning:",test3_df.shape)

# removing rows where gender = 3 (self described)
test3_df = test3_df[test3_df['Gender identity (1 = female; 2 = male; 3 = self-described)'] != 3]
print("After gender = 3 removed:",test3_df.shape)

#data cleaning - element-wise removal
#remove where gender is empty
test3_df.dropna(subset=['Gender identity (1 = female; 2 = male; 3 = self-described)'], how='all', inplace = True)
print("After removing rows where gender is NA:",test3_df.shape)

#remove where Shrek rating  is empty
test3_df.dropna(subset=['Shrek (2001)'], how='all', inplace = True)
print("After removing rows where Shrek rating is NA:",test3_df.shape)

#Plot of ratings original and ratings with dropna
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

sns.histplot(df['Shrek (2001)'], bins=10, kde=True, ax=axes[0], color='blue')
axes[0].set_title("Distribution of Ratings for Shrek (2001)")
axes[0].set_xlabel("Rating")
axes[0].set_ylabel("Frequency")
axes[0].grid(True)

#dropna
sns.histplot(test3_df['Shrek (2001)'], bins=10, kde=True, ax=axes[1], color='green')
axes[1].set_title("Distribution of Ratings for Shrek (2001) element-wise drop")
axes[1].set_xlabel("Rating dropna element-wise drop")
plt.tight_layout()
axes[1].grid(True)
plt.show()


#create two dfs: for male and  female
test3_df_female = test3_df[test3_df['Gender identity (1 = female; 2 = male; 3 = self-described)'] == 1].reset_index(drop=True)
test3_df_male = test3_df[test3_df['Gender identity (1 = female; 2 = male; 3 = self-described)'] == 2].reset_index(drop=True)



stat_q3, p_value_q3 = mannwhitneyu(test3_df_male['Shrek (2001)'], test3_df_female['Shrek (2001)'])
if p_value_q3 < alpha:
    print("There is a significant difference between male and female ratings.")
else:
    print("There is no significant difference between between male and female ratings.")
print("\np_value of Q3:", p_value_q3)
print("\nu-statistic stat_q3:", stat_q3)    


# Visualization
plt.figure(figsize=(10, 6))
plt.hist([test3_df_male['Shrek (2001)'], test3_df_female['Shrek (2001)']], bins=5, label=['Male', 'Female'])
plt.title('Shrek (2001) Ratings by Gender')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.boxplot([test3_df_male['Shrek (2001)'], test3_df_female['Shrek (2001)']], labels=['Male', 'Female'])
plt.title('Shrek (2001) Ratings by Gender')
plt.ylabel('Rating')
plt.show()

###############################################################################################################################

### Q4
# What proportion of movies are rated differently by male and female viewers?
# H0: There is no significant difference in movie ratings between male and female viewers.
# H1: There is a significant difference in movie ratings between male and female viewers.

significant_gender_difference_movies = [] #keeps track of movies
effect_sizes = {} #stores the effect size Cohen's d

# Loop through each movie column, perform the Mann-Whitney U Test between male and female ratings
for movie in columns[:400]:
    # Select relevant columns (movie ratings and gender)
    movie_data = df[[movie, 'Gender identity (1 = female; 2 = male; 3 = self-described)']].dropna()
    movie_data = movie_data[movie_data['Gender identity (1 = female; 2 = male; 3 = self-described)'].isin([1, 2])]
    female_ratings = movie_data[movie_data['Gender identity (1 = female; 2 = male; 3 = self-described)'] == 1][movie]
    male_ratings = movie_data[movie_data['Gender identity (1 = female; 2 = male; 3 = self-described)'] == 2][movie]
    
    if len(female_ratings) > 0 and len(male_ratings) > 0:
        u_statistic, p_value = stats.mannwhitneyu(female_ratings, male_ratings, alternative='two-sided')
        
        # Calculate Cohen's d effect size
        pooled_std = np.sqrt(((len(female_ratings) - 1) * np.var(female_ratings) + (len(male_ratings) - 1) * np.var(male_ratings)) / (len(female_ratings) + len(male_ratings) - 2))
        cohen_d = (np.mean(female_ratings) - np.mean(male_ratings)) / pooled_std
        
        # Check if p-value indicates a significant difference
        if p_value < alpha:
            significant_gender_difference_movies.append(movie)
            effect_sizes[movie] = cohen_d

#Finding proportions
total_movies = len(columns[:400])
proportion_significant = len(significant_gender_difference_movies) / total_movies

print("Proportion of movies rated different by male vs female viewers:", proportion_significant)
print("Number of movies where the ratings were different:", len(significant_gender_difference_movies))

# Print effect sizes for significant movies
print("Effect Sizes (Cohen's d):")
for movie, cohen_d in effect_sizes.items():
    print(f"{movie}: {cohen_d:.2f}")
    
# Print effect sizes for significant movies
print("Effect Sizes (Cohen's d):")
effect_sizes_list = []
movie_names = []
for movie, cohen_d in effect_sizes.items():
    print(f"{movie}: {cohen_d:.2f}")
    effect_sizes_list.append(cohen_d)
    movie_names.append(movie)

# Plot effect sizes
plt.figure(figsize=(10, 6))
plt.bar(movie_names, effect_sizes_list)
plt.xlabel('Movie')
plt.ylabel("Cohen's d")
plt.title('Effect Sizes for Significant Movies')
plt.xticks(rotation=90)  # Rotate x-axis labels for readability
plt.tight_layout()
plt.show()    


#Plot to understand Relative Frequency Distribution of Ratings by Gender Across All Movies
test4_plotdf = df.iloc[:, :400].copy()
test4_plotdf['Gender identity (1 = female; 2 = male; 3 = self-described)'] = df['Gender identity (1 = female; 2 = male; 3 = self-described)']
# Separate the ratings by gender
female_ratings = test4_plotdf[test4_plotdf['Gender identity (1 = female; 2 = male; 3 = self-described)'] == 1].iloc[:, :-1].values.flatten()  # All rating columns for females
male_ratings = test4_plotdf[test4_plotdf['Gender identity (1 = female; 2 = male; 3 = self-described)'] == 2].iloc[:, :-1].values.flatten()    # All rating columns for males

# Drop any missing values if necessary
female_ratings = female_ratings[~pd.isnull(female_ratings)]
male_ratings = male_ratings[~pd.isnull(male_ratings)]

female_ratings_pd=pd.DataFrame(female_ratings,columns={'ratings'})
female_ratings_pd['respondents']=1
female_ratings_pd_freq = female_ratings_pd.groupby('ratings').count().reset_index()
female_ratings_pd_freq ['respondents'] = female_ratings_pd_freq ['respondents']/female_ratings_pd.shape[0]
female_ratings_pd_freq['Gender'] = 'Female'
male_ratings_pd=pd.DataFrame(male_ratings,columns={'ratings'})
male_ratings_pd['respondents']=1
male_ratings_pd_freq = male_ratings_pd.groupby('ratings').count().reset_index()
male_ratings_pd_freq ['respondents'] = male_ratings_pd_freq['respondents']/male_ratings_pd.shape[0]
male_ratings_pd_freq['Gender'] = 'Male'

all_data = pd.concat([female_ratings_pd_freq,male_ratings_pd_freq])
plt.figure(figsize=(12, 6))
sns.barplot(x='ratings',y='respondents',data=all_data,hue='Gender')
plt.title('Relative Frequency Distribution of Ratings by Gender Across All Movies')
#plt.legend()
plt.show()

###############################################################################################################################
### Q9: Is the ratings distribution of ‘Home Alone (1990)’ different than that of ‘Finding Nemo (2003)’?
# H0: The two movies' ratings come from the same distribution.
# H1: The two movies' ratings come from different distributions.

#data
home_alone_ratings = df['Home Alone (1990)']
finding_nemo_ratings = df['Finding Nemo (2003)']

print("Sample size of Home Alone before data cleaning:",home_alone_ratings.shape )
print("Sample size of Finding Nemo before data cleaning:",finding_nemo_ratings.shape )

# removing rows that are null
print("Dropping rows with nulls:")
home_alone_ratings_clean = home_alone_ratings.dropna()
finding_nemo_ratings_clean = finding_nemo_ratings.dropna()
print("Sample size of Home Alone after data cleaning:",home_alone_ratings_clean.shape )
print("Sample size of Finding Nemo after data cleaning:",finding_nemo_ratings_clean.shape )

sstat_q9, pp_val_q9 = stats.ks_2samp(home_alone_ratings_clean, finding_nemo_ratings_clean)
print(f"KS Statistic: {sstat_q9}, p-value: {pp_val_q9}")
if pp_val_q9 < alpha:
    print("Ratings distributions of Home Alone and Finding Nemo differ.")
else:
    print("Ratings distributions of Home Alone and Finding Nemo are similar.")
    
    
#Visualization    
plt.figure(figsize=(8, 6))
sns.kdeplot(home_alone_ratings_clean, label='Home Alone')
sns.kdeplot(finding_nemo_ratings_clean, label='Finding Nemo')
plt.title('Ratings Distribution')
plt.xlabel('Rating')
plt.legend()
plt.show()

#Histogram along with KDE
plt.figure(figsize=(10, 6))
sns.histplot(home_alone_ratings_clean, bins=10, kde=True, label='Home Alone', alpha=0.5, color='orange')
sns.histplot(finding_nemo_ratings_clean, bins=10, kde=True, label='Finding Nemo', alpha=0.5, color='purple')
plt.title('Ratings Distribution Comparison')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.legend()
plt.show()

###############################################################################################################################
### Extra Credit: Do high-stressed individuals (stress levels 4, 5) rate the movie Finding Nemo(2003) differently than those with lower stress levels (1, 2, 3)?

# H0: There is no difference in the ratings of Finding Nemo between individuals with high stress levels (4, 5) and those with lower stress levels (1, 2, 3).
# H1: There is a significant difference in the ratings of Finding Nemo between individuals with high stress levels (4, 5) and those with lower stress levels (1, 2, 3).

test= pd.DataFrame(df[['Finding Nemo (2003)', 'My life is very stressful']])

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
# Plot the distribution for the first movie
sns.histplot(test['Finding Nemo (2003)'], bins=10, kde=True, ax=axes[0], color='blue')
axes[0].set_title("Distribution of Ratings for Finding Nemo (2003)")
axes[0].set_xlabel("Rating")
axes[0].set_ylabel("Frequency")

# Plot the distribution for stress
sns.histplot(test['My life is very stressful'], bins=10, kde=True, ax=axes[1], color='green')
axes[1].set_title("Distribution of Stress levels")
axes[1].set_xlabel("Stress level")
plt.tight_layout()
plt.show()

#Get ratings related to stress levels 4 and 5
high_stress_ratings = test[(test['My life is very stressful'] == 4) | (test['My life is very stressful'] == 5)]['Finding Nemo (2003)']
print("Number of high stress ratings:",len(high_stress_ratings) )
#Get ratings related to stress levels 1, 2 and 3
lower_stress_ratings = test[(test['My life is very stressful'] == 1) | (test['My life is very stressful'] == 2) | (test['My life is very stressful'] == 3)]['Finding Nemo (2003)']
print("Number of low stress ratings:",len(lower_stress_ratings))    

#handling missing data element wise
high_stress_ratings_dropna = high_stress_ratings.dropna()
print("Number of high stress ratings after dropna:",len(high_stress_ratings_dropna) )
lower_stress_ratings_dropna  = lower_stress_ratings.dropna()
print("Number of low stress ratings after dropna:",len(lower_stress_ratings_dropna))

# Checking missing data median - impute
high_stress_ratings_median = high_stress_ratings.fillna(high_stress_ratings.median())
print("Number of high stress ratings after median:",len(high_stress_ratings_median))
lower_stress_ratings_median  = lower_stress_ratings.fillna(lower_stress_ratings.median())
print("Number of low stress ratings after dropna:",len(lower_stress_ratings_median))
#Not using median impute


# Plot distributions
fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharey=True)

data_plot = [
    (high_stress_ratings, "Rating", "Frequency", "Distribution of Ratings high_stress_ratings", 'blue'),
    (lower_stress_ratings, "Rating", "Frequency", "lower_stress_ratings", 'green'),
    (high_stress_ratings_dropna, "Rating", "Frequency", "high_stress_ratings_element-wise removal", 'blue'),
    (lower_stress_ratings_dropna, "Rating", "Frequency", "lower_stress_ratings_element-wise removal", 'green'),
    (high_stress_ratings_median, "Rating", "Frequency", "Distribution of Ratings high_stress_ratings_median impute", 'blue'),
    (lower_stress_ratings_median, "Rating", "Frequency", "lower_stress_ratings_median impute", 'green')
]

for i, (ratings, xlabel, ylabel, title, color) in enumerate(data_plot):
    row = i // 3  # Change here
    col = i % 3   # Change here
    sns.histplot(ratings, bins=10, kde=True, ax=axes[row, col], color=color)
    axes[row, col].set_title(title)
    axes[row, col].set_xlabel(xlabel)
    if ylabel and row == 0:  # Change here
        axes[row, col].set_ylabel(ylabel)
    axes[row, col].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

#Mann-Whitney U test
stat_qe, p_value_qe = mannwhitneyu(high_stress_ratings_dropna, lower_stress_ratings_dropna)
if p_value_qe < alpha:
    print("There is a significant difference in the ratings of the movie Finding Nemo(2003) between individuals with high stress levels (4, 5) and those with lower stress levels (1, 2, 3).")
else:
    print("There is no significant difference in the ratings of the movie Finding Nemo(2003) between individuals with high stress levels (4, 5) and those with lower stress levels (1, 2, 3).")

print("\np_value:", p_value_qe)
print("\nu-statistic statistic:", stat_qe)
print("\nAverage movie rating from high stressed people(4,5):", high_stress_ratings_dropna.mean())
print("\nAverage movie rating for low stressed people(1,2,3):", lower_stress_ratings_dropna.mean())

#Effect size
# Calculate Cohen's d
cohen_d = (high_stress_ratings_dropna.mean() - lower_stress_ratings_dropna.mean()) / np.sqrt((high_stress_ratings_dropna.var() + lower_stress_ratings_dropna.var()) / 2)
print("\nCohen's d effect size:", cohen_d)

#Visualization
plt.figure(figsize=(8, 6))
plt.boxplot([high_stress_ratings_dropna, lower_stress_ratings_dropna], labels=['High-Stress', 'Low-Stress'])
plt.axhline(y=3.416798, color='blue', linestyle='--', label='High-Stress Average')
plt.axhline(y=3.340731, color='green', linestyle='--', label='Low-Stress Average')
plt.ylabel('Rating')
plt.title('Rating Distribution')
plt.legend()
plt.show()
###############################################################################################################################