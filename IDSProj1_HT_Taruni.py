import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('C:\Users\tarun\Downloads\movieReplicationSet.csv')
print(data.shape)
print(data.head())



#Q2) Are movies that are newer rated differently than movies that are older? 
# Extract movie columns and year data
movie_columns = data.columns[data.columns.str.contains(r'\(\d{4}\)')]
years = movie_columns.str.extract(r'\((\d{4})\)').astype(float).squeeze()
median_year = years.median()

print(f"The median year is: {median_year:.0f}")

# Separate movies into older and newer based on the median year
older_movies = movie_columns[years <= median_year]
newer_movies = movie_columns[years > median_year]
average_ratings = data[movie_columns].mean()
newer_movies_average_rating = average_ratings[newer_movies].dropna()
older_movies_average_rating = average_ratings[older_movies].dropna()

# Check if there are enough data points for comparison
if not newer_movies_average_rating.empty and not older_movies_average_rating.empty:
    u_statistic, p_value = stats.mannwhitneyu(newer_movies_average_rating, older_movies_average_rating, alternative='two-sided')

    # Calculate Cohen's d for effect size
    mean_diff = newer_movies_average_rating.mean() - older_movies_average_rating.mean()
    pooled_std = np.sqrt(
        ((len(newer_movies_average_rating) - 1) * newer_movies_average_rating.var() +
         (len(older_movies_average_rating) - 1) * older_movies_average_rating.var()) /
        (len(newer_movies_average_rating) + len(older_movies_average_rating) - 2)
    )
    cohen_d = mean_diff / pooled_std

    # Interpret Cohen's d
    if abs(cohen_d) < 0.2:
        effect_size_interpretation = "very small or negligible"
    elif abs(cohen_d) < 0.5:
        effect_size_interpretation = "small"
    elif abs(cohen_d) < 0.8:
        effect_size_interpretation = "medium"
    else:
        effect_size_interpretation = "large"

    print(f"Average rating of newer movies: {newer_movies_average_rating.mean():.2f}")
    print(f"Average rating of older movies: {older_movies_average_rating.mean():.2f}")
    print(f'Mann-Whitney U: {u_statistic}, P-value: {p_value}')
    print(f'Effect size (Cohen\'s d): {cohen_d:.3f} ({effect_size_interpretation} effect)')

    # Plotting
    plt.figure(figsize=(5, 5))
    sns.histplot(newer_movies_average_rating, color='green', label='Newer Movies', kde=True, stat="density", bins=20)
    sns.histplot(older_movies_average_rating, color='blue', label='Older Movies', kde=True, stat="density", bins=20)
    plt.xlabel("Average Ratings")
    plt.ylabel("Density")
    plt.title("Distribution of Average Ratings: Newer vs Older Movies")
    plt.legend()
    plt.show()

    plt.figure(figsize=(5, 5))
    sns.boxplot(data=[older_movies_average_rating, newer_movies_average_rating], palette=["blue", "green"])
    plt.xticks([0, 1], ["Older Movies", "Newer Movies"])
    plt.ylabel("Average Rating")
    plt.title("Box Plot of Average Ratings: Newer vs Older Movies")
    plt.show()

    # Quartiles for further analysis
    q1_year, q3_year = years.quantile([0.25, 0.75])
    oldest_movies = movie_columns[years <= q1_year]
    mid_old_movies = movie_columns[(years > q1_year) & (years <= median_year)]
    mid_new_movies = movie_columns[(years > median_year) & (years <= q3_year)]
    newest_movies = movie_columns[years > q3_year]
    oldest_avg = average_ratings[oldest_movies].mean()
    mid_old_avg = average_ratings[mid_old_movies].mean()
    mid_new_avg = average_ratings[mid_new_movies].mean()
    newest_avg = average_ratings[newest_movies].mean()

    print(f"Average rating of oldest movies (<= Q1): {oldest_avg:.2f}")
    print(f"Average rating of mid-old movies (Q1 < year <= median): {mid_old_avg:.2f}")
    print(f"Average rating of mid-new movies (median < year <= Q3): {mid_new_avg:.2f}")
    print(f"Average rating of newest movies (> Q3): {newest_avg:.2f}")



#-------------------------------------------------------------------------------------------------------------------------------

# Q5) Do only children enjoy ‘The Lion King (1994)’ more than those with siblings?

lion_king_column = [col for col in movie_columns if 'The Lion King (1994)' in col][0]
only_child_column = 'Are you an only child? (1: Yes; 0: No; -1: Did not respond)'
only_child_ratings = data[data[only_child_column] == 1][lion_king_column].dropna()
siblings_ratings = data[data[only_child_column] == 0][lion_king_column].dropna()

if only_child_ratings.size > 0 and siblings_ratings.size > 0:
    u_statistic, p_value = stats.mannwhitneyu(only_child_ratings, siblings_ratings, alternative='two-sided')
    print(f'Mann-Whitney U: {u_statistic}, P-value: {p_value:.5f}')

    mean_diff = only_child_ratings.mean() - siblings_ratings.mean()
    pooled_std = np.sqrt(((len(only_child_ratings) - 1) * only_child_ratings.var() +
                          (len(siblings_ratings) - 1) * siblings_ratings.var()) /
                         (len(only_child_ratings) + len(siblings_ratings) - 2))
    cohen_d = mean_diff / pooled_std
    effect_size_interpretation = "very small or negligible" if abs(cohen_d) < 0.2 else "small" if abs(cohen_d) < 0.5 else "medium" if abs(cohen_d) < 0.8 else "large"
    print(f"Cohen's d: {cohen_d:.3f} ({effect_size_interpretation} effect)")

    plt.figure(figsize=(5, 5))
    sns.boxplot(data=[only_child_ratings, siblings_ratings], palette=["lightblue", "lightgreen"])
    plt.xticks([0, 1], ['Only Children', 'With Siblings'])
    plt.title("Ratings of 'The Lion King (1994)' by Family Structure")
    plt.ylabel('Rating')
    plt.xlabel('Group')
    plt.show()


#-----------------------------------------------------------------------------------------------------------------------------------------------------------

# Q6) What proportion of movies exhibit an “only child effect”, i.e. are rated different by viewers with siblings vs. those without?

significant_count = 0
significant_movies = []
significant_movies_cohen_d = []

for movie in movie_columns:
    only_child_ratings = data[data[only_child_column] == 1][movie].dropna()
    siblings_ratings = data[data[only_child_column] == 0][movie].dropna()

    if only_child_ratings.size > 0 and siblings_ratings.size > 0:
        u_statistic, p_value = stats.mannwhitneyu(only_child_ratings, siblings_ratings, alternative='two-sided')
        if p_value < 0.005:
            significant_count += 1
            mean_diff = only_child_ratings.mean() - siblings_ratings.mean()
            pooled_std = np.sqrt(((len(only_child_ratings) - 1) * only_child_ratings.var() +
                                  (len(siblings_ratings) - 1) * siblings_ratings.var()) /
                                 (len(only_child_ratings) + len(siblings_ratings) - 2))
            cohen_d = mean_diff / pooled_std
            effect_size_interpretation = "very small or negligible" if abs(cohen_d) < 0.2 else "small" if abs(cohen_d) < 0.5 else "medium" if abs(cohen_d) < 0.8 else "large"
            significant_movies_cohen_d.append((movie, cohen_d, effect_size_interpretation))

total_movies = len(movie_columns)
proportion_with_effect = significant_count / total_movies
print(f"Total movies analyzed: {total_movies}")
print(f"Movies with 'only child effect': {significant_count}")
print(f"Proportion of movies with 'only child effect': {proportion_with_effect:.2%}")
