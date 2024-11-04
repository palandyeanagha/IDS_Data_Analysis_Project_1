# Import necessary libraries
import pandas as pd
from scipy.stats import mannwhitneyu, kruskal
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def main():
    # Q1: Are movies that are more popular rated higher than less popular movies?
    # Using Pathlib
    file_path = Path('C:/Users/91838/Downloads/movieReplicationSet.csv')  # Use forward slashes or double backslashes

    # Now you can read the CSV file
    data = pd.read_csv(file_path)

    # Step 1: Calculate the number of ratings (popularity) for each movie (columns 1 to 400)
    movie_columns = data.columns[:400]
    popularity_counts = data[movie_columns].notna().sum()

    # Step 2: Determine the median of the popularity counts
    median_popularity = popularity_counts.median()

    # Step 3: Divide movies into high and low popularity based on the median
    high_popularity_movies = popularity_counts[popularity_counts > median_popularity].index
    low_popularity_movies = popularity_counts[popularity_counts <= median_popularity].index

    # Step 4: Calculate the mean rating for each movie in high and low popularity groups
    high_popularity_means = data[high_popularity_movies].mean()
    low_popularity_means = data[low_popularity_movies].mean()

    # Step 5: Perform the Mann-Whitney U Test on the mean ratings
    u_statistic, p_value = mannwhitneyu(high_popularity_means, low_popularity_means, alternative='two-sided')

    # Display the results
    print("Mann-Whitney U Test Statistic:", u_statistic)
    print("p-value:", p_value)

    # Step 1: Calculate the overall mean rating for high and low popularity groups
    overall_high_popularity_mean = high_popularity_means.mean()
    overall_low_popularity_mean = low_popularity_means.mean()

    print("Overall Mean Rating for High Popularity Movies:", overall_high_popularity_mean)
    print("Overall Mean Rating for Low Popularity Movies:", overall_low_popularity_mean)

    # Prepare data for visualization
    popularity_data = pd.DataFrame({
        "Mean Rating": pd.concat([high_popularity_means, low_popularity_means]),
        "Popularity Group": ["High"] * len(high_popularity_means) + ["Low"] * len(low_popularity_means)
    })

    # Plot boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Popularity Group", y="Mean Rating", data=popularity_data)
    plt.title("Distribution of Mean Ratings for High and Low Popularity Movies")
    plt.ylabel("Mean Rating")
    plt.xlabel("Popularity Group")
    plt.show()

    # Q7: Do people who like to watch movies socially enjoy The Wolf of Wall Street (2013) more than those who prefer to watch alone?
    wolf_wall_street_column = 'The Wolf of Wall Street (2013)'
    social_watching_column = 'Movies are best enjoyed alone (1: Yes; 0: No; -1: Did not respond)'

    cleaned_q7_data = data[[wolf_wall_street_column, social_watching_column]].dropna()
    cleaned_q7_data = cleaned_q7_data[cleaned_q7_data[social_watching_column].isin([0, 1])]

    social_watch_ratings = cleaned_q7_data[cleaned_q7_data[social_watching_column] == 0][wolf_wall_street_column]
    alone_watch_ratings = cleaned_q7_data[cleaned_q7_data[social_watching_column] == 1][wolf_wall_street_column]

    u_statistic_q7, p_value_q7 = mannwhitneyu(social_watch_ratings, alone_watch_ratings, alternative='two-sided')

    print("Mann-Whitney U Test Statistic for Wolf of Wall Street:", u_statistic_q7)
    print("p-value for Wolf of Wall Street:", p_value_q7)

    # Visualization of the distribution of ratings
    plt.figure(figsize=(14, 6))
    popularity_data = pd.DataFrame({
        'Mean Rating': pd.concat([social_watch_ratings, alone_watch_ratings]),
        'Preference Group': ['Social Watchers'] * len(social_watch_ratings) + ['Watch Alone'] * len(alone_watch_ratings)
    })
    sns.boxplot(x='Preference Group', y='Mean Rating', data=popularity_data)
    plt.ylabel('Ratings')
    plt.title('Distribution of Ratings for "The Wolf of Wall Street (2013)"')
    plt.show()

    # Q8: What proportion of movies exhibit a “social watching” effect?
    filtered_data_q8 = data[data[social_watching_column].isin([0, 1])]
    significant_social_watching_movies = []

    all_social_watch_ratings = []
    all_alone_watch_ratings = []

    for movie in movie_columns:
        movie_data = filtered_data_q8[[movie, social_watching_column]].dropna()
        social_watch_ratings = movie_data[movie_data[social_watching_column] == 0][movie]
        alone_watch_ratings = movie_data[movie_data[social_watching_column] == 1][movie]

        all_social_watch_ratings.extend(social_watch_ratings)
        all_alone_watch_ratings.extend(alone_watch_ratings)

        if len(social_watch_ratings) > 0 and len(alone_watch_ratings) > 0:
            u_statistic, p_value = mannwhitneyu(social_watch_ratings, alone_watch_ratings, alternative='two-sided')
            if p_value < 0.005:
                significant_social_watching_movies.append(movie)

    number_of_significant_movies = len(significant_social_watching_movies)
    proportion_significant_movies = number_of_significant_movies / len(movie_columns) if len(movie_columns) > 0 else 0

    print("Number of Movies with Significant Social Watching Effect:", number_of_significant_movies)
    print("Proportion of Significant Movies:", proportion_significant_movies)

    # Overall Visualization of Ratings for Both Groups
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.hist(all_social_watch_ratings, bins=np.arange(0, 5, 0.5), alpha=0.5, label='Social Watchers', density=True)
    plt.hist(all_alone_watch_ratings, bins=np.arange(0, 5, 0.5), alpha=0.5, label='Watch Alone', density=True)
    plt.xlabel('Ratings')
    plt.ylabel('Density')
    plt.title('Overall Rating Distribution')
    plt.legend()

    plt.subplot(1, 2, 2)
    overall_data = pd.DataFrame({
        'Ratings': all_social_watch_ratings + all_alone_watch_ratings,
        'Preference Group': ['Social Watchers'] * len(all_social_watch_ratings) + ['Watch Alone'] * len(all_alone_watch_ratings)
    })
    sns.boxplot(x='Preference Group', y='Ratings', data=overall_data)
    plt.title('Overall Ratings Distribution')
    plt.ylabel('Ratings')
    plt.xlabel('Preference Group')

    plt.tight_layout()
    plt.show()

    # Q10: Are all movies within a franchise rated equally?
    franchises = {
        'Star Wars': ['Star Wars'],
        'Harry Potter': ['Harry Potter'],
        'The Matrix': ['Matrix'],
        'Indiana Jones': ['Indiana Jones'],
        'Jurassic Park': ['Jurassic Park'],
        'Pirates of the Caribbean': ['Pirates of the Caribbean'],
        'Toy Story': ['Toy Story'],
        'Batman': ['Batman']
    }

    franchise_movies = {franchise: [] for franchise in franchises.keys()}

    for column in data.columns:
        for franchise, keywords in franchises.items():
            if any(keyword in column for keyword in keywords):
                franchise_movies[franchise].append(column)

    rating_values = []
    franchise_labels = []

    for franchise, movies in franchise_movies.items():
        for movie in movies:
            ratings = data[movie].dropna().values
            rating_values.extend(ratings)
            franchise_labels.extend([franchise] * len(ratings))

    ratings_df = pd.DataFrame({
        'Rating': rating_values,
        'Franchise': franchise_labels
    })

    # Group ratings by franchise for the Kruskal-Wallis test
    grouped_ratings = [ratings_df[ratings_df['Franchise'] == franchise]['Rating'] for franchise in franchises.keys()]
    kruskal_statistic, p_value = kruskal(*grouped_ratings)

    print("Kruskal-Wallis Test Statistic:", kruskal_statistic)
    print("p-value for Kruskal-Wallis Test:", p_value)

    # Create a boxplot to visualize the distribution of ratings for each franchise
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Franchise', y='Rating', data=ratings_df, hue='Franchise', palette='Set2', legend=False)
    plt.title('Distribution of Movie Ratings by Franchise', fontsize=16)
    plt.xlabel('Franchise', fontsize=14)
    plt.ylabel('Rating', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
