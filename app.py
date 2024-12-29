import numpy as np
import pandas as pd
from shiny import App, ui, reactive, render
import joblib
import ast

unique_genres = ['Thriller', 'Horror', 'Shounen', 'Samurai', 'Supernatural', 'Psychological',
                 'Hentai', 'Cars', 'Dementia', 'Mecha', 'Ecchi', 'Mystery', 'Romance', 'Demons',
                 'School', 'Martial Arts', 'Seinen', 'Shoujo Ai', 'Music', 'Vampire', 'Game',
                 'Shounen Ai', 'Comedy', 'Magic', 'Space', 'Kids', 'Police', 'Harem', 'Adventure',
                 'Yaoi', 'Drama', 'Historical', 'Sci-Fi', 'Military', 'Shoujo', 'Slice of Life',
                 'Parody', 'Super Power', 'Yuri', 'Fantasy', 'Josei', 'Action', 'Sports']
anime_category = ['Movie', 'Series', 'Movie/Special', 'Special', 'Short Series']
aired_decade = ["<=1950", "1950s", "1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s"]
no_of_episodes = ['1-12', '13-26', '27-60', '61-100', '101-300', '301+']

app_ui = ui.page_fluid(
    ui.h2("Anime Recommendation System"),
    ui.row(
        ui.column(
            6,
            ui.input_slider("score", "Preferred Score:", min=0, max=10, value=8, step=0.1),
            ui.input_selectize(
                "genres",
                "Preferred Genres:",
                choices=unique_genres,
                multiple=True
            ),
            ui.input_selectize(
                "aired",
                "Preferred aired:",
                choices=aired_decade,
                multiple=True
            ),
            ui.input_selectize(
                "Anime_Category",
                "Preferred Anime Category:",
                choices=anime_category,
                multiple=True
            ),
            ui.input_selectize(
                "Number_of_Episodes",
                "Preferred number of Episodes:",
                choices=no_of_episodes,
                multiple=True
            ),
            ui.input_action_button("submit", "Get Recommendations", class_="btn-primary")
        ),
        ui.column(
            6,
            ui.output_data_frame("recommendations_table")
        )
    )
)


def recommend_anime(user_input, df_anime, kmeans_model, minmax_scaler, features):
    user_vector = np.hstack([
        np.array([user_input['score']]),
        np.array(user_input['genres_encoded']),
        np.array(user_input['encoded_title_category']),
        np.array(user_input['encoded_aired_decade']),
        np.array(user_input['encoded_episodes'])
    ])

    user_vector = user_vector.reshape(1, -1)
    user_vector_scaled = minmax_scaler.transform(user_vector)
    user_cluster = kmeans_model.predict(user_vector_scaled)[0]
    cluster_anime = df_anime[kmeans_model.labels_ == user_cluster].copy()

    cluster_anime['similarity'] = cluster_anime.apply(
        lambda row: np.linalg.norm(
            user_vector_scaled - minmax_scaler.transform(np.hstack([row[f] for f in features]).reshape(1, -1))),
        axis=1
    )
    recommendations = cluster_anime.sort_values(by=['similarity', 'score'], ascending=True).head(5)
    return recommendations[['title', 'genre', 'aired_year', 'episodes', 'score', 'similarity']]


def server(input, output, session):
    @reactive.Calc
    @reactive.event(input.submit)
    def recommendations():
        user_input = {
            "score": input.score(),
            "genres_encoded": [1 if g in input.genres() else 0 for g in unique_genres],
            "encoded_title_category": [1 if g in input.Anime_Category() else 0 for g in anime_category],
            "encoded_aired_decade": [1 if g in input.aired() else 0 for g in aired_decade],
            "encoded_episodes": [1 if g in input.Number_of_Episodes() else 0 for g in no_of_episodes]
        }
        features = ['score', 'genres_encoded',
                    'encoded_title_category', 'encoded_aired_decade',
                    'encoded_episodes']
        anime_file_path = 'df_animes_updated.csv'
        df_anime = pd.read_csv(anime_file_path)

        for feature in features:
            if feature in df_anime.columns:
                df_anime[feature] = df_anime[feature].apply(
                    lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else x
                )

        loaded_Kmeans_model = joblib.load("kmeans_model.pkl")
        loaded_minmax_scaler = joblib.load("scaler.pkl")
        return recommend_anime(user_input, df_anime, loaded_Kmeans_model, loaded_minmax_scaler, features)

    @output
    @render.data_frame
    @reactive.event(input.submit)
    def recommendations_table():
        recs = recommendations()  # Fetch the recommendations
        if recs is None or recs.empty:
            return pd.DataFrame({"Message": ["No recommendations found."]})
        return render.DataGrid(recs[['title', 'score', 'genre', 'aired_year', 'episodes', 'similarity']])


app = App(app_ui, server)
