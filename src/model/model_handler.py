import pandas as pd

from src.model.nlp_model import NLPModel
from src.resources.data_controls import download_resources, load_resources
from src.resources.db_controls import DatabaseController


class ModelHandler:
    def __init__(self,
                 db_controller: DatabaseController):
        download_resources()
        self.db_controller = db_controller

        self.model = NLPModel()

        self.db_controller.prepare_db()
        self.db_controller.populate_db()

    def get_scores(self, query):
        embeddings = self.db_controller.get_embeddings()
        scores_df = self.model.calculate_similarity(query, embeddings)
        return scores_df

    def get_suggestions(self, score_df):
        mapping = self.db_controller.get_mapping()
        suggestions = self.model.suggest(score_df, index_mapping=mapping)
        return suggestions






