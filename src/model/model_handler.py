from src.model.nlp_model import NLPModel
from src.resources.data_controls import download_resources, load_resources
from src.resources.db_controls import DatabaseController


class ModelHandler:
    def __init__(self,
                 db_controller: DatabaseController,
                 model_class=NLPModel):
        download_resources()

        self.model_class = model_class
        self.embeddings, self.index_mapping = load_resources()
        self.db_controller = db_controller




