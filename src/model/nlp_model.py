import pandas as pd
from sentence_transformers.util import dot_score

from src.resources.data_controls import download_resources, load_resources, load_transformer
from src.utils.util_funcs import video_idcs_to_names


class NLPModel:
    def __init__(self):
        self.model = load_transformer()

    def calculate_similarity(self, query, embeddings):
        query_embedding = self.model.encode(query)
        scores = dot_score(query_embedding, embeddings.values)
        score_df = pd.DataFrame(scores.numpy().squeeze(), index=embeddings.index)
        score_df = score_df.reset_index()
        score_df.columns = ['video_id', 'video_part_id', 'score']
        score_df = score_df.sort_values(by='score', ascending=False)
        return score_df

    def suggest(self, score_df, index_mapping):
        suggested_videos = []
        suggested_video_ids = []
        for video in score_df.values:
            video_id, index, score = video

            if video_id in suggested_videos:
                continue

            title, name, video_id, time_raw, time_start = video_idcs_to_names(index_mapping, vid=video_id,
                                                                              idx=index)
            suggestion = Suggestion(title, name, video_id, time_raw, time_start)
            suggested_videos.append(suggestion)
            suggested_video_ids.append(video_id)

            if len(suggested_videos) == 10:
                break
        return suggested_videos


class Suggestion:
    def __init__(self, title, name, video_id, time_raw, time_start):
        self.title = title
        self.name = name
        self.video_id = video_id
        self.time_raw = time_raw
        self.time_start = time_start
