import requests
import json
import os
from os.path import join
from time import sleep
import re
from collections import defaultdict

from tqdm import tqdm
import numpy as np
import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

RAW_DATAPATH = join('data', 'raw_jsons')
CAPTIONPATH = join('data', 'captions')

with open('secrets.json', 'r') as f:
    key = json.load(f)['key']
url = 'https://www.googleapis.com/youtube/v3/playlistItems'

params = {
    'playlistId': 'UUlHVl2N3jPEbkNJVx-ItQIQ',
    'key': key,
    'part': 'snippet',
    'maxResults': 50
}

videos = []

for _ in tqdm(range(10)):
    resp = requests.get(url, params)
    resp_json = json.loads(resp.text)

    items = resp_json['items']

    for item in items:
        name = item['id'] + '.json'
        name = join(RAW_DATAPATH, name)
        with open(name, 'w') as f:
            json.dump(item, f)
        params['pageToken'] = resp_json.get('nextPageToken')

metadata = []
for file in tqdm(os.listdir(RAW_DATAPATH)[:]):
    filepath = join(RAW_DATAPATH, file)

    with open(filepath, 'r') as f:
        item = json.load(f)

    snippet = item['snippet']
    desc = snippet['description']

    # get the timestamps
    desc_lines = desc.split('\n')
    match_str = r'^\d{2}:\d{2}'
    tss = [dl for dl in desc_lines if re.match(match_str, dl)]
    timestamps = []

    for ts in tss:
        ts_split = ts.split(' ', 1)
        hms = ts_split[0].split(':')  # hours, minutes, seconds
        hms = np.array(hms, dtype=np.float32)
        if len(hms) == 2:
            time_in_seconds = hms @ np.array([60, 1])
        else:
            time_in_seconds = hms @ np.array([3600, 60, 1])

        timestamps.append((time_in_seconds, ts_split[1]))

    timestamps = pd.DataFrame(timestamps, columns=['time', 'name'])

    # get and fix the title
    title = snippet['title']
    match_str = r'[\|\?,\.\+\/\\\"\'\"\&\:]'
    title_fixed = re.sub(match_str, '', title)
    title_fixed = re.sub(r'\s+', ' ', title_fixed)
    title_fixed = title_fixed.replace(' ', '_')

    # get captions
    video_id = snippet['resourceId']['videoId']
    # try:
    #     captions = YouTubeTranscriptApi.get_transcript(video_id)
    #
    #     if not timestamps.empty:
    #         # match captions with timestamps
    #         cap_dict = defaultdict(list)
    #
    #         for cap in captions:
    #             videopart_no = sum(cap['start'] > timestamps['time']) - 1
    #             cap_dict[videopart_no].append(cap['text'])
    #
    #         for cap_dict_key in cap_dict.keys():
    #             cap_dict[cap_dict_key] = ' '.join(cap_dict[cap_dict_key])
    #
    #         cap_df = pd.DataFrame(cap_dict, index=['captions']).T
    #
    #         cap_ts_df = timestamps.join(cap_df)
    #     else:
    #         captions = [cap['text'] for cap in captions]
    #         cap_ts_df = pd.DataFrame([[0, 'None', ' '.join(captions)]], columns=['time', 'name', 'captions'])
    #
    #     # save to csv
    #     cap_ts_df.to_csv(join(CAPTIONPATH, title_fixed + '.csv'), index=False)
    #     success = True
    #
    # except Exception as e:
    #     print(f'Failed for {title}: \n {e}')
    #     success = False
    metadata.append((title_fixed, title, file, video_id, True))

metadf = pd.DataFrame(metadata, columns=['fixed_title', 'title', 'id', 'video_id', 'success'])
metadf.to_csv(join('data', 'metadata.csv'), index=False)
