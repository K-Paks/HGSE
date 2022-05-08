def video_idcs_to_names(index_mapping, vid, idx):
    # get item
    item = index_mapping[(index_mapping['video_id'] == vid) & (index_mapping['index'] == idx)]

    # split item
    title = item['title'].item()
    name = item['name'].item()
    video_id = item['video_id'].item()
    time = item['time'].item()

    # transform time into a hh:mm:ss format
    time_h = int(time // 3600)
    time_m = int((time - time_h * 3600) // 60)
    time_s = int((time - time_h * 3600 - time_m * 60))
    time_start = f'{time_h:02d}:{time_m:02d}:{time_s:02d}'
    return title, name, video_id, int(time), time_start
