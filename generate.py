from scripts import generator
import numpy as np
import pandas as pd
import sys
import os
import shutil


predictions = ['Top 1 pred', 'Top 2 pred', 'Top 3 pred']
rgb_values = ['ave red', 'ave green', 'ave blue']


def get_next_video(dataframe, picked):
    # find the videos with the same classification
    for i in range(1, 3, 1):
        for j in range(2):
            match = dataframe.loc[dataframe[predictions[j]] == picked.iloc[0, i]]

    if match.empty:
        # todo future improvments
        # if no match found, find the video that has the similar RGB values
        #rgb_diff = np.sum((np.absolute(picked.iloc[0, [4, 5, 6]] - dataframe.iloc[:, [4, 5, 6]])), axis=1)
        # match = dataframe.loc[rgb_diff.index[rgb_diff.argmin()]]

        # sample a random one
        match = dataframe.sample()
    else:
        # pick one random match as the output
        match = match.sample()

    return match


# get the path to the dataset
data_path = sys.argv[1]
vid_len_in_second = sys.argv[2]
vid_frame_rate = sys.argv[3]

isExist = os.path.exists(data_path)
isClassified = isExist or os.path.exists('data/classified.csv')

# check if the path exists
if not isExist:
    print("Dataset path invalid")
elif not isClassified:
    print("Dataset classification file not found")
else:
    # configure the generator
    gen = generator.Generator(dataset_path=data_path)
    gen.set_dream_len(int(vid_len_in_second), int(vid_frame_rate))
    df = pd.read_csv('data/classified.csv')

    # make a new directory for storing temp data
    output_path = 'data/translated'
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
        os.mkdir(output_path)
    else:
        os.mkdir(output_path)

    # start the process from a random input video
    picked = df.sample()
    df = df.drop(picked.index)
    name = picked.iloc[0, 0]
    path = os.path.join(data_path, name)
    gen.set_new_video(path)

    while not gen.is_dream_terminated():
        if df.empty:
            print('no more input videos available')
        else:
            # translate the video
            gen.process('style_cezanne_pretrained')
            # load the next video with similar contents
            picked = get_next_video(df, picked)
            df = df.drop(picked.index)
            name = picked.iloc[0, 0]
            path = os.path.join(data_path, name)
            gen.set_new_video(path)

    # generate the final output video
    dir = os.listdir(output_path)
    if len(dir) == 0:
        print('no files to generate the output video')
    else:
        gen.generate_video(output_path)

