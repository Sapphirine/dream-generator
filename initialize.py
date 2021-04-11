import os
import torch
import pandas as pd
from scripts import processor
from tqdm import tqdm, trange

# get the directory to the video clips
directory = input("Enter the path to the video clips: ")
# check if the path exists
isExist = os.path.exists(directory)

if not isExist:
    print("File path does not exist")
else:
    # configure the ResNet model for classifying the images
    repo = 'pytorch/vision:v0.8.0'
    model = torch.hub.load(repo, 'resnet152', pretrained=True)
    with open("data/imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    proc = processor.Processor(model, categories)

    # create a dataframe for saving the results
    df = pd.DataFrame({'video name': [],
                       'Top 1 pred': [],
                       'Top 2 pred': [],
                       'Top 3 pred': [],
                       'ave red': [],
                       'ave green': [],
                       'ave blue': []})

    # start classifying the videos
    files = [i for i in os.listdir(directory) if i.endswith("mov")]
    # create a progress bar for tracking the progress
    pbar = tqdm(total=len(files))
    # processing
    for count, file in enumerate(files):
        # process the videos
        path = directory + '/' + file
        proc.load_video(path)
        # get the top predictions
        results = proc.classify_frames(skipping=10)
        # get the average RGB values
        ave_rgb = proc.get_average_rgb()
        # save to the data frame
        df = df.append({'video name': file,
                        'Top 1 pred': results[0],
                        'Top 2 pred': results[1],
                        'Top 3 pred': results[2],
                        'ave red': ave_rgb[0],
                        'ave green': ave_rgb[1],
                        'ave blue': ave_rgb[2]}, ignore_index=True)
        pbar.update(1)
    pbar.close()

    # save the dataframe
    df.to_csv(r'data/classified.csv', index=False)
