from decord import VideoReader
from decord import cpu, gpu
import moviepy.video.io.ImageSequenceClip
import os
import shutil
import subprocess
import torch
from PIL import Image


def transform(model, total_frames):
    gpu_avail = '-1'
    if torch.cuda.is_available():
        gpu_avail = '1'

    # generate the translated images
    subprocess.run(['python', 'extern/pytorch-CycleGAN-and-pix2pix/test.py', '--dataroot',
                    'data/raw', '--name', model, '--model', 'test', '--no_dropout', '--num_test', str(total_frames),
                    '--results_dir', 'data/tmp', '--gpu_ids', gpu_avail, '--aspect_ratio', '0.75'])


class Generator:
    def __init__(self, dataset_path):
        self.counter = 0
        self.dataset = dataset_path
        self.models = os.listdir('checkpoints')
        self.dream_len = 0
        self.frame_rate = 0
        self.vr = None

    def set_dream_len(self, len_in_seconds=60, frame_rate=30):
        self.dream_len = len_in_seconds * frame_rate
        self.frame_rate = frame_rate
        self.counter = 0

    def is_dream_terminated(self):
        if self.counter < self.dream_len:
            return False
        else:
            return True

    def set_new_video(self, path):
        self.vr = VideoReader(path, width=320, height=240, ctx=cpu(0))

    def get_total_frame_count(self):
        return len(self.vr)

    def get_a_frame(self, frame):
        if frame < self.get_total_frame_count():
            return self.vr[frame].asnumpy()-1
        else:
            print('frame is out of range')

    def get_frames(self, skipping=1):
        return self.vr.get_batch(range(0, len(self.vr) - 1, skipping))

    def process(self, model_name):
        # make a new directory for storing temp data
        if os.path.exists('data/raw'):
            shutil.rmtree('data/raw')
            os.mkdir('data/raw')
        else:
            os.mkdir('data/raw')

        # check to make sure the model is available
        if model_name not in self.models:
            print('specified model not available')
            return
        else:
            frame_count = 0
            total_frames = self.get_total_frame_count()
            # decompose a video into frames
            while not self.is_dream_terminated():
                if frame_count < total_frames:
                    img = Image.fromarray(self.get_a_frame(frame_count))
                    img.save('data/raw/' + '{num:05d}'.format(num=self.counter) + '.png')
                    frame_count += 1
                    self.counter += 1
                else:
                    break

            # start translating the images
            transform(model_name, total_frames)
            # copy the results to the translated folder
            path = os.path.join('data/tmp', model_name, 'test_latest/images')
            files = [img for img in os.listdir(path) if img.endswith(".png")]
            for file in files:
                if 'fake' in file:
                    # copy the translated images to the translated folder
                    shutil.copy(os.path.join(path, file), 'data/translated')

    def generate_video(self, path):
        image_files = [path + '/' + img for img in os.listdir(path) if img.endswith(".png")]
        image_files.sort()
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=self.frame_rate)
        clip.write_videofile('results/dream.mp4')
