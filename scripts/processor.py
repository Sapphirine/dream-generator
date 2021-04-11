from decord import VideoReader
from decord import cpu, gpu
from PIL import Image
from torchvision import transforms
import torch
import numpy as np

col_labels = ['Top 1 pred', 'Top 2 pred', 'Top 3 pred']


class Processor:
    def __init__(self, model, categories, labeled_dataset=None):
        '''
        https://pytorch.org/hub/pytorch_vision_resnet/
        Classification algorithm based on the ResNet model
        Args:
            model: the trained model
            categories: classification labels
        '''
        self.model = model
        self.categories = categories
        self.model.eval()

    def classify(self, image):
        '''
        Classifies the input image
        Args:
            image: the input image object

        Returns:
            the most probable label based on the ResNet result
            the return should be an integer value
        '''
        # input image must be resized and normalized so that it is the same as the trained model
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            self.model.to('cuda')

        # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
        with torch.no_grad():
            output = self.model(input_batch)

        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Show the most probable class
        top_prob, top_catid = torch.topk(probabilities, 3)
        return top_catid.numpy(), top_prob.numpy()

    def classify_frames(self, skipping=1):
        '''
        function that can iterate through the video frames and classify each frame.
        Args:
            skipping: number of frames to skip while iterating through the video

        Returns:
            The top 3 labels with the highest probabilities.

        '''
        # transform the frames to numpy format
        frames = self.get_frames(skipping).asnumpy()

        results = dict()  # dictionary for storing the classification results
        for frame in frames:
            labels, prob = self.classify(Image.fromarray(frame))
            for i in range(len(labels)):
                if labels[i] in results:
                    results[labels[i]] = results[labels[i]] + prob[i]
                else:
                    results[labels[i]] = prob[i]

        # sort the dictionary in descending order
        sorted_list = sorted(results.items(), key=lambda x: x[1], reverse=True)
        # only return the labels
        # return [item[0] for item in sorted_list]
        return [sorted_list[i][0] for i in range(len(sorted_list)) if i < 3]

    def load_video(self, path):
        '''
        https://github.com/dmlc/decord#installation
        https://github.com/dmlc/decord/blob/master/examples/video_reader.ipynb
        A decord wrapper implemented per the instruction
        Load the video as an object
        Args:
            path: the path to the video file

        Returns:
            none
        '''
        self.vr = VideoReader(path, width=320, height=240, ctx=cpu(0))

    def get_frames(self, skipping=1):
        '''
        Get the sampled frames from the input video
        Args:
            skipping: number of frames to skip in when sampling

        Returns:
            the frames sampled from the video
        '''
        return self.vr.get_batch(range(0, len(self.vr) - 1, skipping))

    def get_average_rgb(self):
        '''
        Get the average RGB values for each color channel
        Args:
            None

        Returns:
            average RGB values
        '''
        frames = self.get_frames().asnumpy()
        rgb = np.zeros(3)
        for i in range(3):
            rgb[i] = np.mean(frames[:, :, :, i])

        return rgb

