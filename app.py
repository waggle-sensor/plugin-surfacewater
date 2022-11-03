
import torch
from torchvision import transforms

import time
import cv2
from PIL import Image
import argparse

from waggle.plugin import Plugin
from waggle.data.vision import Camera

TOPIC_SURFACEWATER = "env.binary.surfacewater"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-stream', dest='stream',
        action='store', default="bottom",
        help='ID or name of a stream, e.g. sample')
    parser.add_argument(
        '-model', dest='model',
        action='store', default='model.pth',
        help='Path to model')
    parser.add_argument(
        '-continuous', dest='continuous',
        action='store_true', default=False,
        help='Continuous run flag')
    parser.add_argument(
        '-sampling-interval', dest='sampling_interval',
        action='store', default=-1, type=int,
        help='Sampling interval between inferencing')
    parser.add_argument(
        '-debug', dest='debug',
        action='store_true', default=False,
        help='Debug flag')

    return parser.parse_args()


def run(model, sample, do_sampling, plugin):
    image = sample.data
    image = image[250:850, 300:1200]
    timestamp = sample.timestamp

    transformation = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize((224,224)),
                                         transforms.ToTensor()])
    img = transformation(image).unsqueeze(0)
    if args.device == 'cpu':
        img = img.to(device='cpu')
    elif args.device == 'cuda':
        img = img.to(device='cuda')
    pred = model(img)

    #prob = torch.sigmoid(pred).squeeze(0)
    _, result = torch.max(pred, 1)

    if result == 0:
        print('nonwater')
    elif result == 1:
        print('water')

    plugin.publish(TOPIC_SURFACEWATER, result.item(), timestamp=timestamp)
    print(f"Standing Water: {result.item()} at time: {timestamp}")

    if do_sampling:
        sample.data = image
        sample.save('sample.jpg')
        plugin.upload_file('sample.jpg')
        print('saved')


if __name__ == '__main__':
    args = get_args()
    if torch.cuda.is_available():
        args.device = 'cuda'
        model = torch.load(args.model, map_location='cuda')
    else:
        args.device = 'cpu'
        model = torch.load(args.model, map_location='cpu')
    model.eval()

    sampling_countdown = -1
    if args.sampling_interval >= 0:
        sampling_countdown = args.sampling_interval

    while True:
        with Plugin() as plugin, Camera(args.stream) as camera:
            sample = camera.snapshot()

            do_sampling = False
            if sampling_countdown > 0:
                sampling_countdown -= 1
            elif sampling_countdown == 0:
                print('here')
                do_sampling = True
                sampling_countdown = args.sampling_interval

            run(model, sample, do_sampling, plugin)
            if not args.continuous:
                exit(0)   ## oneshot
