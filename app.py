import torch
from torchvision import transforms

import argpares

import waggle.plugin as plugin
from waggle.data import open_data_source

TOPIC_SURFACEWATER = "env.coverage.cloud"

def get_args()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-debug', dest='debug',
        action='store_true', default=False,
        help='Debug flag')
    parser.add_argument(
        '-model', dest='model',
        action='store', required=True,
        help='Path to model')
    parser.add_argument(
        '-interval', dest='interval',
        action='store', default=0, type=int,
        help='Inference interval in seconds')
    parser.add_argument(
        '-sampling-interval', dest='sampling_interval',
        action='store', default=-1, type=int,
        help='Sampling interval between inferencing')
    return parser.parse_args()


def run(model, i, yes, no, n, w):
    camera = Camera(args.stream)
    sample = camera.snapshot()
    image = sample.data
    timestamp = sample.timestamp

    transformation = transforms.Compose([transforms.Resize((224,224)),
                                         transforms.ToTensor()])
    img = transformation(image).unsqueeze(0)
    img = img.to(device='cpu')
    pred = model(img)

    #prob = torch.sigmoid(pred).squeeze(0)
    _, result = torch.max(pred, 1)

    if result == 0:
        print('nonwater')
    elif result == 1:
        print('water')

    plugin.publish(TOPIC_SURFACEWATER, result, timestamp=timestamp)
    print(f"Standing Water: {result} at time: {timestamp}")
    cv2.imwrite('street.jpg', image)
    plugin.upload_file('street.jpg')
    print('saved')


if __name__ == '__main__':
    args = get_args()
    model = torch.load(args.model)

    while True:
        run(model, i)
        exit(0)   ## oneshot
