from __future__ import print_function

import sys
import argparse


def pipe2file(s, filename):
    with open(filename, 'a') as fhandle:
        print(s, file=fhandle)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Display CNN structure")
    cnn_choices = ["xception", "vgg16", "vgg19", "inception_v3", "inception_resnet_v2", "resnet50"]
    parser.add_argument('network_name', type=str, choices=cnn_choices, help='CNN name')
    parser.add_argument('output', type=str, help='output filename')
    args = vars(parser.parse_args())

    from keras.applications.xception import Xception
    from keras.applications.vgg16 import VGG16
    from keras.applications.vgg19 import VGG19
    from keras.applications.inception_v3 import InceptionV3
    from keras.applications.inception_resnet_v2 import InceptionResNetV2
    from keras.applications.resnet50 import ResNet50

    model_map = {"xception": Xception, "vgg16": VGG16,  "vgg19": VGG19, "inception_v3": InceptionV3,
                 "inception_resnet_v2": InceptionResNetV2, "resnet50": ResNet50}
    model_map[args["network_name"]](include_top=True).summary(print_fn=lambda s: pipe2file(s, args["output"]))
