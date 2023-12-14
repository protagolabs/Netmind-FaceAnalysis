import argparse
import os
import cv2
import numpy as np
import torch

from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise
)
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)

from backbones import get_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--imageA-path',
        type=str,
        default='./examples/0.png',
        help='Input image path')
    parser.add_argument(
        '--imageB-path',
        type=str,
        default='./examples/1.png',
        help='Input image path')
    parser.add_argument('--aug-smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen-smooth',
        action='store_true',
        help='Reduce noise by taking the first principle component'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=[
                            'gradcam', 'hirescam', 'gradcam++',
                            'scorecam', 'xgradcam', 'ablationcam',
                            'eigencam', 'eigengradcam', 'layercam',
                            'fullgrad', 'gradcamelementwise'
                        ],
                        help='CAM method')

    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory to save the images')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


class SimilarityToConceptTarget:
    def __init__(self, features):
        self.features = features
    
    def __call__(self, model_output):
        cos = torch.nn.CosineSimilarity(dim=0)
        # return cos(model_output, self.features)
        return cos(model_output, self.features)
    
# A model wrapper that gets a resnet model and returns the features before the fully connected layer.
class ResnetFeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super(ResnetFeatureExtractor, self).__init__()
        self.model = model
        self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])
                
    def __call__(self, x):
        return self.feature_extractor(x)[:, :, 0, 0]


def get_image_from_path(path):
    """A function that gets a URL of an image, 
    and returns a numpy image and a preprocessed
    torch tensor ready to pass to the model """

    # img = np.array(Image.open(path))
    # img = cv2.resize(img, (512, 512))
    img = cv2.imread(path, 1)
    img = cv2.resize(img, (112, 112))[:, :, ::-1]
    rgb_img_float = np.float32(img) / 255
    input_tensor = preprocess_image(rgb_img_float,
                                   mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    return img, rgb_img_float, input_tensor

if __name__ == '__main__':
    """ python cam.py --image-path <path_to_image>
    Example usage of loading an image and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = {
        "gradcam": GradCAM,
        "hirescam": HiResCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
        "gradcamelementwise": GradCAMElementWise
    }



    # resnet = models.resnet50(pretrained=True)

    resnet = get_model("r50", dropout=0.0, fp16=False, num_features=512)
    ckpt = torch.load("weights/backbone.pth")
    # print(ckpt)
    resnet.load_state_dict(ckpt)
    resnet = resnet.eval()
    # print(resnet)
    # model = ResnetFeatureExtractor(resnet)
    model = resnet
    # print(model)


    A_img, A_img_float, A_tensor = get_image_from_path(args.imageA_path)
    B_img, B_img_float, B_tensor = get_image_from_path(args.imageB_path)


    A_concept_features = model(A_tensor)[0, :]
    B_concept_features = model(B_tensor)[0, :]

    target_layers = [resnet.layer4[-1]]
    A_targets = [SimilarityToConceptTarget(A_concept_features)]
    B_targets = [SimilarityToConceptTarget(B_concept_features)]

    # Where is the car in the image

    # with GradCAM(model=model,
    #             target_layers=target_layers,
    #             use_cuda=False) as cam:
    with methods[args.method](model=model,
                target_layers=target_layers,
                use_cuda=False) as cam:
        grayscale_cam = cam(input_tensor=A_tensor,
                            targets=B_targets)[0, :]
        
    cam_image = show_cam_on_image(A_img_float, grayscale_cam, use_rgb=True)
    # Image.fromarray(cam_image)
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)


    os.makedirs(args.output_dir, exist_ok=True) 
    # cam_output_path = os.path.join(args.output_dir, f'{args.method}_cam.jpg')
    # cv2.imwrite(cam_output_path, cam_image)
    
    output_path = os.path.join(args.output_dir, f'{args.method}_all.jpg')
    output_img = np.concatenate([A_img[:, :, ::-1], B_img[:, :, ::-1], cam_image], axis=-2)
    cv2.imwrite(output_path, output_img)