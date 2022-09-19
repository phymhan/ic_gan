import sys
import os
import torch

# sys.path[0] = '/content/ic_gan/inference'
sys.path.insert(1, os.path.join(sys.path[0], "."))

import inference.utils as inference_utils
import data_utils.utils as data_utils
from torchvision import datasets, transforms
import pickle
from scipy.stats import truncnorm, dirichlet
import numpy as np
import torchvision.utils as tvu
import pdb
st = pdb.set_trace

norm_mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
norm_std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# def preprocess_input_image(input_image_path, size): 
#   pil_image = Image_PIL.open(input_image_path).convert('RGB')
#   transform_list =  transforms.Compose([data_utils.CenterCropLongEdge(), transforms.Resize((size,size)), transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])
#   tensor_image = transform_list(pil_image)
#   tensor_image = torch.nn.functional.interpolate(tensor_image.unsqueeze(0), 224, mode="bicubic", align_corners=True)
#   return tensor_image

# def preprocess_generated_image(image): 
#     transform_list = transforms.Normalize(norm_mean, norm_std)
#     image = transform_list(image*0.5 + 0.5)
#     image = torch.nn.functional.interpolate(image, 224, mode="bicubic", align_corners=True)
#     return image

def load_icgan(experiment_name, root_ = './'):
    root = os.path.join(root_, experiment_name)
    config = torch.load("%s/%s.pth" %
                        (root, "state_dict_best0"))['config']
    config["base_root"] = ""
    config["samples_root"] = ""
    config["weights_root"] = root_
    config["model_backbone"] = 'biggan'
    config["experiment_name"] = experiment_name
    G, config = inference_utils.load_model_inference(config)
    G.cuda()
    G.eval()
    return G

def load_feature_extractor():
    feat_ext_path = 'download/swav_800ep_pretrain.pth.tar'
    feature_extractor = data_utils.load_pretrained_feature_extractor(feat_ext_path,
        feature_extractor='selfsupervised')
    feature_extractor.eval()
    return feature_extractor

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--truncation', type=float, default=0.7)
    parser.add_argument('--feature_noise_std', type=float, default=0.0)
    parser.add_argument('--n_samples', type=int, default=4)
    parser.add_argument('--zero', action='store_true')
    args = parser.parse_args()

    image_size = 256
    noise_size = 128
    truncation = args.truncation
    seed = 42
    n_samples = args.n_samples

    # load generator
    G = load_icgan('icgan_biggan_imagenet_res256', 'download')

    # load feature extractor
    feature_extractor = load_feature_extractor()

    # load images
    with open('../../active/essl/imagenet/I256_256.pkl', 'rb') as f:
        imgs = pickle.load(f)  # value range (-1, 1)
    imgs = (imgs + 1) / 2  # value range (0, 1)
    imgs = imgs[:n_samples]
    transform = transforms.Compose([
        # transforms.CenterCrop(image_size),
        # transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Normalize(norm_mean, norm_std)
    ])
    input_image_tensor = transform(imgs.cuda())

    # extract features
    with torch.no_grad():
        input_features, _ = feature_extractor(input_image_tensor)
    input_features += torch.randn_like(input_features) * args.feature_noise_std
    input_features /= torch.linalg.norm(input_features, dim=-1, keepdims=True)

    state = np.random.RandomState(seed)
    noise_vector = truncnorm.rvs(-2*truncation, 2*truncation, size=(len(imgs), noise_size), random_state=state).astype(np.float32) #see https://github.com/tensorflow/hub/issues/214
    noise_vector = torch.tensor(noise_vector, requires_grad=False, device='cuda')
    if args.zero:
        noise_vector.zero_()

    # generate
    # NOTE: stochastic truncation
    out = G(noise_vector, None, input_features.cuda())
    out = out.detach().cpu()
    # out_ = preprocess_generated_image(out)
    # NOTE: generated image in value range (-1, 1)
    out = (out + 1) / 2

    suffix = 'zero' if args.zero else ''

    xx = torch.cat([imgs, out], dim=0)
    tvu.save_image(xx, f'test_n={n_samples}_icgan_trunc={truncation}_noise={args.feature_noise_std}_{suffix}.png', value_range=(0, 1), normalize=True, nrow=n_samples)
