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
from tqdm import tqdm

import PIL

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

def convert_to_images(obj):
    """ Convert an output tensor from BigGAN in a list of images.
    """
    # need to fix import, see: https://github.com/huggingface/pytorch-pretrained-BigGAN/pull/14/commits/68a7446951f0b9400ebc7baf466ccc48cdf1b14c
    if not isinstance(obj, np.ndarray):
        obj = obj.detach().numpy()
    obj = obj.transpose((0, 2, 3, 1))
    obj = np.clip(((obj + 1) / 2.0) * 256, 0, 255)
    img = []
    for i, out in enumerate(obj):
        out_array = np.asarray(np.uint8(out), dtype=np.uint8)
        img.append(PIL.Image.fromarray(out_array))
    return img

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
    parser.add_argument('--image_folder', required=True, type=str, help='path to ImageNet train folder')
    parser.add_argument('--save_root', required=True, type=str, help='path to ImageNet train folder')
    parser.add_argument('--truncation', type=float, default=0.3)
    parser.add_argument('--feature_noise_std', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--zero', action='store_true')
    parser.add_argument('--n_part', type=int, default=None)
    parser.add_argument('--part', type=int, default=None)
    parser.add_argument('--image_size', type=int, default=256)
    args = parser.parse_args()

    image_size = args.image_size
    noise_size = 128
    truncation = args.truncation
    seed = 42
    batch_size = args.batch_size

    state = np.random.RandomState(seed)

    # load generator
    G = load_icgan('icgan_biggan_imagenet_res256', 'download')

    # load feature extractor
    feature_extractor = load_feature_extractor()

    # load images
    transform = transforms.Compose([
        transforms.CenterCrop(image_size),  # NOTE: change to rand crop?
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Normalize(norm_mean, norm_std)
    ])
    dataset = datasets.ImageFolder(root=args.image_folder, transform=transform)
    
    os.makedirs(args.save_root, exist_ok=True)

    index = np.arange(len(dataset))
    if args.part is not None:
        part_size = int(np.ceil(len(index) / args.n_part))
        index = index[args.part * part_size:(args.part + 1) * part_size]
    for i in tqdm(range(int(np.ceil(len(index) / batch_size)))):
        inds = index[i * batch_size:(i + 1) * batch_size]
        actual_batch_size = len(inds)
        imgs = []
        paths = []
        for j in inds:
            path, _ = dataset.imgs[j]
            img, _ = dataset[j]
            imgs.append(img)
            paths.append(path)

        imgs = torch.stack(imgs)
        with torch.no_grad():
            # input_image_tensor = transform(imgs.cuda())
            input_image_tensor = imgs.cuda()
            input_features, _ = feature_extractor(input_image_tensor)

        input_features += torch.randn_like(input_features) * args.feature_noise_std
        input_features /= torch.linalg.norm(input_features, dim=-1, keepdims=True)

        noise_vector = truncnorm.rvs(-2*truncation, 2*truncation, size=(actual_batch_size, noise_size), random_state=state).astype(np.float32)
        noise_vector = torch.tensor(noise_vector, requires_grad=False, device='cuda')

        with torch.no_grad():
            out = G(noise_vector, None, input_features.cuda())
            out = out.detach().cpu()
        
        ims = convert_to_images(out)

        for j, im in enumerate(ims):
            classname, imagename = paths[j].split('/')[-2:]
            os.makedirs(os.path.join(args.save_root, classname), exist_ok=True)
            im.save(os.path.join(
                args.save_root, classname, imagename))
