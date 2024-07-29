import os
import torch
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import glob

import clip_net.clip


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip_net.clip.load("ViT-B/32", device=device)


def clip_feat_extract(img):

    image = preprocess(Image.open(img)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features


def ImageClIP_Patch_feat_extract(dir_fps_path, dst_clip_path):

    video_list = os.listdir(dir_fps_path)
    video_idx = 0
    total_nums = len(video_list)

    for video in video_list:

        video_idx = video_idx + 1
        print("\n--> ", video_idx, video)

        save_file = os.path.join(dst_clip_path, video + '.npy')
        if os.path.exists(save_file):
            print(video + '.npy', "is already processed!")
            continue

        video_img_list = sorted(glob.glob(os.path.join(dir_fps_path, video, '*.jpg')))
        
        params_frames = len(video_img_list)
        samples = np.round(np.linspace(0, params_frames-1, params_frames))

        img_list  = [video_img_list[int(sample)] for sample in samples]
        img_features = torch.zeros(len(img_list), patch_nums, C)

        idx = 0
        for img_cont in img_list:
            img_idx_feat = clip_feat_extract(img_cont)
            img_features[idx] = img_idx_feat
            idx += 1

        img_features = img_features.float().cpu().numpy()
        np.save(save_file, img_features)

        print("Process: ", video_idx, " / ", total_nums, " ----- video id: ", video_idx, " ----- save shape: ", img_features.shape)


def ImageClIP_feat_extract(dir_fps_path, dst_clip_path):

    video_list = os.listdir(dir_fps_path)
    video_idx = 0
    total_nums = len(video_list)

    for video in video_list:

        video_idx = video_idx + 1
        print("\n--> ", video_idx, video)

        save_file = os.path.join(dst_clip_path, video + '.npy')
        if os.path.exists(save_file):
            print(video + '.npy', "is already processed!")
            continue

        video_img_list = sorted(glob.glob(os.path.join(dir_fps_path, video, '*.jpg')))
        
        params_frames = len(video_img_list)
        samples = np.round(np.linspace(0, params_frames-1, params_frames))

        img_list  = [video_img_list[int(sample)] for sample in samples]
        img_features = torch.zeros(len(img_list), C)

        idx = 0
        for img_cont in img_list:
            img_idx_feat = clip_feat_extract(img_cont)
            img_features[idx] = img_idx_feat
            idx += 1

        img_features = img_features.float().cpu().numpy()
        np.save(save_file, img_features)

        print("Process: ", video_idx, " / ", total_nums, " ----- video id: ", video_idx, " ----- save shape: ", img_features.shape)

def qst_feat_extract(qst):

    text = clip.tokenize(qst).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
    return image_features

def QstCLIP_feat(json_path, dst_qst_path):

    samples = json.load(open(json_path, 'r'))
    
    ques_vocab = ['<pad>']
    # ans_vocab = []

    i = 0
    for sample in samples:
        i += 1
        question = sample['question_content'].rstrip().split(' ')
        question[-1] = question[-1][:-1]

        p = 0
        for pos in range(len(question)):
            if '<' in question[pos]:
                question[pos] = ast.literal_eval(sample['templ_values'])[p]
                p += 1
        for wd in question:
            if wd not in ques_vocab:
                ques_vocab.append(wd)

        print(question)


if __name__ == "__main__":

    patch_nums = 50
    C = 512


    # 158
    dir_fps_path = '/data/MUSIC-AVQA/avqa-frames-1fps'
    dst_clip_path = '/data/MUSIC-AVQA/clip_vit_b32'
    ImageClIP_feat_extract(dir_fps_path, dst_clip_path)


    