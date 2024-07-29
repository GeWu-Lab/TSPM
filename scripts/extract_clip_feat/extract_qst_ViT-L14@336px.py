import os
import torch
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import glob
import json
import ast
import csv

import clip_net.clip

device = "cuda:2" if torch.cuda.is_available() else "cpu"
model, preprocess = clip_net.clip.load("ViT-L/14@336px", device=device)


def qst_feat_extract(qst):

    text = clip_net.clip.tokenize(qst).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text)
    
    return text_features


def QstCLIP_feat(json_path, dst_qst_path):

    samples = json.load(open(json_path, 'r'))
    
    ques_vocab = ['<pad>']
    # ans_vocab = []

    i = 0
    for sample in samples:
        i += 1
        question = sample['question_content'].rstrip().split(' ')
        question[-1] = question[-1][:-1]

        question_id = sample['question_id']
        print("\n")
        print("question id: ", question_id)

        save_file = os.path.join(dst_qst_path, str(question_id) + '.npy')

        if os.path.exists(save_file):
            print(question_id, " is already exist!")
            continue

        p = 0
        for pos in range(len(question)):
            if '<' in question[pos]:
                question[pos] = sample['templ_values'][p]
                p += 1
        for wd in question:
            if wd not in ques_vocab:
                ques_vocab.append(wd)

        
        # print(len(question))
        question = ' '.join(question) + '?'
        # question = question[:]
        print(question)
        
        
        qst_feat = qst_feat_extract(question)
        print(qst_feat.shape)

        

        qst_features = qst_feat.float().cpu().numpy()

        np.save(save_file, qst_features)


def VideoLableCLIP_feat(label_path, dst_qst_path):

    with open(label_path, 'r', encoding='utf-8') as lp:
        for line in lp:
            line = line.replace("\n", "")
            line_info = line.split(",")
            video_name = line_info[0]
            video_label = line_info[1:]
            video_label = ' '.join(video_label)
            # print(video_name, video_label, type(video_label))

            label_feat = qst_feat_extract(video_label)
            print(video_name, video_label, label_feat.shape)

            save_file = os.path.join(dst_qst_path, str(video_name) + '.npy')
            label_features = label_feat.float().cpu().numpy()

            np.save(save_file, label_features)


if __name__ == "__main__":

    json_path = "/data/AVQA/data/sub-qst-test.json"
    
    dst_qst_path = "/data/AVQA/LLM-AVQA/sub-qst-feat"


    QstCLIP_feat(json_path, dst_qst_path)


    