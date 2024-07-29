import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import ast
import json
from PIL import Image
from munch import munchify
import time
import random


def ids_to_multinomial(id, categories):
    """ label encoding
    Returns:
      1d array, multimonial representation, e.g. [1,0,1,0,0,...]
    """
    id_to_idx = {id: index for index, id in enumerate(categories)}

    return id_to_idx[id]


class AVQA_dataset(Dataset):

    def __init__(self, args, label, 
                       audios_feat_dir, visual_feat_dir, 
                       audios_patch_dir, visual_patch_dir,
                       qst_prompt_dir, qst_feat_dir,
                       transform=None, mode_flag='train'):

        self.args = args

        samples = json.load(open('../dataset/split_que_id/music_avqa_train.json', 'r'))

        # Question
        ques_vocab = ['<pad>']
        ans_vocab = []
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
            if sample['anser'] not in ans_vocab:
                ans_vocab.append(sample['anser'])

        self.ques_vocab = ques_vocab
        self.ans_vocab = ans_vocab
        self.word_to_ix = {word: i for i, word in enumerate(self.ques_vocab)}

        self.samples = json.load(open(label, 'r'))
        self.max_len = 14    # question length

        self.audios_feat_dir = audios_feat_dir
        self.visual_feat_dir = visual_feat_dir

        self.audios_patch_dir = audios_patch_dir
        self.visual_patch_dir = visual_patch_dir

        self.qst_prompt_dir = qst_prompt_dir
        self.qst_feat_dir = qst_feat_dir

        self.transform = transform


    def __len__(self):
        return len(self.samples)

    def get_lstm_embeddings(self, question_input, sample):

        question = sample['question_content'].rstrip().split(' ')
        question[-1] = question[-1][:-1]

        p = 0
        for pos in range(len(question)):
            if '<' in question[pos]:
                question[pos] = ast.literal_eval(sample['templ_values'])[p]
                p += 1
        if len(question) < self.max_len:
            n = self.max_len - len(question)
            for i in range(n):
                question.append('<pad>')

        idxs = [self.word_to_ix[w] for w in question]
        ques = torch.tensor(idxs, dtype=torch.long)

        return ques

    def get_frames_spatial(self, video_name):
        
        frames_path = os.path.join(self.frames_dir, video_name)
        frames_spatial = image_info(frames_path)    # [T, 3, 224, 224]

        return frames_spatial

    def __getitem__(self, idx):
        
        sample = self.samples[idx]
        name = sample['video_id']
        question_id = sample['question_id']

        audios_feat = np.load(os.path.join(self.audios_feat_dir, name + '.npy'))
        visual_feat = np.load(os.path.join(self.visual_feat_dir, name + '.npy'))

        question_feat = np.load(os.path.join(self.qst_feat_dir, str(question_id) + '.npy'))
        question_prompt = np.load(os.path.join(self.qst_prompt_dir, str(question_id) + '.npy'))

        audios_patch_feat = np.load(os.path.join(self.audios_patch_dir, name + '.npy'))
        visual_patch_feat = np.load(os.path.join(self.visual_patch_dir, name + '.npy'))

       
        ### answer
        answer = sample['anser']
        answer_label = ids_to_multinomial(answer, self.ans_vocab)
        answer_label = torch.from_numpy(np.array(answer_label)).long()


        sample = {'video_name': name,
                  'audios_feat': audios_feat, 
                  'visual_feat': visual_feat,
                  'audios_patch_feat': audios_patch_feat,
                  'visual_patch_feat': visual_patch_feat,
                  'question_feat': question_feat,
                  'question_prompt': question_prompt,
                  'answer_label': answer_label, 
                  'question_id': question_id}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):

    def __call__(self, sample):

        video_name = sample['video_name']
        audios_feat = sample['audios_feat']
        visual_feat = sample['visual_feat']
        audios_patch_feat = sample['audios_patch_feat']
        visual_patch_feat = sample['visual_patch_feat']
        question_feat = sample['question_feat']
        question_prompt = sample['question_prompt']
        answer_label = sample['answer_label']
        question_id = sample['question_id']

        return {'video_name': video_name, 
                'audios_feat': torch.from_numpy(audios_feat),
                'visual_feat': torch.from_numpy(visual_feat),
                'audios_patch_feat': torch.from_numpy(audios_patch_feat),
                'visual_patch_feat': torch.from_numpy(visual_patch_feat).to(torch.float32),
                'question_feat': sample['question_feat'],
                'question_prompt': sample['question_prompt'],
                'answer_label': answer_label,
                'question_id':question_id}