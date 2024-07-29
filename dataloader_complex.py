import json
import numpy as np
import os
import random
from os.path import join
import torch
from torch.utils.data import Dataset,DataLoader


ans_vab=['two', 'cello', 'congas', 'zero', 'no', 'pipa', 'six', 'yes', 'one', 'four', 'three', 'seven', 
    'five', 'ukulele', 'right', 'piano', 'left', 'accordion', 'clarinet', 'guzheng', 'more than ten', 'nine', 
    'indoor', 'saxophone', 'drum', 'violin', 'middle', 'outdoor', 'bagpipe', 'bassoon', 'acoustic_guitar', 
    'banjo', 'electric_bass', 'ten', 'eight', 'flute', 'simultaneously', 'trumpet', 'erhu', 'xylophone', 'tuba', 'suona']

class ComplexDataset(Dataset):

    def __init__(self,
                samples,
                audio_feat_dir,
                video_feat_dir,
                audio_patch_dir,
                video_patch_dir,
                question_feat_dir,
                question_prompt_feat_dir,
                *args,
                **kwards
                ) -> None:
        super().__init__()

        self.audio_feat_dir=audio_feat_dir
        self.video_feat_dir=video_feat_dir
        self.audio_patch_dir=audio_patch_dir
        self.video_patch_dir=video_patch_dir
        self.question_feat_dir=question_feat_dir
        self.question_prompt_feat_dir=question_prompt_feat_dir

        self.samples=samples
        
        self.ans2idx={ans:idx for idx,ans in enumerate(ans_vab)}
        self.idx2ans={idx:ans for idx,ans in enumerate(ans_vab)}

    
    def __len__(self):
        return len(self.samples)


    def __getitem__(self,idx):
        sample=self.samples[idx]

        vid=sample['vid']
        qid=sample['qid']
        content=sample['content']
        answer=sample['answer']
        answer=self.ans2idx[answer]

        
        audio_feat=np.load(join(self.audio_feat_dir,vid+'.npy'))
        video_feat=np.load(join(self.video_feat_dir,vid+'.npy'))
        audio_patch_feat=np.load(join(self.audio_patch_dir,vid+'.npy.npy'))
        video_patch_feat=np.load(join(self.video_patch_dir,vid+'.npy'))
        question_feat=np.load(join(self.question_feat_dir,qid+'.npy'))
        question_prompt_feat=np.load(join(self.question_prompt_feat_dir,qid+'.npy'))


        return {
            'audio_feat':torch.from_numpy(audio_feat),
            'video_feat':torch.from_numpy(video_feat),
            'audio_patch_feat':torch.from_numpy(audio_patch_feat),
            'video_patch_feat':torch.from_numpy(video_patch_feat),
            'question_feat':torch.from_numpy(question_feat),
            'question_prompt_feat':torch.from_numpy(question_prompt_feat),
            'answer':torch.tensor(answer).long(),
        }


def get_dataloader(samples,kwards:dict,bs,shuffle,num_workers,drop_last,pin_memory=True):

    dataset=ComplexDataset(samples=samples,**kwards)

    dataloader=DataLoader(
        dataset=dataset,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory
    )

    return dataset,dataloader

