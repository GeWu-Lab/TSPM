import torch

from extract_clip_feat.clip_net import clip
# import clip_net.clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device=device)


def qst_feat_extract(qst):

    text = clip.tokenize(qst).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text)
    
    text_features=text_features.float()

    return text_features


