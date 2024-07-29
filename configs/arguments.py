import os
import argparse

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Implementation of Audio-Visual Question Answering')


### ======================== Dataset Configs ==========================

server_name = '137'

if server_name == '137':
    server_path = "/home/data/MUSIC-AVQA/"
else:
    print("Please select one server!")

parser.add_argument("--audios_feat_dir", type=str, default=os.path.join(server_path, 'vggish'), 
                    help="audio feat dir")
parser.add_argument("--visual_feat_dir", type=str, default=os.path.join(server_path, 'clip_feats/frame_1fps_ViT-L14@336px'), 
                    help="visual feat dir")
parser.add_argument("--audios_patch_dir", type=str, default=os.path.join(server_path, 'ToMe_feat/audio_tome14'), 
                    help="audio patch dir")
parser.add_argument("--visual_patch_dir", type=str, default=os.path.join(server_path, 'ToMe_feat/tome14'), 
                    help="visual patch dir")
parser.add_argument("--qst_feat_dir", type=str, default=os.path.join(server_path, 'clip_feats/qst_ViT-L14@336px'), 
                    help="question features")
parser.add_argument("--qst_prompt_dir", type=str, default=os.path.join(server_path, 'clip_feats/qaPrompt_ViT-L14@336px'), 
                    help="question answers prompt construction")


### ======================== Label Configs ==========================
parser.add_argument("--label_train", type=str, default="../dataset/split_que_id/music_avqa_train.json", 
                    help="train csv file")
parser.add_argument("--label_val", type=str, default="../dataset/split_que_id/music_avqa_val.json", 
                    help="val csv file")
parser.add_argument("--label_test", type=str, default="../dataset/split_que_id/music_avqa_test.json", 
                    help="test csv file")

### ======================== Model Configs ==========================
parser.add_argument("--Temp_Selection", type=bool, default=True, metavar='tssm',
                    help="temporal segments selection module")
parser.add_argument("--top_k", type=int, default=10, metavar='TK',
                    help="top K temporal segments")
parser.add_argument("--Spatio_Perception", type=bool, default=True, metavar='sp',
                    help="Spatio_Perception")
parser.add_argument("--AV_Attn_Module", type=bool, default=True, metavar='av_attn',
                    help="Audio-Visual Perception w/ Self-modal Attention + Cross-modal Attention")
parser.add_argument("--Temp_QTGM", type=bool, default=True, metavar='qtgm',
                    help="question as query, temporal grounding module")


### ======================== Learning Configs ==========================
parser.add_argument('--batch-size', type=int, default=64, metavar='N', 
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=30, metavar='E', 
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', 
                    help='learning rate (default: 3e-4)')
parser.add_argument('--seed', type=int, default=1, metavar='S', 
                    help='random seed (default: 1)')


### ======================== Save Configs ==========================
parser.add_argument( "--checkpoint", type=str, default='TSPM', 
                    help="save model name")
parser.add_argument("--model_save_dir", type=str, default='models/', 
                    help="model save dir")
parser.add_argument("--mode", type=str, default='train', 
                    help="with mode to use")
parser.add_argument("--result_dir", type=str, default='results/', 
                    help="results files")


### ======================== Runtime Configs ==========================
parser.add_argument('--log-interval', type=int, default=50, metavar='N', 
                    help='how many batches to wait before logging training status')
parser.add_argument('--num_workers', type=int, default=14, 
                    help='num_workers number')
parser.add_argument('--gpu', type=str, default='1, 2', 
                    help='gpu device number')
