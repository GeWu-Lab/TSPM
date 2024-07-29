import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math

from configs.arguments import parser


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class AVHanLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(AVHanLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cm_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src_q, src_v, src_mask=None, src_key_padding_mask=None):

        src_q = src_q.permute(1, 0, 2)
        src_v = src_v.permute(1, 0, 2)
        src1 = self.cm_attn(src_q, src_v, src_v, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src2 = self.self_attn(src_q, src_q, src_q, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src_q = src_q + self.dropout11(src1) + self.dropout12(src2)
        src_q = self.norm1(src_q)

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src_q))))
        src_q = src_q + self.dropout2(src2)
        src_q = self.norm2(src_q)
        return src_q.permute(1, 0, 2)


class AV_Attn(nn.Module):

    def __init__(self, args, encoder_layer, num_layers, norm=None):
        super(AV_Attn, self).__init__()

        self.args = args

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self.norm = norm

    def forward(self, src_a, src_v, mask=None, src_key_padding_mask=None):
        
        audio_output = src_a
        visual_output = src_v

        for i in range(self.num_layers):
            audio_output = self.layers[i](src_a, src_v, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            visual_output = self.layers[i](src_v, src_a, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm:
            audio_output = self.norm1(audio_output)
            visual_output = self.norm2(visual_output)

        return audio_output, visual_output


class TemporalPerception(nn.Module):

    def __init__(self, args):
        super(TemporalPerception, self).__init__()

        self.args = args

        # question as query on audio-visual clip
        self.attn_qst_query = nn.MultiheadAttention(512, 4, dropout=0.1)
        self.qst_query_linear1 = nn.Linear(512, 512)
        self.qst_query_relu = nn.ReLU()
        self.qst_query_dropout1 = nn.Dropout(0.1)
        self.qst_query_linear2 = nn.Linear(512, 512)
        self.qst_query_dropout2 = nn.Dropout(0.1)
        self.qst_query_visual_norm = nn.LayerNorm(512)

    def QstQueryClipAttn(self, query_feat, kv_feat):

        kv_feat = kv_feat.permute(1, 0, 2)

        query_feat = query_feat.unsqueeze(0)
        attn_feat, temp_weights = self.attn_qst_query(query_feat, kv_feat, kv_feat, 
                                                      attn_mask=None, key_padding_mask=None)
        attn_feat = attn_feat.squeeze(0)
        src = self.qst_query_linear1(attn_feat)
        src = self.qst_query_relu(src)
        src = self.qst_query_dropout1(src)
        src = self.qst_query_linear2(src)
        src = self.qst_query_dropout2(src)

        attn = attn_feat + src
        attn = self.qst_query_visual_norm(attn)

        return attn, temp_weights

    def SelectTopK(self, temp_weights, audio_input, visual_input, B, C):
        '''
            Top-k temporal segments selection
        '''

        # return temporal indices
        sort_index = torch.argsort(temp_weights, dim=-1)        # [B, 1, T]
        top_k_index = sort_index[:, :, -self.args.top_k:]       # [B, 1, Top_K]

        top_k_index_sort, indices = torch.sort(top_k_index)     # [B, 1, Top_K]
        top_k_index_sort = top_k_index_sort.cpu().numpy()       # [B, 1, Top_K],

        output_audio = torch.zeros(B, self.args.top_k, C).cuda()
        output_visual = torch.zeros(B, self.args.top_k, C).cuda()

        for batch_idx in range(B):
            idx = 0
            for temp_idx in top_k_index_sort.tolist()[batch_idx][0]:
                output_audio[batch_idx, idx, :] = audio_input[batch_idx, temp_idx, :]
                output_visual[batch_idx, idx, :] = visual_input[batch_idx, temp_idx, :]
                idx = idx + 1

        return output_audio, output_visual, top_k_index_sort


    def forward(self, audio_input, visual_input, qst_input):

        B, T, C = audio_input.size()
        temp_clip_attn_feat, temp_weights = self.QstQueryClipAttn(qst_input, visual_input)
        output_audio, output_visual, top_k_index_sort = self.SelectTopK(temp_weights, audio_input, visual_input, B, C)

        return output_audio, output_visual, top_k_index_sort


class QstTemporalGrounding(nn.Module):
    '''
        Question as query on audio and visual features.
    '''
    def __init__(self, args):
        super(QstTemporalGrounding, self).__init__()

        # question as query on audio and visual_feat_grd
        self.attn_qst_query = nn.MultiheadAttention(512, 4, dropout=0.1)
        self.qst_query_linear1 = nn.Linear(512, 512)
        self.qst_query_relu = nn.ReLU()
        self.qst_query_dropout1 = nn.Dropout(0.1)
        self.qst_query_linear2 = nn.Linear(512, 512)
        self.qst_query_dropout2 = nn.Dropout(0.1)
        self.qst_query_visual_norm = nn.LayerNorm(512)

    def QstQueryAttn(self, query_feat, key_feat, value_feat):
        
        key_feat = key_feat.permute(1, 0, 2)
        value_feat = value_feat.permute(1, 0, 2)
        query_feat = query_feat.unsqueeze(0)
        attn_feat, temp_weights = self.attn_qst_query(query_feat, key_feat, value_feat, 
                                                      attn_mask=None, key_padding_mask=None)
        attn_feat = attn_feat.squeeze(0)
        src = self.qst_query_linear1(attn_feat)
        src = self.qst_query_relu(src)
        src = self.qst_query_dropout1(src)
        src = self.qst_query_linear2(src)
        src = self.qst_query_dropout2(src)

        attn = attn_feat + src
        attn = self.qst_query_visual_norm(attn)

        return attn

    def forward(self, qst_input, audio_input, visual_input):
        
        audio_feat = self.QstQueryAttn(qst_input, audio_input, audio_input)
        visual_feat = self.QstQueryAttn(qst_input, visual_input, visual_input)

        return audio_feat, visual_feat


class TokensSelfAttn(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(TokensSelfAttn, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src_q, src_mask=None, src_key_padding_mask=None):
        # print("src_q: ", src_q.shape)


        src_q = src_q.permute(1, 0, 2)

        src2 = self.self_attn(src_q, src_q, src_q, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0]
        src_q = src_q + self.dropout12(src2)
        src_q = self.norm1(src_q)

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src_q))))
        src_q = src_q + self.dropout2(src2)
        src_q = self.norm2(src_q)

        return src_q.permute(1, 0, 2)


class SpatioPerceptionModule(nn.Module):

    def __init__(self, args, hidden_size=512):
        super(SpatioPerceptionModule, self).__init__()

        self.args = args
        self.attn_qst_query = nn.MultiheadAttention(512, 4, dropout=0.1)

        self.qst_query_linear1 = nn.Linear(512, 512)
        self.qst_query_relu = nn.ReLU()
        self.qst_query_dropout1 = nn.Dropout(0.1)
        self.qst_query_linear2 = nn.Linear(512, 512)
        self.qst_query_dropout2 = nn.Dropout(0.1)
        self.qst_query_visual_norm = nn.LayerNorm(512)

        self.TokensAttn = TokensSelfAttn(d_model=512, nhead=1)

    def TopKSegs(self, visual_patch, top_k_index_sort):
        
        # patch feat: [B, T, N, C]
        B, T, N, C = visual_patch.size()
        visual_patch_select = torch.zeros(B, self.args.top_k, N, C).cuda()
        
        for batch_idx in range(B):
            temp_idx = 0
            for k_idx in range(self.args.top_k):
                visual_patch_select[batch_idx, temp_idx, :, :] = visual_patch[batch_idx, k_idx, :, :]
                temp_idx = temp_idx + 1

        return avisual_patch_select


    def AudioGuidedPatchAttn(self, query_feat, visual_patch):
        '''
            query feat:   [B, Top_k, C] [B, 15, 512]
            visual_patch: [B, Top_k, N, C] [B, 15, 14, 512]
        '''

        # input visual: [B, T, N, C]
        B, T, N, C = visual_patch.size()

        visual_patch_feat = visual_patch.view(B*T, N, C)
        audio_query_feat = query_feat.view(B*T, 1, C)

        visual_patch_feat = self.TokensAttn(visual_patch_feat)

        kv_feat = visual_patch_feat.permute(1, 0, 2)
        q_feat = audio_query_feat.permute(1, 0, 2)

        attn_feat, visual_patch_weights = self.attn_qst_query(q_feat, kv_feat, kv_feat, 
                                                          attn_mask=None, key_padding_mask=None)
        src = self.qst_query_linear1(attn_feat)
        src = self.qst_query_relu(src)
        src = self.qst_query_dropout1(src)
        src = self.qst_query_linear2(src)
        src = self.qst_query_dropout2(src)

        attn = attn_feat + src
        attn = self.qst_query_visual_norm(attn)

        visual_patch_out = attn.view(B, T, C)

        return visual_patch_out
        

    def forward(self, audios_feat, visual_patch, qst_feat_prompt, top_k_index_sort):

        # output: [B, top-k frames, N, C]
        visual_patch_top_k = self.TopKSegs(visual_patch, top_k_index_sort)

        # audio guided visual perception
        visual_patch_out = self.AudioGuidedPatchAttn(audios_feat, visual_patch_top_k)

        return visual_patch_out


class TSPM(nn.Module):

    def __init__(self, args, hidden_size=512):
        super(TSPM, self).__init__()

        self.args = args
    
        # features input
        self.input_a = nn.Linear(128, hidden_size)
        self.input_v = nn.Linear(768, hidden_size)
        self.input_v_patch = nn.Linear(1024, hidden_size)

        self.input_qst = nn.Linear(768, hidden_size)
        self.input_qst_prompt = nn.Linear(768, hidden_size)

        # Modules
        self.AV_Attn = AV_Attn(args, AVHanLayer(d_model=512, nhead=1, dim_feedforward=512), num_layers=1) 
        self.TemporalPerception = TemporalPerceptionModule(args)
        self.SpatioPerception = SpatioPerceptionModule(args)
        self.QstTempGrd_Module = QstTemporalGrounding(args)

        # fusion
        self.av_fusion_tanh = nn.Tanh()
        self.av_relu = nn.ReLU()
        self.av_fusion_fc = nn.Linear(3072, 512)
        self.avq_fusion_tanh = nn.Tanh()

        # answer prediction
        self.answer_pred_fc = nn.Linear(512, 42)


    def forward(self, audio, visual, audios_patch, visual_patch, question, question_prompt):
        '''
            audios input: [B, T, C] [32, 60, 512]
            visual input: [B, T, C] [32, 60, 768]
            visual patch: [B, T, N, C] [32, 60, 14, 1024]
            question: [B, 1, C] [32, 1, 1024] 
        '''

        ### 1. features input *************************************************************************
        
        audio_feat = self.input_a(audio)                                          # [B, T, C]    
        visual_feat = self.input_v(visual)                                        # [B, T, C]
        visual_patch = self.input_v_patch(visual_patch)

        qst_feat = self.input_qst(question).squeeze(-2)                           # [B, C]
        qst_feat_prompt = self.input_qst_prompt(question_prompt).squeeze(-2)      # [B, C]


        ### 2. Audio-Visual Perception w/ Self-modal Attention + Cross-modal Attention
        if self.args.AV_Attn_Module:
            audio_feat_AVattn, visual_feat_AVattn = self.AV_Attn(audio_feat, visual_feat)


        ### 3. Temporal segments selection
        if self.args.Temp_Selection:
            audio_feat_tssm, visual_feat_tssm, top_k_index_sort = self.TemporalPerception(audio_feat, visual_feat, qst_feat_prompt)

        ### 4. Spatial regions perception
        if self.args.Spatio_Perception:
            visual_feat_sp = self.SpatioPerception(audio_feat_tssm, visual_patch, qst_feat_prompt, top_k_index_sort)

        ### 5. Question as query, temporal grounding
        if self.args.Temp_QTGM:
            audio_feat_qtgm, visual_feat_qtgm = self.QstTempGrd_Module(qst_feat, audio_feat_tssm, visual_feat_sp)



        ### 6. Fusion **************************************************************************
        av_feat = torch.cat([audio_feat_qtgm, audio_feat_AVattn.mean(dim=-2), audio_feat_tssm.mean(dim=-2), 
                             visual_feat_qtgm, visual_feat_AVattn.mean(dim=-2), visual_feat_sp.mean(dim=-2)], dim=-1)

        av_feat = self.av_fusion_tanh(av_feat)
        av_feat = self.av_fusion_fc(av_feat)

        avq_feat = torch.mul(av_feat, qst_feat)          # [batch_size, embed_size]
        avq_feat = self.avq_fusion_tanh(avq_feat)


        ### 7. Answer prediction moudule *************************************************************
        answer_pred = self.answer_pred_fc(avq_feat)  # [batch_size, ans_vocab_size=42]

        return answer_pred
