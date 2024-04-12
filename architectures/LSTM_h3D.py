import logging
import numpy as np
import torch
import torch.nn as nn
import random
from .decode_beam_2 import beam_decode

n_joint  = 22

class Encoder(nn.Module):
    def __init__(self,hidden_size,dropout=None,hidden_dim=128,device=None):
        super().__init__()
        # input_dim ex: vocab_size
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        # HUMAN ML3D GRAPH
        [root, torso, rarm, larm, rleg, lleg] = [0], [0, 3, 6, 9, 12, 15], [14, 17, 19, 21], [ 13, 16, 18, 20], [2, 5, 8, 11], [1, 4, 7, 10]
        self.body_parts = [root, torso, rarm, larm, rleg, lleg]
        njoint_per_part = [len(p) for p in self.body_parts]
        self.six_layers = nn.ModuleList([nn.Linear(inp_dim*3,hidden_dim,device=device) for inp_dim in njoint_per_part])
        self.six_layers_2 = nn.ModuleList([nn.Linear(hidden_dim,hidden_size//2,device=device) for _ in njoint_per_part])
        self.six_layers_v = nn.ModuleList([nn.Linear(inp_dim*3,hidden_dim,device=device) for inp_dim in njoint_per_part])
        self.six_layers_v2 = nn.ModuleList([nn.Linear(hidden_dim,hidden_size//2,device=device) for _ in njoint_per_part])

    def forward(self, x):
        T,N,CV = x.size()
        x = x.reshape(T,N,n_joint,3)

        skel_parts = [x[:, :, prt, :].reshape(T,N, len(prt) * 3) for prt in self.body_parts]

        velocity = torch.cat([(x[0] - x[0]).unsqueeze(0),  x[1:]-x[:-1]], dim=0)
        skel_parts_v = [velocity[:, :, prt, :].reshape(T,N, len(prt) * 3) for prt in self.body_parts]

        # Pose features "P_ij"
        outputs = torch.zeros(x.size()[:-2]+(self.hidden_size,6))
        i = 0
        for prt,prt_v in zip(skel_parts,skel_parts_v):
            out = torch.tanh(self.six_layers[i](prt))
            out = torch.tanh(self.six_layers_2[i](out))
            out_v = torch.tanh(self.six_layers_v[i](prt_v))
            out_v = torch.tanh(self.six_layers_v2[i](out_v))
            outputs[:,:,:,i] = torch.cat([out,out_v],dim=-1)
            i += 1
        return outputs

class Decoder(nn.Module):
    def __init__(self, input_dim,embedding_dim, hidden_size, batch_size=32, attention="bahadanau", num_layers=1,
                 device=torch.device("cpu"),dropout=None,beam_search=False):
        super().__init__()

        self.device = device
        self.beam_search = beam_search
        self.batch_size = batch_size
        self.dropout = nn.Dropout(dropout)
        self.dec_hidden_size =  num_layers * hidden_size
        self.output_dim = input_dim  # the input_dim is the vocabulary size which also is the output_dim probability
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=None)

        #----------- TOP and BOTTOM LSTM

        self.top_lstm    = nn.LSTM(input_size=hidden_size,num_layers=num_layers,
                           hidden_size=self.dec_hidden_size , bidirectional=False)
        self.bottom_lstm = nn.LSTM(input_size=embedding_dim, num_layers=num_layers,
                           hidden_size=self.dec_hidden_size , bidirectional=False)


        self.attention_type = attention
        logging.info(f"Applying {self.attention_type} attention")

    def forward(self, x):
        # x shape [1,batch_size]
        x = self.embedding(x)
        x = self.dropout(x)
        return  x



class seq2seq(nn.Module):
    def __init__(self, input_dim, hidden_size, embedding_dim, num_layers=1,device=torch.device('cpu'),
                 dropout=0,beam_size=1,attention="bahadanau",encoder_type="MLP",hidden_dim=128,K=n_joint):
        super(seq2seq, self).__init__()

        self.device = device
        self.output_dim = input_dim # vocab_size
        self.hidden_size = hidden_size
        self.encoder_type=encoder_type
        self.beam_size = beam_size # default beam_size =  1 --> Greedy search

        self.enc_pose = Encoder(hidden_size,dropout=dropout,hidden_dim=hidden_dim,device=device)
        self.dec = Decoder(input_dim, embedding_dim=embedding_dim,hidden_size=hidden_size, num_layers=num_layers, device=device,
                           dropout=dropout,attention=attention,beam_search=True if self.beam_size>1 else False)

        hi_dim = hidden_dim+embedding_dim+hidden_size

        self.mixture_feat = nn.Linear(hi_dim,hi_dim)

        # --- Final Layer
        self.final_layer = nn.Linear(hi_dim,self.output_dim)
        self.attention_type = attention

        self.K = K
        self.feat_extract_x = torch.nn.Linear(in_features=self.hidden_size*self.K,out_features=hidden_dim)
        self.feat_extract_hdec = torch.nn.Linear(in_features=hidden_size,out_features=hidden_dim)
        self.feat_extract_henc = torch.nn.Linear(in_features=hidden_size,out_features=hidden_dim)
        self.spatial_att  = torch.nn.Linear(in_features=hidden_dim,out_features=self.K) # Us
        self.num_layers = num_layers
        self.gate_var = nn.Linear(hidden_size,1)
        self.adapt_layer  = nn.Linear(embedding_dim, 1)
        self.ctproject = nn.Linear(hidden_size,hidden_dim)
        self.htproject = nn.Linear(hidden_size,hidden_dim)

        self.feat_extract_g = nn.Linear(in_features=self.K*self.hidden_size,out_features=hidden_dim)
        self.tempfeat_extract_hdec =  nn.Linear(in_features=hidden_size,out_features=hidden_dim)
        self.temporal_att = nn.Linear(in_features=hidden_dim,out_features=1)

        self.hidden_dim = hidden_dim
    def forward(self, x, y, teacher_force_ratio=0,src_lens=None):

        self.src_lens = src_lens # for packed sequence
        enc_masks = torch.zeros(x.shape[:2],device=self.device) # (seq_len,batch_size)
        for i,l in enumerate(src_lens):
            enc_masks[:l,i]=1

        #-------------- FEATURE EXTRACTION [R] // ENCODER
        # x shape : (T,N,CV)
        #concatenate velocity and motion
        xparts = self.enc_pose(x).to(self.device)
        T,N,CV = x.size()

        #  ------------ Frame wise pose features : R # shape  : (T,N,V,C)

        R = xparts.reshape(T,N,6*self.hidden_size).to(self.device)

        #------------ Decoder Initialization

             # ----------- LSTM Top (initial memory and hidden state)
        top_ht_1 = torch.zeros((1,x.size(1), self.hidden_size),device=self.device)
        top_mt_1 = torch.zeros((1,x.size(1), self.hidden_size),device=self.device)

            #------------ LSTM Bottom (initial memory and hidden state)

        bot_ht_1  =  top_ht_1
        bot_mt_1  = top_mt_1

        dec_pred_output = []
        # first tokens : <sos> index : 1
        dec_pred_output.append(torch.ones((1, y.size(1)), dtype=torch.int, device=self.device))
        trg_len = y.size(0)
        output_list = []
        self.attention_weights_list = []

        self.spatial_attentions = torch.zeros((trg_len-1,x.size(0),x.size(1),self.K),device=self.device)
        self.temporal_attentions = torch.zeros((trg_len-1,x.size(0),x.size(1)),device=self.device)

        self.beta = torch.zeros((trg_len - 1,x.size(1)), device=self.device)


        if self.beam_size==1:
            # TODO USE FULL TEACHER FORCING TO SPEED UP TRAINING AND GET OUT THIS LOOP !
            for j in range(trg_len-1):
                # ------- CLIP COIN
                thr = random.random()
                y_s = y[j].unsqueeze(0) if thr < teacher_force_ratio else dec_pred_output[j]

                # ------------ TOP and Bottom LSTM prediction
                _,(bot_ht_1,bot_mt_1) = self.dec.bottom_lstm(self.dec(y_s),(bot_ht_1,bot_mt_1)) #torch.cat([self.dec(y_s),global_motion.unsqueeze(0)],dim=-1)
                _,(top_ht_1,top_mt_1) = self.dec.top_lstm(bot_ht_1,(top_ht_1,top_mt_1))

                # ------------ SPATIAL ATTENTION
                hidden_dim = self.hidden_dim
                # FORMULA : s_t = epsilon_t = W.tanh(W_a.h_t + U_a.R + b_a)
                s_t = self.spatial_att(torch.tanh(self.feat_extract_hdec(bot_ht_1).expand(x.size()[:2] + (hidden_dim,))+self.feat_extract_x(R)))
                # alpha_t = softmax(s_t) # shape : (T[src_len], N, K) N:batch_size
                alpha_tk = torch.softmax(s_t,dim=-1) # (,,K)

                alpha_tk = alpha_tk.masked_fill(enc_masks.unsqueeze(-1).to(self.device) == 0, 0.)

                self.spatial_attentions[j] = alpha_tk

                # ------ SPATIAL FEATURES shape : (T,N,V*C)
                # ------ Region features

                spatial_features = torch.mean(torch.mul(xparts, alpha_tk.unsqueeze(-2)), dim=-1).to(self.device)

                # ------ TEMPORAL ATTENTION shape : (T,N,V*C)--------------------------------------------------------
                ep_t = self.temporal_att(torch.tanh(self.tempfeat_extract_hdec(bot_ht_1).expand(x.size()[:2] + (hidden_dim,))+self.feat_extract_g(R)))
                ep_t = ep_t.masked_fill(enc_masks.unsqueeze(-1).to(self.device) == 0, float('-inf'))
                b_t = torch.softmax(ep_t,dim=0)

                #------ TEMPORAL GAUSSIAN APPROXIMATION -------------------------------------------------------------
                frames = torch.arange(0,b_t.shape[0],1).unsqueeze(-1).unsqueeze(-1).to(self.device)
                # MEAN #
                mean_ga = torch.sum(torch.mul(b_t,frames),dim=0)
                # STD #
                std = torch.sqrt(torch.sum(torch.mul(b_t,(frames-mean_ga.unsqueeze(0).expand_as(b_t))**2),dim=0))

                # BUILD GAUSSIAN WINDOW
                src_len = max(src_lens)
                s = torch.arange(0, src_len, device=self.device).reshape(src_len, 1, 1).expand(src_len, len(src_lens),1)
                sigma = std.reshape(1, len(src_lens), 1).expand(s.size())
                window = torch.exp(-(mean_ga.unsqueeze(0) - s) ** 2 / (2 * sigma ** 2))
                self.temporal_attentions[j] = window.squeeze(-1)

                # ------ APPLY SPATIAL TEMPORAL ATTENTION: (T,N,V*C)
                sp_temp_fet = torch.mul(spatial_features,window)

                # ------ CONTEXT VECTOR shape : (N,V*C,1)
                context_vector = torch.sum(sp_temp_fet.masked_fill(enc_masks.unsqueeze(-1).to(self.device) == 0, 0.),axis=0)

                # ------ Gate variable beta
                beta = torch.sigmoid(self.gate_var(bot_ht_1)+self.adapt_layer(self.dec.embedding(y_s)) ).squeeze(0)

                self.beta[j] = beta.squeeze(-1)

                adaptive_context_vector = beta*torch.tanh(self.ctproject(context_vector))+\
                                          (1-beta)*torch.tanh(self.htproject(top_ht_1.squeeze(0)))

                # ------ Word Probability distribution pt = softmax(Up.tanh(Wp[ht; ̄ct] + bp) +d)
                language_or_motion = self.mixture_feat(torch.cat([self.dec(y_s).squeeze(0),adaptive_context_vector,bot_ht_1.squeeze(0)],dim=-1) ) # Wp[ht; ̄ct] + bp
                dec_output = self.final_layer( torch.tanh(language_or_motion) ).unsqueeze(0) # pt

                self.dec_output = dec_output
                output_list.append(dec_output)
                dec_next_input = torch.argmax(torch.softmax(dec_output, dim=2), dim=2)  # (1,batch_size)
                dec_pred_output.append(dec_next_input)
                self.dec_next_input = dec_next_input

            del b_t,spatial_features,ep_t,alpha_tk
            torch.cuda.empty_cache()

            self.output_list = output_list

            self.spat_temp = torch.mul(self.spatial_attentions,self.temporal_attentions.unsqueeze(-1))

            dec_pred_output = torch.cat(dec_pred_output, dim=0)
            # For evaluation
            self.dec_pred_output = dec_pred_output
            self.target_and_prediction = [y, dec_pred_output]
            outputs_logits = torch.cat(output_list, dim=0)
            # outputs shape : (trg_len,batch_size, output_dim) output_dim --> logits
            return outputs_logits

        else:
            self.x = x
            init_hidden = torch.zeros((1,x.size(1), self.hidden_size),device=self.device)
            decoded_batch  = beam_decode(self, y, (init_hidden,init_hidden), xparts)
            return decoded_batch
