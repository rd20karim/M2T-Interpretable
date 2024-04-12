
import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from queue import PriorityQueue

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 1
EOS_token = 2
MAX_LENGTH = 50
import logging
import random

class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length,att_weights,att_position=None):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        :param att_weights:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length
        self.att_weights = att_weights
        self.att_position = att_position
    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward



def beam_decode(self,target_tensor, decoder_hiddens, encoder_outputs=None):
    '''
    :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch
    '''

    beam_width = self.beam_size
    topk = beam_width
    decoded_batch = []
    dec_pred_output = []


    # decoding goes sentence by sentence
    target_tensor = target_tensor.permute(1,0)
    B = target_tensor.size(0) # batch_size
    #TODO REMOVE DEBUG PARAM IN THE FINALE VESION
    debug_param = B
    #  ------------ Spatial-Temp pose feature : R # shape  : (T,N,V,C)
    T, N, CV = self.x.size()
    R = encoder_outputs.reshape(T, N, 6 * self.hidden_size).to(self.device)

    # TODO CHANGE THIS CODE  TO DECODE PER BATCH
    for idx in range(debug_param):
        #x = xparts #shape[T,B,K,D] (k=6) (D=feature_dimension)
        x = self.x[:,idx,:].unsqueeze(1)
        enc_masks = torch.zeros((x.shape[0],1)) # (seq_len,batch_size)
        enc_masks[:self.src_lens[idx],0]=1
        self.attention_weights_list = []  # torch.empty((src_len,batch_size,trg_len))
        print(f"---\r Beam searching sample {idx+1}/{B} ---", end="")

        if isinstance(decoder_hiddens, tuple):  # LSTM case
            decoder_hidden = (decoder_hiddens[0][:,idx, :].unsqueeze(0),decoder_hiddens[1][:,idx, :].unsqueeze(0))
            # TOP AND BOTTOM HIDDEN STATE INIT
            bot_ht_1 = top_ht_1  = decoder_hidden[0]
            bot_mt_1 = top_mt_1  = decoder_hidden[1]
            decoder_hidden  = (bot_ht_1,bot_mt_1,top_ht_1,top_mt_1)
        else:
            decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)

        encoder_output = encoder_outputs[:,idx, :].unsqueeze(1)
        # Number of sentence to generate
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))
        thr = random.random()
        teacher_force_ratio = 0

        # Start with the start of the special token <sos>
        dec_pred_output.append(torch.ones((1,1), dtype=torch.int, device=self.device))  # first tokens : <sos> index : 1

        decoder_input = target_tensor[idx].unsqueeze(0) if thr < teacher_force_ratio else dec_pred_output[idx]

        # starting node -  hidden vector, previous node, word id, logp, length

        node = BeamSearchNode(hiddenstate=decoder_hidden, previousNode=None, wordId=decoder_input.item(),
                              logProb=0,length=1,att_weights=None,att_position=None)
        nodes = PriorityQueue()
        # start the queue
        nodes.put((-node.eval(), node))
        qsize = 1
        # start beam search
        while True:
        #for idx in range(trg_len-1):
            # give up when decoding takes too long
            if qsize > 100: break
            # fetch the best node

            score, n = nodes.get()

            # Get previous word
            decoder_input = n.wordid

            (bot_ht_1,bot_mt_1,top_ht_1,top_mt_1) = n.h
            #print(f"next word {idx}-->",n.wordid)
            if n.wordid == 2 and n.prevNode != None:
                endnodes.append((score, n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            """"--------------------RUN MODEL PREDICT------------------------"""
            # decode for one step using decoder
            # ------------ TOP and Bottom LSTM prediction
            _, (bot_ht_1, bot_mt_1) = self.dec.bottom_lstm(self.dec(torch.tensor([[decoder_input]],device=self.device)), (bot_ht_1, bot_mt_1))
            _, (top_ht_1, top_mt_1) = self.dec.top_lstm(bot_ht_1, (top_ht_1, top_mt_1))

            # ------------ SPATIAL ATTENTION
            hidden_dim = self.hidden_dim
            s_t = self.spatial_att(
                torch.tanh(self.feat_extract_hdec(bot_ht_1).expand(x.size()[:2] + (hidden_dim,)) + self.feat_extract_x(R)))

            alpha_tk = torch.softmax(s_t, dim=-1)  # (,,K)
            alpha_tk = alpha_tk.masked_fill(enc_masks.unsqueeze(-1).to(self.device) == 0, 0.)
            # torch.cuda.empty_cache()
            spatial_features = torch.mean(torch.mul(encoder_output, alpha_tk.unsqueeze(-2)), dim=-1).to(self.device)

            # ------ TEMPORAL ATTENTION shape : (T,N,V*C)--------------------------------------------------------
            ep_t = self.temporal_att(torch.tanh(
                self.tempfeat_extract_hdec(bot_ht_1).expand(x.size()[:2] + (hidden_dim,)) + self.feat_extract_g(R)))  # +self.speed_temp(velocity)))
            ep_t = ep_t.masked_fill(enc_masks.unsqueeze(-1).to(self.device) == 0, float('-inf'))
            b_t = torch.softmax(ep_t, dim=0)

            # ------ TEMPORAL GAUSSIAN APPROXIMATION -------------------------------------------------------------
            frames = torch.range(0, b_t.shape[0] - 1, 1).unsqueeze(-1).unsqueeze(-1).to(self.device)
            # MEAN
            mean_ga = torch.sum(torch.mul(b_t, frames), dim=0)
            # STD
            std = torch.sqrt(torch.sum(torch.mul(b_t, (frames - mean_ga.unsqueeze(0).expand_as(b_t)) ** 2), dim=0))
            # BUILD GAUSSIAN WINDOW
            src_len = max(self.src_lens)
            s = torch.arange(0, src_len, device=self.device).reshape(src_len, 1, 1).expand(src_len, len(self.src_lens), 1)
            sigma = std.reshape(1, len(self.src_lens), 1).expand(s.size())
            window = torch.exp(-(mean_ga.unsqueeze(0) - s) ** 2 / (2 * sigma ** 2))
            # ------ APPLY SPATIAL TEMPORAL ATTENTION: (T,N,V*C)
            sp_temp_fet = torch.mul(spatial_features, window)

            # ------ CONTEXT VECTOR shape : (N,V*C,1)
            context_vector = torch.sum(sp_temp_fet.masked_fill(enc_masks.unsqueeze(-1).to(self.device) == 0, 0.), axis=0)
            # TODO OPTIMIZE THIS IT SHOULD BE PER BATCH
            context_vector = context_vector[idx,:].unsqueeze(0)
            # ------ Gate variable beta
            beta = torch.sigmoid(self.gate_var(bot_ht_1) + self.adapt_layer(self.dec.embedding(torch.tensor([[decoder_input]],device=self.device)))).squeeze(0)
            # TODO ACTIVE FOR VISUALIZATION
            adaptive_context_vector = beta * torch.tanh(self.ctproject(context_vector)) + \
                                      (1 - beta) * torch.tanh(self.htproject(top_ht_1.squeeze(0)))

            # ------ Word Probability distribution pt = softmax(Up.tanh(Wp[ht; ̄ct] + bp) +d)
            language_or_motion = self.mixture_feat(torch.cat([self.dec(torch.tensor([[decoder_input]],device=self.device)).squeeze(0), adaptive_context_vector, bot_ht_1.squeeze(0)],dim=-1))
            decoder_logits = self.final_layer(torch.tanh(language_or_motion)).unsqueeze(0)
            decoder_hidden = (bot_ht_1,bot_mt_1,top_ht_1,top_mt_1)

            """--------------------------------- END MODEL PREDICTION--------------------------------"""
            #TODO ADAPT
            # decoder_logits, decoder_hidden = self.dec(torch.tensor([[decoder_input]],device=self.device), decoder_hidden, encoder_output,enc_masks)

            decoder_output = torch.log_softmax(decoder_logits,axis=-1)
            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(decoder_output.squeeze(0), beam_width)
            nextnodes = []
            # beam loop
            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].view(1, -1)
                log_p = log_prob[0][new_k].item()

                node = BeamSearchNode(hiddenstate=decoder_hidden, previousNode=n, wordId=decoded_t.item(),
                                      logProb=n.logp + log_p, length=n.leng + 1, att_weights=None,
                                      att_position=None)

                score = - node.eval()
                nextnodes.append((score, node))
            # put them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, nn))
                # increase qsize
            qsize += len(nextnodes) - 1

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]
        utterances = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.wordid)
            # back trace
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.wordid)
            # reverse words to have the correct order eos->sos >> sos->eos
            utterance = utterance[::-1]
            utterances.append(utterance)
        decoded_batch.append(utterances)

    return decoded_batch

if __name__=="__main__":
    hidden_size = 64
    embedding_dim = 64
    #decoder = seq2seq(642, hidden_size, embedding_dim, num_layers=1, device=device,
    #                  bidirectional=False, attention="local", mask=True, beam_size=2).to(device)
