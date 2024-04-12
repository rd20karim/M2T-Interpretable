import torch
from matplotlib import ticker
from torch.nn.utils.rnn import pad_sequence
from datasets.visualization import decode_predictions_and_compute_bleu_score

import numpy as np

def shift_poses(poseswtx,n_joint=22):
    temp = np.concatenate(poseswtx, axis=0).reshape(-1, n_joint, 3)
    x = temp[:, :, 0].flatten()
    y = temp[:, :, 1].flatten()
    z = temp[:, :, 2].flatten()
    # m,n,u,v,r,p = max(x),max(y),max(z),min(x),min(y),min(z)
    # print(m,n,u,v,r,p)

    sx, sy, sz = [np.sqrt(np.var(cord)) for cord in [x, y, z]]
    mx, my, mz = [np.mean(cord) for cord in [x, y, z]]

    normalized_poses = []
    meandata = np.expand_dims(np.array([mx, my, mz]), axis=(0, 1))
    stddata = np.expand_dims(np.array([sx, sy, sz]), axis=(0, 1))
    for k in range(len(poseswtx)):
        normalized_poses.append((poseswtx[k].reshape(-1, n_joint, 3) - meandata) / stddata)

    shift_poses = []
    for k in range(len(normalized_poses)):
        shift_poses.append(
            normalized_poses[k] - np.expand_dims(normalized_poses[k].reshape(-1, n_joint, 3)[0, 0, :], axis=(0, 1)))

    return np.asarray(normalized_poses, dtype=object)

def visualize_attention(input_sentence, output_words, attentions ,fig ,i ,fsz=20):
    # Set up figure with colorbar
    # fig = plt.figure()
    fig.set_facecolor("white")
    ax = fig.add_subplot(i)
    cax = ax.matshow(attentions  )  # , cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    print("input " ,len(input_sentence.split(' ')))
    print("output" ,len(output_words.split(' ')))
    print("attention shape" ,attentions.shape)
    ax.set_xticklabels([''] + input_sentence.split(' ') , rotation=90 ,fontsize = fsz)
    ax.set_yticklabels([''] + output_words.split(' ') ,fontsize=fsz)
    ax.set_title("Input words" ,fontsize=fsz ,color='red')
    # ax.set_xlabel("input words)
    ax.set_ylabel("predicted words" ,fontsize=fsz ,color='green')
    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))


def calculate_attention_batch(data_loader ,loaded_model ,vocab_obj,indx_batch=0,word=None,multiple_references=True,device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    loaded_model.eval()
    if word: indx_batch=-1
    for idx, batch in enumerate(data_loader):
        if idx == indx_batch: break # or data_loader.dataset.lang.token_to_idx[word] in
    #bx,by,lens= batch

    # # Calculate predictions
    n_joint = batch[0].shape[2]//3

    # Relative coordinates except for the root

    poses_input = batch[0] - batch[0][:, :, :3].unsqueeze(-1).expand(batch[0][:, :, :3].size() + (n_joint,)).permute(0,1,3,2).reshape(
        batch[0].size()[:2] + (n_joint*3,))
    poses_input[:,:,:3] = batch[0][:, :, :3]

    src = poses_input.to(device).permute(1, 0, 2)

    # src = batch[0].to(device).permute(1, 0, 2)
    src = torch.as_tensor(src, dtype=torch.float32)
    # shape (batch_size,src_len,flatten joint dim = n_joint*3)
    trg = batch[1].to(device).permute(1, 0) if not multiple_references else \
        pad_sequence([torch.as_tensor(refs[0]) for refs in batch[1]], batch_first=False, padding_value=0).to(device)
    src_lens = batch[2]  # (batch_size,)
    # trg_lens = torch.sum(trg.masked_fill(trg !=0, 1),dim=0) #    trg_lens = batch[3]
    output_pose = loaded_model(src, trg, teacher_force_ratio=0, src_lens=src_lens)
    _dec_numeric_sentence = vocab_obj.decode_numeric_sentence
    # Decode Predictions
    bleu_score, output_predictions, target_sentences = decode_predictions_and_compute_bleu_score(output_pose, batch[1] if multiple_references else trg,
                                                                 vocab_obj,num_grams=4, batch_first=False,multiple_references=multiple_references)

    print("BLEU-4 SCORE BATCH %.3f" % bleu_score)
    #print("input\n", target_sentences, "\noutput\n", output_predictions)
    # Get temporal or/and spatial attention weights for the batch index (idx)
    att_spat = loaded_model.spatial_attentions.cpu().detach().numpy()
    att_temp = loaded_model.temporal_attentions.cpu().detach().numpy()
    att_spat_temp = loaded_model.spat_temp.cpu().detach().numpy()

    betas = loaded_model.beta.cpu().detach().numpy()

    return  att_spat_temp,target_sentences,output_predictions,src_lens,batch[0],betas,att_spat,att_temp