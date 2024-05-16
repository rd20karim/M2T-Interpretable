import sys
import torch
from src.evaluate_m2L import  load_model_config
import matplotlib.pyplot as plt
from visualizations.attention_visualization import calculate_attention_batch
from visualizations.subplot_3d_with_txt import SubplotAnimation
from matplotlib import rc
import seaborn as sns
from matplotlib import patches
import os
import argparse
import pandas as pd
import numpy as np
from torchtext.data.metrics import bleu_score
import yaml
import spacy

joint_names = ["BP","BT","BLN","BUN","EY","LS","LE","LW","RS","RE","RW",
               "LH","LK","LA","LM","LF","RH","RK","RA","RM","RF"]

part_names = ["Root","Torso","LeftArm","RightArm","LeftLeg",'RightLeg']

actions = ['wave', 'stumble', 'kick', 'wipe', 'stand', 'throw', 'punch', 'bend', 'lift', 'bow', 'stretch',
           'pick', 'boxing', 'open', 'rotate', 'clean', 'stomp', 'bend', 'squat', 'squad', 'kneel', 'handstand',
           'walk', 'turn', 'run', 'jump', 'mov', 'play', 'jog', 'climb']

def map_pose2concept(intensity,src_poses,start_pad,pred,sample_id,name_directory=None,
                     idxs=None,ref=None,_format='.mp4',save=False,dataset_name='h3D',betas=None):
    #W = att_w[:start_pad+1,:len(pred)]
    # idxs = np.argmax(W,axis=0) if idxs is None else idxs
    # #print(idxs)
    Frames_indxs = np.arange(start_pad)
    #idxs = np.round(Frames_indxs@W)
    n_joint = 21 if "kit" in dataset_name else 22
    beta_words = betas[:len(pred),sample_id]
    id_w = np.argmax(beta_words)
    #intensity = intensity[id_w,:,sample_id,:]
    intensity = intensity[:len(pred), :, sample_id, :] # (Ty,Tx,J)
    #sum_intensity = np.sum(intensity,axis=0) # sum across words
    #intensity = sum_intensity/np.max(sum_intensity)
    idxs = np.argmax(np.max(spat_temp[:len(pred), :, sample_id, :], axis=-1), axis=1)
    ani = SubplotAnimation(src_poses, frames=Frames_indxs,use_kps_3d=range(n_joint),sample=sample_id,
                           down_sample_factor=1,idxs=idxs ,pred=pred,ref=ref,dataset_name=dataset_name,
                           intensity=intensity,beta_words=beta_words,att_temp=temp_att_batch[:len(pred),:,id_sample])
    if save:
        os.makedirs(name_directory,exist_ok=True)
        name_file = name_directory+f"/attention_sample_{sample_id}"+_format
        ani.save(name_file)
        print("Animation saved in ", os.getcwd() + "/" + name_file)
    plt.rc('animation', html='jshtml')
    return ani


def save_attention_figs(limit,start=0):
    global df
    os.makedirs(args.save_results,exist_ok=True)
    for id_sample in range(start,len(lens[:limit])):
        att_w = spat_temp[:,id_sample,:]
        prediction = preds[id_sample]
        start_pad = lens[id_sample]
        id_best_ref = np.argmax([bleu_score([prediction],[[ref]]) for ref in trgs[id_sample]])
        trg = trgs[id_sample][id_best_ref]
        df = pd.DataFrame(data=att_w[:start_pad,:len(prediction) ],columns=prediction)
        fig,ax = plt.subplots(figsize=(len(df)//3,len(prediction)//2))
        fsz= 12
        min_att = df[df>.005].idxmin(axis=0).values
        max_att = df.idxmax(axis=0).values
        ax = sns.heatmap((df).transpose(),annot=False,fmt=".0f",cmap="viridis",linewidths=0.009,ax=ax)
        iw = 0
        for indx_min,indx_max in zip(min_att,max_att):
            ax.add_patch(patches.Rectangle((indx_min,iw), 1, 1, fill=False, edgecolor='orange', lw=2))
            ax.add_patch(patches.Rectangle((indx_max,iw), 1, 1, fill=False, edgecolor='red', lw=3))
            iw += 1
        ax.set_xlabel(f" Frame index ", fontsize=fsz)
        ax.set_ylabel(" Predicted words ", fontsize=fsz, color='green')
        ax.tick_params(labelbottom=True, labeltop=True,bottom=True, top=True)
        ax.set_title(f" {' '.join(trg) } ", fontsize=fsz, color='red')
        fig.tight_layout()
        fig.savefig(args.save_results+f"/attention_sample_{id_sample}")


def save_spatial_attention_figs(args,save_spTemp=True):
    global df
    os.makedirs(args.save_results,exist_ok=True)
    nlp = spacy.load('en_core_web_md')

    # Save spatio-temporal attention maps
    att_st_word = {}
    beta_word = {}
    for id_sample in range(args.n_map):#range(start,len(lens[:limit]))
        start_pad = lens[id_sample]
        prediction = preds[id_sample]
        beta = loaded_model.beta.cpu().detach().numpy()
        BLEUS = [bleu_score([prediction],[[ref]]) for ref in trgs[id_sample]]
        id_best_ref = np.argmax(BLEUS)
        best_bleu  = BLEUS[id_best_ref]
        print(f"\r id_sample {id_sample} ------> ",best_bleu,end='')
        trg = trgs[id_sample][id_best_ref]
        min_bleu = 0 # 0.1
        for index, _token in enumerate(prediction):
            id_complet = _token + f'_s-{id_sample}-id-{index}'
            part_names = ["Root", "Torso", "LeftArm", "RightArm", "LeftLeg", 'RightLeg']
            if (nlp(_token)[0].pos_ == 'VERB' or _token in actions) and best_bleu >= min_bleu :
                att_w = spat_temp[index][:start_pad,id_sample,:]
                if args.dataset_name == 'h3D': att_w = att_w[:, [0, 1, 3, 2, 5, 4]]
                # mean_token,std_token = statistics[index,id_sample]
                att_st_word[id_complet] = {'spatemp':att_w,'pred':' '.join(prediction),
                                           'trg': ' '.join(trg),'part_avg_attention':att_w.sum(-1)}

                if save_spTemp:  # Take time only for spatio-temporal attention visualization
                    df = pd.DataFrame(data=att_w,columns=part_names)
                    fig,ax = plt.subplots(figsize=(len(df)//3,len(part_names)//2))
                    fsz = 15
                    ax = sns.heatmap((df).transpose(),annot=False,fmt=".1f",cmap="viridis",linewidths=0.009,ax=ax)
                    ax.set_ylabel(" Body part  ", fontsize=fsz, color='green')
                    twin_ax = ax.twinx()
                    #twin_ax.yaxis.tick_right()
                    # twin_ax([k+0.5 for k in range(len(prediction))])
                    twin_ax.set_yticks([])
                    print(id_sample,index)

                    value_parts = [str(value) for value in np.round(beta[:len(prediction), id_sample], decimals=2)]
                    wb  = [prediction[i]+ ' ' + value_parts[i] for i in range(len(prediction))]
                    #ax.set_xlabel( ' '.join(prediction) , fontsize=fsz, color='red')
                    #ax.set_xlabel('\n'+ ' '.join(value_parts) , fontsize=fsz, color='blue')
                    twin_ax.set_ylim(ax.get_ylim())
                    COLOR = "darkviolet"
                    row_token = ("P="+' '.join(prediction))#.split(" ")
                    row_betas = ("$%s$"% "\\beta="+' '.join(value_parts))# .split(" ")
                    ax.set_title(row_token+'\n'+row_betas,loc='center',  fontsize=fsz, color=COLOR)
                    ax.set_xlabel("R=" + f"{' '.join(trg) }" , fontsize=fsz, color='darkorange')

                    fig.tight_layout()
                    save_to = args.save_results+f"/Histograms/{id_complet}_.pdf"
                    print("Saving... ",save_to)
                    fig.savefig(save_to,dpi=300,bbox_inches="tight")
                    plt.close(fig)

            # beta values for "grouped by word" category
            beta_word[id_complet]= beta[index, id_sample]
    return  att_st_word,beta_word


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path",type=str,help="Path of model weights not used if the config file is passed")
    parser.add_argument("--dataset_name",type=str,default="h3D",choices=["h3D","kit"])
    parser.add_argument("--run_id",type=str,default="h3D",choices=["h3D","kit"])
    parser.add_argument("--kind",type=str,default="map",choices=["map","hist",'adapt','all'])
    parser.add_argument("--config",type=str,default="../configs/lstm_eval_h3D.yaml")
    parser.add_argument("--device",type=str,default="cpu")
    parser.add_argument("--multiple_references",type=str,default="True",choices=["True","False"],help="Specify evaluation mode use flattened references or all for corpus level")
    parser.add_argument("--encoder_type",type=str,default="MLP")
    parser.add_argument("--attention_type",type=str,default="relative_bahdanau")
    parser.add_argument("--name_file", type=str, default="preds", help="File name of predicted sentences")
    parser.add_argument("--subset", type=str, default="test", help="Subset on which evaluating the data",choices=["test", "val","train"])
    parser.add_argument("--hidden_dim", type=int, default=128, help='hidden dimension of the encoder layers for the MLP')
    parser.add_argument("--hidden_size", type=int, default=64, help='hidden size of the decoder and encoder output dimension')
    parser.add_argument("--embedding_dim", type=int, default=64, help='embedding dimension of words')
    parser.add_argument("--min_freq", type=int, default=3, help='minimum word frequency to keep in the dataset')
    parser.add_argument("--beam_size", type=int, default=1, help='beam search width')
    parser.add_argument("--K",type=int,default=6,help="number of spatial part attention")
    parser.add_argument("--batch_size", type=int, default=1024, help='Batch size should be >= to length of data to not have a variable BLEU score')
    parser.add_argument("--n_map",type=int,default=10,required=False,help="Number of attention map to generate")
    parser.add_argument("--n_gifs",type=int,default=100,help="Number of animation to generate")
    parser.add_argument("--save_results",type=str,help="Directory where to save generated plots")
    # parser.add_argument("--start",type=int,default=0,help="Start sample index generation point")
    # parser.add_argument('--indexs',type=int, nargs='+', help='word indexes')
    # parser.add_argument('--spat',type=int, nargs='+', help='sample indexes')

    args = parser.parse_args()
    home_path = r"C:\Users\karim\PycharmProjects"
    abs_path_project = home_path + "\M2T-Interpretable"

    # # Fix manually ---------------------------------------------------------------------------------
    # run_id =  'ton5mfwh'
    # args.dataset_name = 'kit'

    run_id = args.run_id

    args.path = abs_path_project + f"\models\Interpretable_MC_{args.dataset_name}_f\model_{run_id}"

    # From the loaded model -------------------------------------------------------------------
    config = torch.load(args.path,map_location=torch.device(args.device))["metadata"]
    parser.set_defaults(**config)
    args = parser.parse_args()
    args.dataset_name = 'kit' # Important twice because is not default
    args.path = abs_path_project + f"\models\Interpretable_MC_{args.dataset_name}_f\model_{run_id}"


    # ------------------------------------------------------------------------------------------
    # # Model with no config : Update args parser--------------------------------
    # with open(args.config,'r') as f:
    #     choices = yaml.load(f,Loader=yaml.Loader)
    # parser.set_defaults(**choices)
    # args = parser.parse_args()
    #
    # # Saved Model with config -------------------------------------------------------------------
    # config = torch.load(args.path,map_location=torch.device(args.device))["metadata"]
    # parser.set_defaults(**config)
    # args = parser.parse_args()
    # ------------------------------------------------------------------------------------------

    # Fix manually ---------
    args.beam_size = 1
    args.multiple_references = True if args.multiple_references=="True" else False
    args.batch_size = 415
    args.n_gifs = 415
    args.save_results = f"./Animations/{args.dataset_name}/{args.lambdas}"
    device = torch.device(args.device)

    loaded_model, train_data_loader, test_data_loader,val_data_loader = load_model_config(device=device,args=args)
    data_loader = locals()[args.subset+"_data_loader"]

    # Compute batch attention -------------------------------------------------------------------------------------
    indx_batch = 0
    spat_temp, trgs, preds, lens, src_poses,betas,spat_att_batch, temp_att_batch = calculate_attention_batch(data_loader=data_loader, loaded_model=loaded_model,
                                                                              vocab_obj=data_loader.dataset.lang,indx_batch=indx_batch,
                                                                              word=None, multiple_references=args.multiple_references,device=device)


    id_w2p = None
    body_parts = loaded_model.enc_pose.body_parts
    n_joint  = 22 if args.dataset_name=="h3D" else 21
    intensity = np.zeros(spat_temp.shape[:-1]+(n_joint,))
    for idp, part in enumerate(body_parts):
        intensity[:,:,:,part] = spat_temp[:,:,:,idp:idp+1]


    # --------------------------- Human Pose 3D Animation ---------------------------------------------------------

    for ll, id_sample in enumerate((range(args.n_gifs))):#[0,5,13,25,42,43,53,54,59]
      pred = preds[id_sample]
      bleus = [bleu_score([pred], [[ref]]) for ref in trgs[id_sample]]
      id_best_ref = np.argmax(bleus)
      if np.max(bleus)>=.3: # Select predictions above a threshold
          trg = trgs[id_sample][id_best_ref]
          trg = ' '.join(trg)
          start_pad = lens[id_sample]
          map_pose2concept(intensity,np.asarray(src_poses.cpu()),start_pad,pred,
                           sample_id = id_sample,name_directory=args.save_results,ref=trg,
                           idxs=None,dataset_name=args.dataset_name,save=True,betas=betas)


    # TODO DECOMMENT THE FOLLOWING FOR INTERPRETABILITY ANALYSIS
    # # # plt.figure(1)
    # # for id_sample in range(args.start,args.start+args.n_gifs):
    # #     print(id_sample)
    # #     id_best_ref = np.argmax([bleu_score([preds[id_sample]], [[ref]]) for ref in trgs[id_sample]])
    # #     trg = trgs[id_sample][id_best_ref]
    # #     trg = ' '.join(trg)
    # #     map_pose2concept(None,np.asarray(src_poses[:lens[id_sample],id_sample,:].cpu()),lens[id_sample],
    # #                      preds[id_sample],sample_id = id_sample,name_directory=args.save_results,ref=trg)
    #
    # # --------------------- GENERATE AND SAVE SPATIAL-TEMPORAL ATTENTION MAPS -----------------------------------
    #
    # args.n_map = len(spat_temp[0,0])
    # att_st_word,beta_word = save_spatial_attention_figs(args,save_spTemp=True)
    #
    # # if args.n_map:
    # #     save_attention_figs(limit=args.start+args.n_map,start=args.start)
    #
    # #--------------------- Interpretability analysis ----------------------------------------------------
    #
    # import matplotlib.pyplot as plt
    # from nltk.stem import PorterStemmer
    # import numpy as np
    #
    # # Create a stemmer object
    # stemmer = PorterStemmer()
    #
    # # ------------------------ WORD STEMMING ---------------------------------------
    #
    # # Dictionary to store stemmed words and corresponding maximum attention values and body parts
    # stemmed_data = {}
    # part_names = {
    #     0: 'Root',
    #     1: 'Torso',
    #     2: 'LeftArm',
    #     3: 'RightArm',
    #     4: 'LeftLeg',
    #     5: 'RightLeg'
    # }
    # data_dict = att_st_word
    #
    # # Stem each and group corresponding attention map data
    # for key, value in data_dict.items():
    #     wordname, sid = key.split('_')
    #     id_sample = sid.split("-")[0]
    #     id_tk = sid.split("-")[-1]
    #     word_stem = stemmer.stem(wordname)
    #     if word_stem not in stemmed_data:
    #         stemmed_data[word_stem] = {'values': {i: [] for i in range(6)}, 'parts': list(range(6))}
    #
    #     # deprecated
    #     max_attention = np.max(value['spatemp'], axis=0)
    #     #part_avg_attention = np.max(value['spatemp'], axis=0)
    #     # part_avg_attention = value["part_avg_attention"]
    #     numb_parts = len(part_names)
    #     for i in range(numb_parts):
    #         stemmed_data[word_stem]['values'][i].append(max_attention[i])
    #
    #
    # # ------------------------ HISTOGRAMS BODY PART ATTENTION ---------------------------------------
    #
    # # Create histograms
    # # fig, axs = plt.subplots(len(stemmed_data), figsize=(10, 6 * len(stemmed_data)))
    # # Plot histograms for each stemmed word-name and add part labels
    # hist_path = f"./Histograms/{args.dataset_name}_{args.lambdas}"
    # os.makedirs(hist_path, exist_ok=True)
    # for i, (stemmed_word, data) in enumerate(stemmed_data.items()):
    #     fig,ax = plt.subplots(1) #axs[i]
    #     ax.set_title(f'{", ".join(set([key.split("_")[0] for key in data_dict.keys() if stemmed_word in stemmer.stem(key.split("_")[0])]))}')
    #     for part_index, part_values in data['values'].items():
    #         ax.hist(part_values, alpha=0.5, bins=25, label=part_names[part_index],rwidth=0.9,density=False)
    #
    #     ax.set_xlabel('Maximum Attention Value',fontsize=13,color='black')
    #     ax.set_ylabel('Count')
    #     list_stem_words = f'{", ".join(set([key.split("_")[0] for key in data_dict.keys() if stemmed_word in stemmer.stem(key.split("_")[0])]))}'
    #     ax.set_title(list_stem_words,color="teal",fontsize=15)
    #     ax.set_xlim(0, 1)
    #     ax.legend()
    #     # # Get y-axis maximum value for normalization factor
    #     max_y = ax.get_ylim()[1]
    #     ax.text(0., max_y+0.1, f'Total # Words: {len(data["values"][0])}', ha='left',fontsize=12,color="darkviolet")
    #     ax.get_figure().tight_layout()
    #     ax.get_figure().savefig(hist_path + f"/hist_{stemmed_word}_{args.dataset_name}_{args.lambdas}.pdf",dpi=300,bbox_inches="tight")
    #     plt.close(fig)
    # plt.tight_layout()
    # plt.show()
    #
    #
    # # ------------------------ HISTOGRAMS ADAPTIVE GATE ------------------------------------------------
    #
    # all_beta = {}
    # for token,_value in beta_word.items():
    #     try:
    #         all_beta[stemmer.stem(token.split("_")[0])] += [_value]
    #     except KeyError:
    #         all_beta[stemmer.stem(token.split("_")[0])] = [_value]
    #
    #
    # stemmed_token = "walk"
    #
    #
    # #----------------- Motion Words ------------------------------------------
    #
    # df_beta = pd.DataFrame.from_dict(all_beta,orient='index').T
    # motion = ["run","pick","kick","turn","forward","dance","squat","wipes","punch","backward","jump"]
    # stem_motion = [stemmer.stem(m) for m in motion]
    # ax = df_beta[stem_motion].plot(kind='density',fontsize=13)
    # ax.set_xlim([0,1])
    # ax.set_xlabel("$%s$"% "\\beta",fontsize=15)
    # ax.figure.tight_layout()
    # ax.figure.savefig(f"./{args.dataset_name}_{args.lambdas}_Motion.pdf",dpi=300,bbox_inches="tight")
    # plt.show()
    #
    #
    # #--------------- Non-Motion Words -----------------------------------------
    #
    # df_beta = pd.DataFrame.from_dict(all_beta,orient='index').T
    # non_motion = ["a","the","an","like"]
    # stem_non_motion = [stemmer.stem(m) for m in non_motion]
    # ax = df_beta[stem_non_motion].plot(kind='density',fontsize=13)
    # ax.set_xlim([0,1])
    # ax.set_xlabel("$%s$"% "\\beta",fontsize=15)
    # ax.figure.tight_layout()
    # ax.figure.savefig(f"./{args.dataset_name}_{args.lambdas}_nonMotion.pdf",dpi=300,bbox_inches="tight")
    # plt.show()
    #
    #
    # # Search compositional motion samples --------------------------------------------------------------------------
    # from bleu_from_csv import read_pred_refs
    # predictions,references = read_pred_refs(f"../src/Predictions/LSTM_{args.dataset_name}_preds_{args.lambdas}.csv",tokenize=False)
    # for i,k in enumerate(predictions):
    #     if "pick" in k:
    #         print(i,k)
    #         print(references[i])
    #
    #
    # import torch
    # import matplotlib.pyplot as plt
    #
    # # Plot temporal gaussian attention for the chosen sample and motion words ------------------------------------------
    # tensor = loaded_model.temporal_attentions.detach().cpu()
    # id_sample = 373
    # start_pad = lens[id_sample]
    # prediction = preds[id_sample]
    # (Trg_len, src_len, Batch_size) = tensor.shape
    # # Plotting the curves
    # fig,ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed
    # nlp = spacy.load('en_core_web_md')
    # for trg_id in range(len(prediction)):
    #     _token = prediction[trg_id]
    #     id_t = f'{id_sample}-{preds[id_sample][trg_id]}'
    #     if (nlp(_token)[0].pos_ == 'VERB' or _token in actions) and _token!="starts":
    #         plt.plot(range(start_pad), tensor[trg_id, :start_pad, id_sample],
    #                  label=f'{_token}')
    #         # fig.axes[0].text(torch.argmax(tensor[trg_id, :, batch_idx]),1.1)
    #
    # # id_sample = 373
    # x_1 = 10
    # x_2 = 18
    # x_3 = 22
    # x_4 = 29
    #
    # # # id_sample = 51
    # # x_1 = 9
    # # x_2 = 18
    # # x_3 = 39
    # plt.vlines([x_1,x_2,x_3],0,1,colors="darkred",linestyles='--')
    # ax.text(x_1+0.1,0.15,'walks forward',fontsize=14,weight="bold")
    # ax.text(x_2+0.1,0.15,'turns',fontsize=14,weight="bold")
    #
    # ax.text(x_3+0.1,0.15,'walks back',fontsize=14,weight="bold")
    #
    # plt.xlabel('Frame time', fontsize=15)
    # plt.ylabel('Frame weight', fontsize=15)
    # _pred = ' '.join(prediction)
    # plt.title('Pred : '+ _pred+'\n'+'Ref : '+references[id_sample][0], color='darkviolet', fontsize=15) #Temporal Gaussian Attention\n
    # plt.legend()
    # #plt.grid(True)
    # ax.figure.savefig(f"./TAG_{args.dataset_name}-{id_sample}-{args.lambdas}.pdf")
    # plt.show()