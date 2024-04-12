import argparse
import logging
import torch
import yaml
from datasets.loader import build_data
from train_wandb import run_batch
from bleu_from_csv import read_pred_refs,calculate_bleu
global min_freq,batch_size,hidden_size,embedding_dim


def load_model_config(args= None,device=None):
    multiple_references = args.multiple_references; path_=args.path; attention_type = args.attention_type
    hidden_size = args.hidden_size; embedding_dim = args.embedding_dim
    min_freq = args.min_freq; batch_size = args.batch_size

    project_path = r"C:\Users\karim\PycharmProjects\SemMotion"
    aug_path = r"C:\Users\karim\PycharmProjects\HumanML3D"

    if "kit" in args.dataset_name:

        # ------------ [Augmented-KIT] ------------
        from architectures.LSTM_kit import seq2seq
        from datasets.kit_h3mld import dataset_class
        path_txt = project_path+"\datasets\sentences_corrections.csv"
        path_motion = aug_path+"\kit_with_splits_2023.npz"

        # -----------  H3D IMPORTS   ---------------------
    elif args.dataset_name=="h3D":
        from architectures.LSTM_h3D import seq2seq
        from datasets.h3d_m2t_dataset_ import dataset_class
        path_txt = aug_path+"\sentences_corrections_h3d.csv"
        path_motion = aug_path+"\\all_humanML3D.npz"

    train_data_loader, val_data_loader, test_data_loader = build_data(dataset_class=dataset_class, min_freq=min_freq,
                                                                      path=path_motion,
                                                                      train_batch_size=batch_size,
                                                                      test_batch_size=batch_size,
                                                                      return_lengths=True, path_txt=path_txt,
                                                                      return_trg_len=True, joint_angles=False,
                                                                      multiple_references=multiple_references)

    input_dim = train_data_loader.dataset.lang.vocab_size_unk

    logging.info("VOCAB SIZE  = %d "  % (input_dim))

    loaded_model = seq2seq(input_dim, hidden_size, embedding_dim, num_layers=1, device=device,attention=attention_type,
                           beam_size=args.beam_size,hidden_dim=args.hidden_dim,K=args.K).to(device)
    print(loaded_model)
    model_dict = torch.load(path_,map_location=torch.device(args.device))
    loaded_model.load_state_dict(model_dict["model"])
    return loaded_model,train_data_loader,test_data_loader,val_data_loader


def evaluate(loaded_model,data_loader,mode,multiple_references=False,name_file=None,beam_size=1):
    loaded_model.eval()
    epoch_loss = 0
    output_per_batch,target_per_batch = [],[]
    name_file = f"./Predictions/LSTM_{args.dataset_name}_{name_file}_{args.lambdas}"
    BLEU_scores = []
    with open(name_file + ".csv", mode="w") as _:pass
    logging.info(f"Compute BLEU scores per batch and write predictions/refs to {name_file}")
    loaded_model.eval()
    if beam_size==1:
        for i, batch in enumerate(data_loader):
            loss_b,bleu_score_4,pred,refs = run_batch(model=loaded_model,batch=batch,data_loader=data_loader,mode=mode,teacher_force_ratio=0,
                                                          device=device,multiple_references=multiple_references,attention_type=args.attention_type)


            BLEU_scores.append(bleu_score_4)
            # write predictions and targets for every batch
            with open(name_file + ".csv", mode="a") as f:
                for p, t in zip(pred, refs):
                    f.writelines(("%s" + ",%s" * len(t) + "\n") % ((" ".join(p).replace("\n", ""),) + tuple(" ".join(k).replace("\n", "") for k in t)))

            logging.info("Loss/test_batch %d --> %.3f  BLEU score_batch %.3f" % (i, loss_b, bleu_score_4))
            epoch_loss += loss_b.item()
        loss = epoch_loss / len(data_loader)

        BLEU_score = sum(BLEU_scores) / len(BLEU_scores)
        logging.info(f"LOSS {mode} %.3f BLEU_4 score %.3f" % (loss, BLEU_score))

        # #---------------- CORPUS-LEVEL BLEU SCORE -----------------------------
        # path = "./"+name_file+".csv"
        # pa, ra = read_pred_refs(path, split=True)
        # BLEU_score_CORPUS= calculate_bleu(pa, ra, num_grams=4)
        # logging.info(f"CORPUS-LEVEL BLEU score %.3f" % (BLEU_score_CORPUS))


    else:
        logging.info("START BEAM SEARCHING")
        file_save_beam = f"./Predictions/LSTM_{args.dataset_name}_preds_{args.lambdas}_beam_size_{beam_size}.csv"
        with open(file_save_beam,'w'): pass #create the file
        beam_bleus = [[] for _ in range(beam_size)]
        for i, batch in enumerate(data_loader):
            bleu_score_beam, predicted_sentences, refs = run_batch(model=loaded_model, batch=batch, data_loader=data_loader, mode=mode, teacher_force_ratio=0,
                                                                device=device, multiple_references=multiple_references, beam_size=beam_size,
                                                                attention_type=args.attention_type,file_beam=file_save_beam)
            for bm, sc in enumerate(bleu_score_beam):
                logging.info(f"BATCH-BLEU_4 score beam {bm} --> %.3f" % (sc,))
                beam_bleus[bm].append(sc)

        for bm in range(len(beam_bleus)):
            print(f'CORPUS-BLEU@4 - {bm+1} ---> {sum(beam_bleus[bm])/len(beam_bleus[bm])}')

    return output_per_batch,target_per_batch

if __name__=="__main__":

    home_path = r"C:\Users\karim\PycharmProjects"
    abs_path_project = home_path + "\m2LSpTemp"

    parser = argparse.ArgumentParser()
    parser.add_argument("--path",type=str,help="Path of model weights not used if the config file is passed")
    parser.add_argument("--dataset_name",type=str,default="kit",choices=["h3D","kit"])
    parser.add_argument("--config",type=str,default="../configs/lstm_eval_h3D.yaml")
    parser.add_argument("--device",type=str,default="cuda")
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
    parser.add_argument("--use_unknown_token", type=str, default=True, help='To use or not the unknown token for evaluation')
    parser.add_argument("--K",type=int,default=6,help="number of spatial part attention")
    parser.add_argument("--batch_size", type=int, default=1024, help='Batch size should be >= to length of data to not have a variable BLEU score')
    args = parser.parse_args()


    # FIRST GET THE RUN ID FROM THE WANDB DASHBOARD ARTIFACTS --------------------------------------
    # OR INDICATE FULL MODEL PATH
    run_id = 'ton5mfwh'
    args.dataset_name = 'kit'
    args.path = abs_path_project + f"\models\Interpretable_MC_{args.dataset_name}_f\model_{run_id}"

    # Then get model config -------------------------------------------------------------------
    config = torch.load(args.path,map_location=torch.device(args.device))["metadata"]
    parser.set_defaults(**config)
    args = parser.parse_args()
    args.path = abs_path_project + f"\models\Interpretable_MC_{args.dataset_name}_f\model_{run_id}"
    device = torch.device(args.device)

    # ------------------------------------------------------------------------------------------
    # Fix manually
    # args.beam_size = 1
    # args.device = "cuda:0"
    # args.batch_size = 64
    # device = torch.device(args.device)


    # Update args parser
    # args.config =f"../configs/lstm_eval_{args.dataset_name}.yaml"
    #
    # with open(args.config,'r') as f:
    #     choices = yaml.load(f,Loader=yaml.Loader)
    # default_arg = vars(args)
    # parser.set_defaults(**choices)
    # args = parser.parse_args()


    loaded_model, train_data_loader, test_data_loader,val_data_loader = load_model_config(device=device,args=args)
    data_loader = locals()[args.subset+"_data_loader"]
    logging.info(f"Checkpoint on {args.subset} set")

    print(args)
    output_per_batch, target_per_batch = evaluate(loaded_model=loaded_model, data_loader=data_loader,multiple_references=args.multiple_references,
                                                  name_file=args.name_file,beam_size=args.beam_size,mode=args.subset)



    # Write predictions only
    # pa, _ = read_pred_refs(f"./Predictions/LSTM_{args.dataset_name}_preds_{args.lambdas}.csv", split=True)
    # with open("temp.csv","w") as f:
    #     for token_list in pa:
    #         f.write(' '.join(token_list)+"\n")