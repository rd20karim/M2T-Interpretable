import sys
sys.path.extend([r'C:\Users\karim\PycharmProjects\m2LSpTemp'])
import argparse
from datasets.visualization import decode_predictions_and_compute_bleu_score
import  logging
from datasets.loader import build_data
import os,random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.nn.utils.rnn import pad_sequence
import  yaml
import torchtext

# -------------------- SET THE SEED FOR REPRODUCIBILITY -----------------#
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def run_batch(model, batch, data_loader, mode, teacher_force_ratio, device=None, optimizer=None,
              multiple_references=None, beam_size=1,lambdas=(0,0),attention_type=None,file_beam=None):
    epoch_loss = 0
    TRG_PAD_IDX = data_loader.dataset.lang.token_to_idx["<pad>"]
    L_spat,L_adapt = lambdas

    n_joint = batch[0].shape[2]//3
    if  "relative" in attention_type:
        poses_input = batch[0] - batch[0][:, :, :3].unsqueeze(-1).expand(batch[0][:, :, :3].size() + (n_joint,)).permute(0,1,3,2).reshape(
            batch[0].size()[:2] + (n_joint*3,))
        poses_input[:,:,:3] = batch[0][:, :, :3]
        src = poses_input.to(device).permute(1, 0, 2)
    else:
        src = batch[0].to(device).permute(1, 0, 2)

    src = torch.as_tensor(src, dtype=torch.float32)
    # shape (batch_size,src_len,flatten joint dim = n_joint*3)
    trg = batch[1].to(device).permute(1, 0) if not multiple_references else \
        pad_sequence([torch.as_tensor(refs[0]) for refs in batch[1]], batch_first=False, padding_value=0).to(device)
    src_lens = batch[2]
    gth_gates = torch.nn.utils.rnn.pad_sequence([torch.as_tensor(ik) for ik in batch[3]], batch_first=True, padding_value=-1)
    gth_gates = gth_gates[:,1:].float().to(device)

    gth_alphas = torch.nn.utils.rnn.pad_sequence([torch.as_tensor(ik) for ik in batch[4]], batch_first=True, padding_value=-1)
    gth_alphas = gth_alphas[:,1:,:].expand(max(src_lens),-1,-1,-1).permute(2,0,1,3).float().to(device)
    num_grams = 4
    vocab_obj = data_loader.dataset.lang
    if beam_size==1:
        if "test" in mode: logging.info("START Greedy SEARCH ")
        ## Run model
        output_pose = model(src, trg,teacher_force_ratio=teacher_force_ratio, src_lens=src_lens)

        bleu_score, pred, refs = decode_predictions_and_compute_bleu_score(output_pose.squeeze(0),
                                                                           batch[1] if multiple_references else trg,
                                                                           vocab_obj, num_grams=num_grams,
                                                                           batch_first=False,multiple_references=multiple_references)


        # ADAPTIVE loss----------------------------------------------------------------------------
        # SHAPE gth_gates = [TRG_LEN,BATCH_SIZE]
        loss_adap = torch.nn.BCELoss(reduction='none')(model.beta.permute(1, 0).float(),  gth_gates.float())
        loss_adap = loss_adap.masked_fill(gth_gates == -1, 0) # mask unsupervised tokens
        loss_adap = torch.sum(loss_adap)/torch.count_nonzero(loss_adap)

        # SPATIAL loss----------------------------------------------------------------------------
        # SHAPE gth_alphas = [TRG_LEN, SRC_LEN, Bsize , K]
        loss_spatial = torch.nn.BCELoss(reduction='none')(model.spatial_attentions.float(), gth_alphas.float())
        loss_spatial = loss_spatial.masked_fill(gth_alphas==-1,0) # mask unsupervised tokens
        loss_spatial = torch.sum(loss_spatial)/torch.count_nonzero(loss_spatial)

        # Language Loss --------------------------------------------------------------------------
        criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX, reduction='mean')
        loss_lang = criterion(output_pose.permute(1, 2, 0), trg[1:, :].permute(1, 0))

        # -----------------------  COMBINED LOSS SUPERVISION --------------------------------------

        loss = loss_lang + L_adapt*loss_adap + L_spat*loss_spatial

        logging.info(f"loss_LANG ----> {loss_lang.item()}")
        logging.info(f"loss_SPAT ----> {loss_spatial.item()}")
        logging.info(f"loss_ADAPT----> {loss_adap.item()}")

        if mode == "train":
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        epoch_loss += loss_lang.item() # Loss_lang used to have training curve scale compared to no superv for comparison

        torch.cuda.empty_cache()
        return loss, bleu_score, pred, refs

    else:  # only for evaluation
        logging.info("START BEAM SEARCH")
        decoded_preds = model(src, trg, teacher_force_ratio=0, src_lens=src_lens)
        predicted_sentences = []
        _dec_numeric_sentence = vocab_obj.decode_numeric_sentence
        for hyps in decoded_preds:
            predicted_sentences += [
                [_dec_numeric_sentence(beam_path, remove_sos_eos=True).split(
                    " ") for beam_path in hyps]]
        logging.info("Write beam predictions ...")

        with open(file_beam, "a") as g:
            for m in predicted_sentences:
                g.writelines([" ".join(k) + "," for k in m] + ["\n"])

        Yrefs = batch[1] if multiple_references else trg
        ref_sentences = [[_dec_numeric_sentence(ref, remove_sos_eos=True).split(" ") for ref in refs] for refs in Yrefs]
        # TODO REMOVE THE DEBUG PARAM
        bleu_score_beam = [torchtext.data.metrics.bleu_score(
            candidate_corpus=[m[k] if len(m) >= k + 1 else m[-1] for m in predicted_sentences],
            references_corpus=ref_sentences,
            max_n=num_grams, weights=[1 / num_grams] * num_grams) for k in range(beam_size)]
        return bleu_score_beam, predicted_sentences, Yrefs

def train_m2l():

    # CREATE A DIRECTORY PER PROJECT
    abs_path = r"C:\\Users\karim\PycharmProjects\SemMotion\Wandb"

    os.makedirs(abs_path,exist_ok=True)
    os.makedirs(abs_path + PROJECT_NAME, exist_ok=True)

    with wandb.init() as run:
        # DIR TO SAVE MODEL WITH UNIQUE ID GENERATED PER RUN FOR THE SPECIFIED PROJECT NAME
        unique_path = abs_path + PROJECT_NAME + f'/model_{wandb.run.id}'

        config = dict(wandb.config)
        config["path"] = unique_path

        train_data_loader, val_data_loader, test_data_loader = build_data(dataset_class=dataset_class,
                                                                          min_freq=config["min_freq"],
                                                                          train_batch_size=config["batch_size"],
                                                                          test_batch_size=config["batch_size"],
                                                                          return_lengths=True,
                                                                          path_txt=path_txt,
                                                                          return_trg_len=True,
                                                                          joint_angles= False,
                                                                          multiple_references=args.multiple_references)

        input_dim = train_data_loader.dataset.lang.vocab_size_unk
        logging.info("VOCAB SIZE  = %d " % (input_dim))

        # -------------------- CREATE MODEL (MLP-Mixer-2-LSTM) -----------------------

        model = seq2seq(input_dim, hidden_size=config["hidden_size"], embedding_dim=config["embedding_dim"], num_layers = config["num_layers"], device = config["device"],
                        dropout = config["rate_dropout"],attention = config["attention_type"], hidden_dim = config["hidden_dim"], K = config["K"] )

        model = model.to(config["device"])
        logging.info(f"Model Architecture {model}")
        optimizer = optim.AdamW(model.parameters(),lr=config["lr"],weight_decay=config['weight_decay'])
        n_epochs = int(config["n_epochs"])
        start = 0
        best_valid_bleu = 0
        logging.info("************ START TRAINING ************")
        for epoch in range(start,n_epochs):
            model.train()
            teacher_force_ratio = config["teacher_force_ratio"]
            epoch_loss = 0
            BLEU_scores = []
            mode  = "train"
            #  ------------------------- BATCH TRAINING ----------------------
            for i, batch in enumerate(train_data_loader):
                loss_train_b, bleu_score,_,_ = run_batch(model,batch,train_data_loader, mode=mode,optimizer=optimizer,
                                                         teacher_force_ratio=teacher_force_ratio,device=config["device"],
                                                         attention_type=config["attention_type"],lambdas=config["lambdas"])
                BLEU_scores += [bleu_score]
                loss_train_b = loss_train_b.item()
                epoch_loss += loss_train_b
                logging.info(f"Loss/{mode}_batch %d --> %.3f BLEU score_batch %.3f" % (i, loss_train_b, bleu_score))
            # ----------------------------------------------------------------

            loss_train = epoch_loss / len(train_data_loader)
            BLEU_score_train = sum(BLEU_scores) / len(BLEU_scores)
            logging.info(f"\nEpoch %d Train Loss --> %.3f BLEU_train score  %.3f\n" % (epoch, loss_train, BLEU_score_train))

            # ----------------------------- EVALUATE --------------------------
            evaluate = True
            if evaluate:
                mode = "val"
                model.eval()
                epoch_loss = 0
                BLEU_scores = []
                for i, batch in enumerate(val_data_loader):
                    loss_val_b, bleu_score, _, _ = run_batch(model, batch, val_data_loader, mode=mode,optimizer=optimizer,
                                                             teacher_force_ratio=teacher_force_ratio,device=config["device"],
                                                             attention_type=config["attention_type"],lambdas=config["lambdas"])
                    BLEU_scores += [bleu_score]
                    BLEU_scores += [bleu_score]
                    loss_val_b = loss_val_b.item()
                    epoch_loss += loss_val_b
                    logging.info(f"Loss/{mode}_batch %d --> %.3f BLEU score_batch %.3f" % (i, loss_val_b, bleu_score))

                loss_val = epoch_loss / len(val_data_loader)
                BLEU_score_val = sum(BLEU_scores) / len(BLEU_scores)
                logging.info("LOSS VAL %.3f BLEU score %.3f" % (loss_val, BLEU_score_val))
                logging.info(f"\nEpoch  %d LOSS VAL %.3f  BLEU_val score %.3f" % (epoch, loss_val, BLEU_score_val))

                # ---------------------------------------------- LOG TO WANDB ----------------------------------------------------

                wandb.log({"loss_val": loss_val,"bleu_val": BLEU_score_val,
                           "loss_train": loss_train,"bleu_train": BLEU_score_train},
                           step=epoch)


                # SAVE BEST MODEL
                if BLEU_score_val >= best_valid_bleu:
                    best_valid_bleu = BLEU_score_val
                    # -------------ARTIFACTS------------
                    model_artifact = wandb.Artifact(f"LSTM_{wandb.run.id}", type="model",
                                                    description="Guided attention for Interpretable Motion Captioning",
                                                    metadata=config)
                    # SAVE THE PYTORCH MODEL TO DIRECTORY
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                    'epoch': epoch, 'val_bleu': BLEU_score_val, 'val_loss': loss_val,
                                    'train_bleu': BLEU_score_train, 'metadata': config},
                               unique_path)
                    model_artifact.add_file(unique_path)
                    # wandb.save(unique_path)
                    # LOG ARTIFACTS
                    run.log_artifact(model_artifact)
                    model_artifact.finalize()

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path",type=str,default=".",help="Path where to save checkpoints")
    parser.add_argument("--dataset_name",type=str,default="kit",choices=["h3D","kit"])
    parser.add_argument("--device",type=str,default="cuda")
    parser.add_argument("--config",type=str,default="./configs/lstm_kit.yaml")
    parser.add_argument("--multiple_references",type=bool,default=False,help="Specify evaluation mode use flattened references or all at one")
    parser.add_argument("--encoder_type",type=str,default="MLP")
    parser.add_argument("--attention_type",type=str,default="relative_bahdanau")
    parser.add_argument("--experience_suffix_name",type=str,default="",help='Run name')
    parser.add_argument("--epoch",type=int,default=200,help='Number of epoch')
    parser.add_argument("--save_checkpoint",type=bool,default=True,help="save checkpoint at each end")

    args = parser.parse_args()


    # with open(args.config,'r') as f:
    #     choices= yaml.load(f,Loader=yaml.Loader)
    # parser.set_defaults(**choices)
    # args = parser.parse_args()

    # args.dataset_name = "h3D"

    project_path = r"C:\Users\karim\PycharmProjects\SemMotion"
    aug_path = r"C:\Users\karim\PycharmProjects\HumanML3D"

    if args.dataset_name=="kit":
        # -------------KIT IMPORTS------------------
        from architectures.LSTM_kit import seq2seq
        from datasets.kit_h3mld import dataset_class
        path_txt = project_path+"\datasets\sentences_corrections.csv"
        path_motion = aug_path+"\kit_with_splits_2023.npz"


    elif args.dataset_name=="h3D":
        # -----------H3D IMPORTS---------------------
        from architectures.LSTM_h3D import seq2seq
        from datasets.h3d_m2t_dataset_ import dataset_class
        path_txt = aug_path+"\sentences_corrections_h3d.csv"
        path_motion = aug_path+"\\all_humanML3D.npz"

    # Login in your wandb account
    wandb.login()

    # Set the config file defining the search space
    args.config = f"./configs/lstm_{args.dataset_name}.yaml"
    with open(args.config,"r") as f:
        sweep_config = yaml.safe_load(f)

    PROJECT_NAME = f"Interpretable_MC_{args.dataset_name}_f"
    sweep_id = wandb.sweep(sweep_config,project=PROJECT_NAME)

    wandb.agent(sweep_id,function=train_m2l)

