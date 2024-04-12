import numpy as np
import pandas as pd
import torch
import torchtext
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import logging
import time
from tempfile import mktemp
import os
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
from nltk.translate import meteor_score
from nltk.corpus.reader.wordnet import  WordNetCorpusReader
from bert_score import score

#from nlgeval import NLGEval,compute_metrics

def read_pred_refs(path,split=True,tokenize=True):
    predictions = []
    references = []
    with open(path, mode="r") as f:
        if tokenize:
            for line in f.readlines():
                line = line.replace("\n", "")
                predictions.append(line.split(',')[0].split(" ")) if split else predictions.append(line.split(',')[0])
                references.append([ref.split(" ") for ref in line.split(',')[1:]]) if split else references.append([ref for ref in line.split(',')[1:]])
        else:
            for line in f.readlines():
                line = line.replace("\n", "")
                predictions.append(line.split(',')[0]) if split else predictions.append(line.split(',')[0])
                references.append([ref for ref in line.split(',')[1:]]) if split else references.append(
                    [ref for ref in line.split(',')[1:]])
    return predictions,references

def calculate_bleu(predictions,references,num_grams=4,single_bleu=False,smooth_method=None):
    bleu_score = torchtext.data.metrics.bleu_score(predictions, references,num_grams, weights=[1/num_grams]*num_grams)\
                                if not single_bleu else sentence_bleu(references,predictions,weights=(1/num_grams,)*num_grams,
                                                                      smoothing_function=smooth_method)
    return bleu_score

def bleu_to_df(predictions,references,smooth_method):
    scores = -np.ones((len(predictions),7),dtype=object)
    for k in range(len(predictions)):
        scores[k,0]=len(predictions[k]) # length prediction column
        scores[k,-1]=" ".join(predictions[k])
        for gr in range(1, 6):
            scores[k,gr] = calculate_bleu( predictions[k],references[k],num_grams=gr,single_bleu=True,smooth_method=smooth_method) #bleu scores columns
    df_bleu =  pd.DataFrame(scores,columns=["Length","bleu_1","bleu_2","bleu_3","bleu_4","bleu_5","prediction"])
    return df_bleu

def bleu_vs_ngram(predictions,references,n_plot=1,fig=None,color="ro--",legend_name="Joint angles",shift=0,single_figure=True):
    BLEU_scores_gram = []
    max_g = [1, 2, 3, 4, 5]
    for gr in max_g:
        bleu_score_g = calculate_bleu(predictions,references,num_grams=gr)
        BLEU_scores_gram.append(bleu_score_g)
    k = 1 if single_figure else n_plot
    fig = plt.figure(figsize=(6.4 * k, 4.8 * k)) if fig is None else fig
    ax1 = fig.axes[0] if len(fig.axes)!=0 else fig.add_subplot(k,1,1)
    ax1.set_ylim(0, 1)
    ax1.set_xticks([0,1,2,3,4,5])
    ax1.plot(max_g, BLEU_scores_gram, color,label=legend_name)
    ax1.set_ylabel("BLEU Score")
    ax1.set_xlabel("Gram number")
    #ax1.legend([legend_name])
    for x, y in zip(max_g, BLEU_scores_gram):
        ax1.text(x-0.12, y+shift, "%.3f" % y,color=color[0])
        print("\033[1;32m BLEU-",x,"--> %.2f"%(y*100))
    fig.tight_layout()
    if n_plot==1 : plt.legend();fig.savefig("save_.png");plt.show()
    else : return  fig
    # write predictions and targets for every batch

def bleu_vs_sentence_length(predictions, references):

    logging.info("plot BLEU Score per n_gram")
    n_plot = 3
    # fig = bleu_vs_ngram(predictions, references,n_plot=n_plot)
    fig,axs = plt.figure().add_subplot(n_plot,1,1)
    logging.info("Plot bleu score per sentence length")
    trg = [ref[0] for ref in references] # TODO use max BLEU ON REFERENCES
    pred = predictions
    ids_sort_trg = sorted(range(len(trg)),key= lambda k: len(trg[k]))
    trg_sorted = [trg[k] for k in ids_sort_trg]
    pred_sorted = [pred[k] for k in ids_sort_trg]
    trg_lens = list(set(len(k) for k in trg_sorted))
    max_trg_len = max(trg_lens)
    pred_per_len = [[] for _ in range(max_trg_len+1)]
    trg_per_len = [[] for _ in range(max_trg_len+1)]
    for pr,tr in zip(pred_sorted,trg_sorted):
        pred_per_len[len(tr)].append(pr)
        trg_per_len[len(tr)].append([tr]) # list of reference we have one we use zero index tr =[["sentences"]]
    bleu_scores_list = []
    for k in trg_lens:
        bleu_score = calculate_bleu(pred_per_len[k],trg_per_len[k],num_grams=4)
        bleu_scores_list.append(bleu_score)
        logging.info("sentences length %d  --> BLEU Score %.3f"%(k,bleu_score))
    ax2 = fig.add_subplot(n_plot,1,2)
    ax2.set_ylim(0,1)
    ax2.plot(trg_lens,bleu_scores_list,"go--")
    ax2.set_ylabel(f" BLEU score n_gram = {4}")
    ax2.set_xlabel(" Sentence length ")


    logging.info(" PLot Histogram ")
    ax3 = fig.add_subplot(n_plot,1,3)
    ax3.hist([len(k) for k in trg_sorted],bins=50)
    ax3.set_title("Number of sentences")
    ax3.set_xlabel("Sentence length")
    fig.tight_layout()
    plt.show()


def semantic_bleu(predictions,references):
    return sum(meteor_score.meteor_score(references[k],predictions[k],wordnet=WordNetCorpusReader )
               for k in range(len(references)))/len(references)

def compute_bert_score(predictions,references,device="cpu"):
    P,R,F1 = score(predictions,references,lang='en',rescale_with_baseline=True,idf=True,device=device)
    _bert_sc = F1.mean().item()
    return _bert_sc
def bleu_rouge_cider_dict(predictions,references):
    with open("hyp_temp.txt",'w') as f:
        f.writelines(predictions)
    for i, refs in enumerate(references):
        with open(f"./temp/ref_{i}.txt","w") as f:
            f.write('\n'.join(refs))
    nlg_eval = NLGEval(
        metrics_to_omit=['METEOR','EmbeddingAverageCosineSimilarity','SkipThoughtCS','VectorExtremaCosineSimilarity','GreedyMatchingScore'])
    return compute_metrics("hyp_temp.txt",[f"./temp/ref_{i}.txt" for i in range(len(references))])
def bleu_rouge_cider_dict_2(predictions,references):
    nlg_eval = NLGEval(
        metrics_to_omit=['METEOR','EmbeddingAverageCosineSimilarity','SkipThoughtCS','VectorExtremaCosineSimilarity','GreedyMatchingScore'])
    ref_list = [list(refs) for refs in zip(*references)]
    cand_list = predictions
    return nlg_eval.compute_metrics(ref_list, cand_list)


def write_first_beam_and_refs(name_file,path,path_refs):
    _, refs = read_pred_refs(path_refs, tokenize=False)
    pred, _ = read_pred_refs(path, tokenize=False)
    #with open(name_file,'w') as f : pass # create file with name_file

    with open(name_file + ".csv", mode="a") as f:
        for p, t in zip(pred, refs):
            f.writelines(("%s" + ",%s" * len(t) + "\n") % (
                        ("".join(p).replace("\n", ""),) + tuple("".join(k).replace("\n", "") for k in t)))


if __name__=="__main__":

    name_file = 'LSTM_h3D_preds_[0, 3]_beam_size_2' #'LSTM_kit_preds_[2, 3]_beam_size_3'#

    path = "/home/karim/PycharmProjects/m2LSpTemp/src/Predictions/"+name_file+'.csv'
    path_refs = "/home/karim/PycharmProjects/m2LSpTemp/src/Predictions/LSTM_h3D_preds_[0, 3].csv"

    write_first_beam_and_refs("./src/temp/"+name_file,path,path_refs)



    # predictions,references = read_pred_refs(path_refs,tokenize=False)
    # bleu_score = calculate_bleu(predictions,references,num_grams=4)
    #
    # #_bert_score = compute_bert_score(predictions,references,device='cuda:0')
    #
    #  #_b_r_c_dict = bleu_rouge_cider_dict(predictions,references)
    #
    # #_b_r_c_dict = bleu_rouge_cider_dict_2(predictions,references)
    #
    # pa, ra= read_pred_refs(path,split=True)
    # bleu_score = calculate_bleu(pa,ra,num_grams=4)
    # df_bleu_ = bleu_to_df(pa,ra,smooth_method=SmoothingFunction().method0)
    #
    #

    nlg_eval = NLGEval(
        metrics_to_omit=['METEOR',
                         'EmbeddingAverageCosineSimilarity' ,
                         'SkipThoughtCS',
                         'VectorExtremaCosineSimilarity',
                         'GreedyMatchingScore']
    )

    # from collections import OrderedDict
    # def evaluate_bleu_rouge_cider(text_loaders, file):
    #     bleu_score_dict = OrderedDict({})
    #     rouge_score_dict = OrderedDict({})
    #     cider_score_dict = OrderedDict({})
    #     # print(text_loaders.keys())
    #     print('========== Evaluating NLG Score ==========')
    #     for text_loader_name, text_loader in text_loaders.items():
    #
    #         ref_list = [list(refs) for refs in zip(*text_loader.dataset.all_caption_list)]
    #         cand_list = text_loader.dataset.generated_texts_list
    #         scores = nlg_eval.compute_metrics(ref_list, cand_list)
    #         bleu_score_dict[text_loader_name] = np.array(
    #             [scores['Bleu_1'], scores['Bleu_2'], scores['Bleu_3'], scores['Bleu_4']])
    #         rouge_score_dict[text_loader_name] = scores['ROUGE_L']
    #         cider_score_dict[text_loader_name] = scores['CIDEr']
    #
    #         line = f'---> [{text_loader_name}] BLEU: '
    #         for i in range(4):
    #             line += '(%d): %.4f ' % (i + 1, scores['Bleu_%d' % (i + 1)])
    #         print(line)
    #         print(line, file=file, flush=True)
    #
    #         print(f'---> [{text_loader_name}] ROUGE_L: {scores["ROUGE_L"]:.4f}')
    #         print(f'---> [{text_loader_name}] ROUGE_L: {scores["ROUGE_L"]:.4f}', file=file, flush=True)
    #         print(f'---> [{text_loader_name}] CIDER: {scores["CIDEr"]:.4f}')
    #         print(f'---> [{text_loader_name}] CIDER: {scores["CIDEr"]:.4f}', file=file, flush=True)
    #     return bleu_score_dict, rouge_score_dict, cider_score_dict
