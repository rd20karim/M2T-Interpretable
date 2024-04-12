from pathlib import Path
import sys
from bert_score import score
import numpy as np
from nlgeval.pycocoevalcap.cider.cider import Cider
from nlgeval.pycocoevalcap.rouge.rouge import Rouge
from nlgeval.pycocoevalcap.bleu.bleu import Bleu
from tqdm import tqdm
from nltk.tokenize.nist import NISTTokenizer
from nltk.translate.bleu_score import corpus_bleu
import torchtext
from bleu_from_csv import read_pred_refs



if __name__ == "__main__":
    # Path to csv Predictions
    input_path = Path(sys.argv[1])
    table_text = (
        "\\begin{tabular}{lcccccc}\n"
        + "\\toprule\n"
        + "File & Dataset & \\lambda_{spat} & \\lambda_{adapt} & Bleu@1 & Bleu@4 & ROUGE_L & CIDEr & BERTScore \\\\ \\midrule \n"
    )
    for file in tqdm(list(input_path.glob("*.csv"))):
        print(file.stem)
        filename_parts = file.stem.split("_")
        model = filename_parts[0]
        dataset = filename_parts[1]
        params = [param.strip() for param in filename_parts[3][1:-1].split(",")]
        lambda_spat = params[0]
        lambda_adapt = params[1]

        with open(file, "r") as f:
            lines = f.readlines()
        refs = []
        preds = []


        for line in lines:
            rp = line.replace("\n", "").split(",")
            local_refs = rp[1:]
            pred = rp[0]
            refs.append(local_refs)
            preds.append(pred)
        print(len(preds), len(refs))

        refs_nlgeval = {}
        preds_nlgeval = {}
        refs_nltk = []
        preds_nltk = []

        for index, loc_refs in enumerate(refs):
            refs_nlgeval[index] = loc_refs
            preds_nlgeval[index] = [preds[index]]
            refs_nltk.append([ref.split(' ') for ref in loc_refs])
            preds_nltk.append(preds[index].split(' '))
        scorers = {
            "CIDEr": Cider(),
            "ROUGE_L": Rouge(),
            "Bleu_4": Bleu(n=4),
            "Bleu_1": Bleu(n=1),
        }

        scores = {
            scorer: method.compute_score(refs_nlgeval, preds_nlgeval)[0]
            for scorer, method in scorers.items()
        }

        scores["Bleu_4"] = scores["Bleu_4"][-1]
        scores["Bleu_1"] = scores["Bleu_1"][-1]

        P, R, F1 = score(
            preds, refs, lang="en", verbose=True, idf=True, rescale_with_baseline=True
        )
        scores["BERTScore"] = F1.mean().item()

        nltk_bleu_scores = corpus_bleu(refs_nltk, preds_nltk)
        tochtext_bleu = torchtext.data.metrics.bleu_score(preds_nltk, refs_nltk)

        scores["Bleu4_NLTK"] = nltk_bleu_scores
        scores["Bleu4_Torchtext"] = tochtext_bleu

        local_text = f"{file.stem} & {dataset} & {lambda_spat} & {lambda_adapt} & {scores['Bleu_1']* 100:.1f} & {scores['Bleu_4']* 100:.1f}  & {scores['ROUGE_L'] * 100:.1f} & {scores['CIDEr'] * 100:.1f} & {scores['BERTScore'] * 100:.1f} \\\\ \n"

        print(scores)
        table_text += local_text

    table_text += "\\bottomrule \n \\end{tabular}"

    print(table_text)
    with open(f"vs_beam_results.tex", "w") as f:
        f.write(table_text)


