import spacy
#import contextualSpellCheck
#from spellchecker import SpellChecker
import logging

logging.basicConfig(
    level=logging.INFO,
    format="\t\033[1;32m[%(levelname)s] %(message)s",#%(asctime)s
    handlers=[
        logging.FileHandler("vocabulary_py_out.txt"),
        logging.StreamHandler()
    ]
)


class vocabulary():
    def __init__(self, sentences, context_correction=False, correct_tokens=False,ask_user=False):

        self.sentences = [ref.lower() for ref in sentences] # list of sentences
        self.points = ["-", "=", "<", ">", "\n", "/", '.', ',', '(', ')',
                       ':', ';',":","..",'!',"\r", '?', '"']

        self.correct_tokens = correct_tokens
        self.context_correction = context_correction
        self.corrected_sentences = []
        # assert self.correct_tokens and not correct_tokens
        self.spacy_eng = spacy.load("en_core_web_sm")
        if self.context_correction:
            # Warning time expansive
            #contextualSpellCheck.add_to_pipe(self.spacy_eng)
            self.context_sentence_correction()
        elif self.correct_tokens:
            pass
        else:
            self.corrected_sentences = self.sentences[:] # independent copy

    def context_sentence_correction(self):
        for desc in self.sentences:
            input_desc = str(desc)
            corrected_desc = self.spacy_eng(input_desc)._.outcome_spellCheck
            if len(corrected_desc) >= 1:
                self.corrected_sentences.append(desc)
                # for user examination
                with open("sentence_correction.txt", mode="a") as fa:
                    fa.write("%s \t %s\n" % (desc, corrected_desc))
            # no correction made conserve the original description
            else:
                self.corrected_sentences.append(desc)

    def build_vocabulary(self, min_freq=1):
        self.token_to_idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.idx_to_token = {}
        self.token_freq = {}

        for iak,phrase in enumerate(self.corrected_sentences):
            tokens = self.language_tokenizer(phrase)
            for token in tokens:
                token = str(token)
                # Initialization
                if token not in self.token_freq: self.token_freq[token] = 1
                # Token counter
                else: self.token_freq[token] += 1

        self.low_frq_word = {}
        sp_token = ["<pad>", "<sos>", "<eos>", "<unk>"]

        self.vocab = sp_token + list(self.token_freq)
        self.vocab_size = len(self.vocab)
        sp_count = len(sp_token) # count special tokens
        # sanity check
        assert self.vocab_size ==len(self.token_freq)+sp_count

        # with open("./dataset_vocab.txt", mode='w',encoding='utf-8') as fw:
        #     for key, value in self.token_freq.items():
        #         fw.write("%s:%s\n" % (key, value))

        # word2idx idx2word mapping
        ##!!! IMPORTANT ORDERED TOKENS !!!##
        #sorted() --> alphanumeric order
        ord = sorted(self.token_freq)
        id = 0
        for idx, token in enumerate(ord):
            if self.token_freq[token] < min_freq:
                self.token_to_idx[token] = self.token_to_idx["<unk>"]
                self.low_frq_word[token] = self.token_freq[token]
            else:
                self.token_to_idx[token] = len(sp_token)+ id
                id += 1

        self.idx_to_token = {idx:token for token,idx in self.token_to_idx.items() if token!="<unk>"}
        self.idx_to_token[self.token_to_idx["<unk>"]] = "<unk>"

        # Note that self.idx_to_token don't have the same length as self.token_to_idx when  min_freq !=1

        self.vocab_size_unk  = len(self.idx_to_token)
        logging.info(f"The vocab size is {self.vocab_size}"
                     f" with minimum frequency of {min_freq} it becomes --> {self.vocab_size_unk} tokens")
        # Verify if we have successive int indexes of tokens
        idxs = set(self.token_to_idx.values())
        try:
            assert sum(idxs) == self.vocab_size_unk*(self.vocab_size_unk-1)//2
        except AssertionError:
            logging.info(f"token indexes are not successive or vocab size problem"
                          f" {self.vocab_size_unk} != {len(idxs)} ? "
                          f" and idxs={idxs}")
            raise AssertionError

    def numericalize(self,sentence):
        "Return list of associated integers to the sentence using the token_to_idx dictionary"
        return  [self.token_to_idx[token] for token in self.language_tokenizer(sentence)]

    def decode_numeric_sentence(self,list_int,remove_sos_eos=False,ignore_pad = False):
        " Return human read-able sentence"
        if remove_sos_eos:
            sos_id = 1 if self.token_to_idx["<sos>"] in list_int else 0 # exclude sos from the targets <-- some predicted sentences does not include sos or eos
            try :
                eos_id = list_int.index(self.token_to_idx["<eos>"])
                return " ".join([self.idx_to_token[idx] for idx in list_int[sos_id:eos_id+1]
                                 if idx != self.token_to_idx["<pad>"] or not ignore_pad])
            except ValueError :
                # WHEN THE EOS TOKEN IS NOT GENERATE IN THE PREDICTION
                return " ".join([self.idx_to_token[idx] for idx in list_int[sos_id:]
                                 if idx != self.token_to_idx["<pad>"] or not ignore_pad])

        else:
            try:
                pad_id = list_int.index(self.token_to_idx["<pad>"])
                return " ".join([self.idx_to_token[idx] for idx in list_int[:pad_id] if
                                 idx != self.token_to_idx["<pad>"] or not ignore_pad])
            except ValueError:
                return " ".join([self.idx_to_token[idx] for idx in list_int
                                 if idx!=self.token_to_idx["<pad>"] or not ignore_pad])

    def language_tokenizer(self, sentence):
        sentence = str(sentence) # numpy str --> str
        return [str(token) for token in self.spacy_eng.tokenizer(sentence)]