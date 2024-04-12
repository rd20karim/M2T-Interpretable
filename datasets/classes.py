from nltk.stem import PorterStemmer
import spacy


nlp = spacy.load("en_core_web_md")


alphas_values = {"root":[1,0,0,0,0,0],
                 "arms":[0,0,1,1,0,0], # uncertainty
                 "legs":[0,0,0,0,1,1], # uncertainty
                 "torso":[0,1,0,0,0,0]}

connection_words = ("is","the","of","his","her","its","on","their")

subjects = ("a","person","human","man")



def stemming(sent):
    ps = PorterStemmer()
    return ps.stem(sent)


trajectory_words = ("circle","circuit","clockwise","anticlockwise","forward","backward")
trajectory_words = tuple(stemming(wh) for wh in trajectory_words )

local_action = ("wave","stumble","kick", "wipe" , "stand" , "throw" , "punch" , "bend" , "lift" ,"bow" , "pick" , "boxing" ,
                "open" , "rotate" , "clean" , "stomp" , "bend" ,"squat" ,"squad", "kneel","handstand","draw")

global_action = ("walk","turn","run","jump","mov","play","jog","climb")

local_action =  tuple(stemming(la) for la in local_action)
global_action = tuple(stemming(ga) for ga in global_action)

alphas = {global_action: "root"}


alphas_local = {("open","waves","wipe","throw","punch","pick","boxing","clean","swipe","catch","handstand","draw"): "arms",
                ("kick","stomp","lift","kneel","squat","squad","stand","stumble","rotate"): "legs",
                ("bend","bow"):"torso",
                }

alphas_local_stem = {tuple(stemming(k) for k in v):value for v,value in zip(alphas_local.keys(),alphas_local.values()) }

def categorize_tokens_new(vocab_list):

    # Define a dictionary for ground truth spatial weights and adaptive gate

    gths_alpha = {}
    gths_beta = {}

    for untoken in vocab_list:
        token = stemming(untoken)

        #   Maximize weight on the ROOT part
        if  token in trajectory_words: #token in global_action or
            gths_alpha[untoken]=[1,0,0,0,0,0]
            gths_beta[untoken]=1

        # Maximize weights on other body parts than root
        elif token in local_action:
            tests = [token in subset for subset in alphas_local_stem]
            i = tests.index(True)
            body_parts = list(alphas_local_stem.values())[i]
            gths_alpha[untoken]= alphas_values[body_parts]
            gths_beta[untoken] = 1

        elif len(untoken)>0 and nlp(untoken)[0].pos_ in ["VERB","ADV"] and untoken not in connection_words :
            gths_alpha[untoken]= [-1]*6 # no supervision
            gths_beta[untoken] = 1

        elif untoken in connection_words or untoken in subjects:
            gths_alpha[untoken]= [-1]*6 # no supervision
            gths_beta[untoken] = 0

        # For remaining uncategorized words, learn freely the corresponding weights
        # Intuitively we expect the model to infer from supervised words of same meaning
        else :
            gths_beta[untoken] = -1
            gths_alpha[untoken]= [-1]*6

    return gths_beta,gths_alpha
