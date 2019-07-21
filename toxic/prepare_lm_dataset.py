import spacy
import pandas as pd
from tqdm import tqdm

NLP = spacy.load("en_core_web_sm", disable=["ner", "tagger"])


def split_sentences(text):
    doc = NLP(text)
    return [sent.text.strip() for sent in doc.sents]


def main():
    df_train = pd.read_csv("data/train.csv")[["comment_text"]].sample(200000)
    with open("data/lm_dataset.txt", "w") as fout:
        for doc in tqdm(NLP.pipe(df_train.comment_text, n_threads=4), total=df_train.shape[0]):
            flag = False
            sentences = [sent.text.strip() for sent in doc.sents]
            for sent in sentences:
                if len(sent) > 1:
                    fout.write(sent + "\n")
                    flag = True
            if flag:
                fout.write("\n")


if __name__ == "__main__":
    main()
