import jieba
from torch.utils.data import Dataset


def tokenizer(text):
    """a tokenizer from `jieba`.

    Args:
        text (str): text to be tokenized.

    Returns:
        list: result.
    """
    return list(jieba.cut(text))


def get_word2id(file_paths):
    """Get word2id dict from file_paths.

    Read all files in file_paths, and calculate word2id.

    Args:
        file_paths (str or list): file paths.

    Returns:
        dict: word2id.
    """
    word2id = {"<unkonwn>": 0}
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    for file_path in file_paths:
        with open(file_path, mode="r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                line = line.strip("\n")
                line = line.split("\t")
                for sent in line:
                    sent = tokenizer(sent)
                    for word in sent:
                        if word not in word2id:
                            word2id[word] = len(word2id)
    return word2id


class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, word2id, is_test=False):
        self.texts = list()
        self.is_test = is_test
        with open(file_path, mode="r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                line = line.strip("\n")
                line = line.split("\t")
                head, tail = line[0], line[1]
                rel = "<unkonwn>" if is_test else line[2]
                text = line[2] if is_test else line[3]
                data = {"head": head, "tail": tail, "relation": rel, "text": text}
                data = self._preprocess(data, tokenizer, word2id)
                self.texts.append(data)

    def _preprocess(self, data, tokenizer, word2id):
        """Preprocess a sentence.

        Using tookenizer to split sentence into [word1, word2, word3],
        then use word2id to map word to id, into [index1, index2, index3].

        Args:
            data (dict): including head, tail, relation and text.
            tokenizer (callable): a callable object, such as python function.
            word2id (dict): a diction maps word to id.

        Return:
            dict: preprocessed data.
        """
        for k in data:
            data[k] = tokenizer(data[k])
            for idx, word in enumerate(data[k]):
                if word not in word2id:
                    data[k][idx] = word2id["<unkonwn>"]
                else:
                    data[k][idx] = word2id[word]
        return data

    def __getitem__(self, index):
        return self.texts[index]

    def __len__(self):
        return len(self.texts)
