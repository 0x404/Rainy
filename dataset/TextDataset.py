import os
import jieba
import json
from torch.utils.data import Dataset


def tokenizer(text):
    return list(jieba.cut(text))


def get_word2id(file_paths):
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


def get_max_length(file_paths):
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    max_length = 0
    for file_path in file_paths:
        with open(file_path, mode="r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                line = line.strip("\n")
                line = line.split("\t")
                for sent in line:
                    sent = tokenizer(sent)
                    max_length = max(max_length, len(sent))
    return max_length


def get_relation(json_path):
    with open(json_path, mode="r", encoding="utf-8") as f:
        relation = json.load(f)
    return relation


class TextDataset(Dataset):
    def __init__(
        self, file_path, tokenizer, word2id, relation, max_sent, is_test=False
    ):
        self.texts = []
        self.labels = []
        self.is_test = is_test
        self.word2id = word2id
        self.relation = relation
        self.max_sent = max_sent
        with open(file_path, mode="r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                line = line.strip("\n")
                line = line.split("\t")
                head, tail = line[0], line[1]
                rel = "<unkonwn>" if is_test else line[2]
                text = line[2] if is_test else line[3]
                if head.endswith("placeholder") or tail.endswith("placeholder"):
                    continue
                data = {"head": head, "tail": tail, "relation": rel, "text": text}
                data = self._preprocess(data, tokenizer, self.word2id, self.relation)
                data = self._position(data)
                if not data:
                    continue
                data.pop("head")
                data.pop("tail")
                self.labels.append(data["relation"])
                data.pop("relation")
                self.texts.append(data)

    def _position(self, data):
        pos1 = []
        pos2 = []
        head1 = data["head"][0]
        head2 = data["head"][-1]
        tail1 = data["tail"][0]
        tail2 = data["tail"][-1]
        text = data["text"]
        try:
            head = (text.index(head1) + text.index(head2)) // 2
            tail = (text.index(tail1) + text.index(tail2)) // 2
            for i, _ in enumerate(text):
                pos1.append(abs(i - head))
                pos2.append(abs(i - tail))
            data["pos1"] = pos1
            data["pos2"] = pos2
            return data
        except Exception:
            return False

    def _preprocess(self, data, tokenizer, word2id, relation):
        for k in data:
            if k == "relation":
                if data[k] != "<unkonwn>":
                    data[k] = relation[1][data[k]]
            else:
                data[k] = tokenizer(data[k])
                for idx, word in enumerate(data[k]):
                    if word not in word2id:
                        data[k][idx] = word2id["<unkonwn>"]
                    else:
                        data[k][idx] = word2id[word]
        while len(data["text"]) < self.max_sent:
            data["text"].append(word2id["<unkonwn>"])
        return data

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]

    def __len__(self):
        return len(self.texts)


if __name__ == "__main__":
    x = get_max_length(
        [
            os.path.join("remote-data", "exp3-data", "data_train.txt"),
            os.path.join("remote-data", "exp3-data", "data_val.txt"),
            os.path.join("remote-data", "exp3-data", "test_exp3.txt"),
        ]
    )
    print(x)
