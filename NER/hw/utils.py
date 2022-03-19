import os
import torch
import json


def save_obj(obj, save_path):

    with open(save_path, mode="w") as f:
        json.dump(obj, f)


def get_word_dict(data_dir, limit=500):
    frequency = dict()
    target = ["train_data.json", "valid_data.json", "test_data.json"]
    for tar in target:
        file_path = os.path.join(data_dir, tar)
        if os.path.exists(file_path):
            with open(file_path, mode="r") as f:
                data = json.load(f)
                for items in data:
                    for item in items:
                        if item["text"] not in frequency:
                            frequency[item["text"]] = 0
                        else:
                            frequency[item["text"]] += 1
    ans = [[k, v] for k, v in frequency.items()]
    ans.sort(key=lambda x: -x[1])
    ans = [k for k, _ in ans]
    return {val: pos for pos, val in enumerate(ans[:limit])}


def get_tensor_onehot(word, word_dict, out_dtype=torch.float32):
    tensor = [0 for _ in range(len(word_dict))]
    if word in word_dict:
        tensor[word_dict[word]] = 1
    return torch.tensor(tensor, dtype=out_dtype)


def tag_sentence(sentence):
    # tag
    return sentence


def split_data(data_dir, save_dir):
    file_exists = os.path.isfile(data_dir)
    if not file_exists:
        raise RuntimeError("data file not exists!")
    train_data = list()
    valid_data = list()
    test_data = list()
    with open(data_dir, mode="r", encoding="ISO-8859-1") as f:
        for id, line in enumerate(f):
            line = line.strip().split("  ")
            if len(line) <= 1:
                continue
            line = [word.split("/") for word in line]
            line = [{"text": word[0], "tag": word[1]} for word in line]
            date = int(line[0]["text"].split("-")[0])
            line = tag_sentence(line[1:])
            if 19980101 <= date <= 19980120:
                train_data.append(line)
            elif 19980121 <= date <= 19980125:
                valid_data.append(line)
            elif 19980126 <= date <= 19980131:
                test_data.append(line)
            else:
                break
    save_obj(train_data, os.path.join(save_dir, "train_data.json"))
    save_obj(valid_data, os.path.join(save_dir, "valid_data.json"))
    save_obj(test_data, os.path.join(save_dir, "test_data.json"))


if __name__ == "__main__":
    ans = get_word_dict(limit=100, data_dir="/home/zengqunhong/torch-learning/NER/data")
    print(get_tensor_onehot("的", ans))
