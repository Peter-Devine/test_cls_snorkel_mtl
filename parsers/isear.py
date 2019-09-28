import numpy as np
import torch

from snorkel.classification.data import DictDataset

from utils import Classification_Task_Data_Handler

TASK_NAME = "ISEAR"

def parse(jsonl_path, tokenizer, max_data_samples, max_sequence_length):
    Classification_Task_Data_Handler.get_inputs_and_outputs("ISEAR", "D:\Common_Voice\snorkel-superglue\parsers", 60)
    return DictDataset(
        name="SuperGLUE",
        X_dict={
            "sentence1": sent1s,
            "sentence2": sent2s,
            "choice1": choice1s,
            "choice2": choice2s,
            "token1_ids": bert_token1_ids,
        },
        Y_dict={"ISEAR": labels},
        split= "valid" if "val" == file_name else file_name
    )
