import logging
import os

from snorkel.classification.data import DictDataLoader

from utils import Classification_Task_Data_Handler
from snorkel.classification.data import DictDataset

import torch

logger = logging.getLogger(__name__)

def get_dataloaders(
    data_dir,
    task_name="MultiRC",
    splits=["train", "dev", "test"],
    max_data_samples=None,
    max_sequence_length=256,
    tokenizer_name="bert-base-uncased",
    batch_size=16,
):
    """Load data and return dataloaders"""

    dataloaders = []

    split_datasets, output_label_to_int_dict = Classification_Task_Data_Handler.get_inputs_and_outputs(task_name, data_dir, seq_len=max_sequence_length, language_model_type=tokenizer_name)

    for split in splits:
        input_tensor = torch.tensor(split_datasets[split]["input"], dtype=torch.long)

        token_ids = input_tensor[:,0,:]
        token_type_ids = input_tensor[:,1,:]
        attention_mask = input_tensor[:,2,:]

        output_tensor = torch.tensor(split_datasets[split]["output"], dtype=torch.long)

        dataset = DictDataset(
            name=task_name,
            X_dict={
                "token_ids": token_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
            },
            Y_dict={"ISEAR": output_tensor},
            split= "valid" if split == "dev" else split
        )

        dataloader = DictDataLoader(
            #task_to_label_dict={task_name: "labels"},
            dataset=dataset,
            #split=split,
            batch_size=batch_size,
            shuffle=(split == "train"),
        )
        dataloaders.append(dataloader)

        logger.info(f"Loaded {split} for {task_name} with {len(dataset)} samples.")

    return dataloaders
