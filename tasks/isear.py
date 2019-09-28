import sys
from functools import partial

from modules.bert_module import BertLastCLSModule, BertModule
from torch import nn
import torch.nn.functional as F

from snorkel.analysis import Scorer
from snorkel.classification import Task, Operation


from . import utils

sys.path.append("..")  # Adds higher directory to python modules path.


TASK_NAME = "ISEAR"


def build_task(bert_model_name, last_hidden_dropout_prob=0.0):

    bert_module = BertModule(bert_model_name)
    bert_output_dim = 768 if "base" in bert_model_name else 1024

    task_cardinality = 7

    metrics = ["accuracy"]

    custom_metric_funcs = {}

    loss_fn = F.cross_entropy #partial(utils.ce_loss, f"{TASK_NAME}_pred_head")
    output_fn = partial(F.softmax, dim=1)

    task = Task(
        name=TASK_NAME,
        module_pool=nn.ModuleDict(
            {
                "bert_module": bert_module,
                "dropout": BertLastCLSModule(
                    dropout_prob=last_hidden_dropout_prob
                ),
                "linear_module": nn.Linear(bert_output_dim, task_cardinality)
            }
        ),
        op_sequence=[
            Operation(
                name="BERT_LAYER",
                module_name="bert_module",
                inputs=[("_input_", "token_ids"),
                       ("_input_", "token_type_ids"),
                       ("_input_", "attention_mask")],
            ),
            Operation(
                name="DROPOUT",
                module_name=f"dropout",
                inputs=[("BERT_LAYER", 1)],
            ),
            Operation(
                name="linear_layer",
                module_name="linear_module",
                inputs=[("DROPOUT", 0)],
            ),
        ],
        loss_func=loss_fn,
        output_func=output_fn,
        scorer=Scorer(metrics=metrics, custom_metric_funcs=custom_metric_funcs),
    )

    return task
