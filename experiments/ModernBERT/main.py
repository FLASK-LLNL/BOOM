################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC and other
## FLASK Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
from transformers import TrainingArguments, Trainer
from transformers import EvalPrediction
from dataset import BOOMHFDatasetWrapper
from model import ModernBertAdapted
import numpy as np
from tokenizer import SMILESTokenizer
import os
from transformers.models.modernbert import ModernBertConfig

from boom.viz.ParityPlot import OODParityPlot
import argparse


def comma_separated_list(arg):
    return [x.lower() for x in arg.split(",")]


parser = argparse.ArgumentParser(description="ModernBERT training script")
parser.add_argument(
    "properties", type=comma_separated_list, help="List of properties to test"
)
args = parser.parse_args()


def write_csv_file(file_name, preds, labels):

    with open(file_name, "w") as f:
        f.write("predictions,labels\n")
        for pred, label in zip(preds, labels):
            f.write(f"{pred},{label}\n")


def save_predictions(predictions_dict):
    property = predictions_dict["property"]
    id_test = predictions_dict["id_test"]
    ood_test = predictions_dict["ood_test"]

    true_labels = {
        "id": id_test["labels"],
        "ood": ood_test["labels"],
    }
    pred_labels = {
        "id": id_test["predictions"].reshape(-1),
        "ood": ood_test["predictions"].reshape(-1),
    }
    os.makedirs(f"results/{property}", exist_ok=True)
    fig = OODParityPlot(
        true_labels=true_labels,
        pred_labels=pred_labels,
        model_name="ModernBERT",
        target_value=property,
        title=f"{property}",
    )

    fig.savefig(f"results/{property}/parity_plot.png", dpi=300)
    write_csv_file(
        f"results/{property}/id_test.csv",
        id_test["predictions"].reshape(-1),
        id_test["labels"],
    )
    write_csv_file(
        f"results/{property}/ood_test.csv",
        ood_test["predictions"].reshape(-1),
        ood_test["labels"],
    )


def compute_metrics(p: EvalPrediction):
    preds = p.predictions
    labels = p.label_ids
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = preds.squeeze(-1)
    preds = preds.reshape(-1)
    labels = labels.reshape(-1)
    mse = ((preds - labels) ** 2).mean()
    rmse = mse**0.5
    return {
        "mse": mse,
        "rmse": rmse,
        "pearson": np.corrcoef(preds, labels)[0, 1],
        "spearman": np.corrcoef(preds, labels)[0, 1],
    }


def run_experiment(
    train_dataset,
    id_test_dataset,
    ood_test_dataset,
    epochs: int = 10,
    outdir: str = "results",
    out_prefix: str = "results",
    lr: float = 1e-5,
    warmup_ratio: float = 0.1,
    minlr_ratio: float = 0.1,
    vocab_size: int = 1123,
):
    special_token_ids = dict(
        pad_token_id=0,
        eos_token_id=8,
        bos_token_id=7,
        cls_token_id=2,
        sep_token_id=3,
    )
    num_token_id = 6

    head_size = 64  # Default is 64
    heads = 12  # Default is 12
    hidden_size = head_size * heads

    model = ModernBertAdapted(
        ModernBertConfig(
            num_labels=1,
            problem_type="regression",
            classifier_pooling="mean",
            reference_compile=False,
            num_hidden_layers=8,
            num_attention_heads=heads,
            hidden_size=hidden_size,
            intermediate_size=int(hidden_size * 1.5),
            vocab_size=vocab_size,
            **special_token_ids,
        )
    )
    training_args = TrainingArguments(
        output_dir=outdir,
        run_name=outdir,
        logging_dir=f"{out_prefix}/{outdir}",
        eval_strategy="no",
        do_train=True,
        per_device_train_batch_size=32,
        num_train_epochs=epochs,
        max_grad_norm=1,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,  # 0.1, 0.5 worked well
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="no",
        report_to="none",
        save_steps=1,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": lr * minlr_ratio},
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    trainer.save_model(outdir)
    id_predictions, id_labels, id_metrics = trainer.predict(
        id_test_dataset,
        metric_key_prefix="id_test",
    )
    ood_predictions, ood_labels, ood_metrics = trainer.predict(
        ood_test_dataset,
        metric_key_prefix="ood_test",
    )

    return_eval = {
        "id_test": {
            "predictions": id_predictions,
            "labels": id_labels,
            "metrics": id_metrics,
        },
        "ood_test": {
            "predictions": ood_predictions,
            "labels": ood_labels,
            "metrics": ood_metrics,
        },
    }
    return return_eval


def main(props):
    _properties = [
        "density",
        "hof",
        "alpha",
        "cv",
        "homo",
        "lumo",
        "mu",
        "r2",
        "zpve",
        "gap",
    ]

    for property in props:
        if property not in _properties:
            raise ValueError(
                f"Property {property} is not supported. Supported properties are: {_properties}"
            )
    _properties = props

    tokenizer = SMILESTokenizer(
        vocab_file="vocab_smi.txt",
    )
    for property in _properties:
        dataset = BOOMHFDatasetWrapper(property, normalize=True, tokenizer=tokenizer)
        train_dataset = dataset.train_dataset()
        id_test_dataset = dataset.id_test_dataset()
        ood_test_dataset = dataset.ood_test_dataset()

        test_results = run_experiment(
            train_dataset=train_dataset,
            id_test_dataset=id_test_dataset,
            ood_test_dataset=ood_test_dataset,
            epochs=10,
            outdir=f"results/{property}",
            out_prefix="results",
        )
        id_pred = dataset.denormalize(test_results["id_test"]["predictions"])
        id_labels = dataset.denormalize(test_results["id_test"]["labels"])
        ood_pred = dataset.denormalize(test_results["ood_test"]["predictions"])
        ood_labels = dataset.denormalize(test_results["ood_test"]["labels"])

        results_for_metric = {
            "property": property,
            "id_test": {
                "predictions": id_pred,
                "labels": id_labels,
            },
            "ood_test": {
                "predictions": ood_pred,
                "labels": ood_labels,
            },
        }

        save_predictions(results_for_metric)


if __name__ == "__main__":
    assert len(args.properties) > 0, "Please provide a list of properties to test"
    props = args.properties
    main(props)
