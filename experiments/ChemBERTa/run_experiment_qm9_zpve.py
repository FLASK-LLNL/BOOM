################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC and other
## FLASK Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
from tqdm import tqdm
from flask_ood.datasets.SMILESDataset import *

from flask_ood.viz.ParityPlot import *

from transformers import (
    RobertaConfig,
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    AdamW,
    RobertaModel,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class SmilesDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label,
        }


class regression_model(nn.Module):
    def __init__(self, if_pretrained=False):
        super(regression_model, self).__init__()

        roberta_model = RobertaForSequenceClassification.from_pretrained(
            "seyonec/PubChem10M_SMILES_BPE_450k", num_labels=1
        )
        if if_pretrained:
            self.robert = roberta_model
        else:
            config = roberta_model.config
            self.robert = RobertaForSequenceClassification(config)

    def forward(self, input_ids, attention_mask):
        outputs = self.robert(input_ids, attention_mask)
        class_label_output = outputs.logits
        return class_label_output


def dataframe_wrapper(dataset, normalizing_dataset):
    """
    Wraps the dataset into a pandas dataframe.
    """
    num_samples = len(dataset)
    df = pd.DataFrame(columns=["text", "labels"])
    mean = np.mean([target for _, target in normalizing_dataset])
    std = np.std([target for _, target in normalizing_dataset])
    for i in range(num_samples):
        smiles, target = dataset[i]
        target = (target - mean) / std
        df.loc[i] = [smiles, target]
    return df


def denormalize_target(target, normalizing_dataset):
    """
    Denormalizes the target.
    """
    target_array = np.array(target)
    mean = np.mean([target for _, target in normalizing_dataset])
    std = np.std([target for _, target in normalizing_dataset])
    return target_array * std + mean


def train_model(target, train_dataset, tokenizer, criterion, pre_trained, num_epochs=5):
    """
    Trains a model on the dataset.
    """

    train_df = dataframe_wrapper(train_dataset, train_dataset)
    train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=42)

    train_texts, train_labels = train_df["text"].tolist(), train_df["labels"].tolist()
    valid_texts, valid_labels = valid_df["text"].tolist(), valid_df["labels"].tolist()

    _train_dataset = SmilesDataset(train_texts, train_labels, tokenizer)
    _valid_dataset = SmilesDataset(valid_texts, valid_labels, tokenizer)

    batch_size = 32
    train_dataloader = DataLoader(_train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(_valid_dataset, batch_size=batch_size)

    model = regression_model(if_pretrained=pre_trained)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=1e-5)
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        train_num_samples = 0
        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(torch.float32).to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask).to(device).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            train_num_samples += len(input_ids)
        average_train_loss = total_train_loss / train_num_samples
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {average_train_loss:.4f}"
        )

        model.eval()
        with torch.no_grad():
            total_loss = 0.0
            num_samples = 0
            predictions = []
            _labels = []
            for batch in valid_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(torch.float32).to(device)
                outputs = model(input_ids, attention_mask).to(device).squeeze()
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                num_samples += len(input_ids)

                predictions.extend(outputs.cpu().numpy())
                _labels.extend(labels.cpu().numpy())

            average_loss = total_loss / num_samples
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {average_loss:.4f}"
            )

    l1_score = np.abs(np.array(predictions) - np.array(_labels))
    mean_l1_score = np.mean(l1_score)
    std_l1_score = np.std(l1_score)

    model_path = f"./chemBerta_{target}_{str(pre_trained)}.pth"
    torch.save(model.state_dict(), model_path)
    return model_path, mean_l1_score, std_l1_score


def load_model(model_path):
    model = regression_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the saved model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model


def run_experiment(
    train_dataset, iid_test_dataset, ood_test_dataset, target, pre_trained, num_epochs=5
):
    """
    Runs the experiment.
    """
    tokenizer = RobertaTokenizerFast.from_pretrained(
        "seyonec/SMILES_tokenized_PubChem_shard00_160k"
    )
    criterion = nn.MSELoss()

    model_path, mean_score, std_score = train_model(
        target,
        train_dataset,
        tokenizer=tokenizer,
        criterion=criterion,
        num_epochs=num_epochs,
        pre_trained=pre_trained,
    )

    model = load_model(model_path)

    iid_df = dataframe_wrapper(iid_test_dataset, train_dataset)
    ood_df = dataframe_wrapper(ood_test_dataset, train_dataset)

    ood_test_texts, ood_test_labels = ood_df["text"].tolist(), ood_df["labels"].tolist()
    _ood_test_dataset = SmilesDataset(ood_test_texts, ood_test_labels, tokenizer)
    ood_test_dataloader = DataLoader(_ood_test_dataset, batch_size=32)

    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        num_samples = 0
        all_ood_predictions = []
        all_ood_labels = []
        for batch in ood_test_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(torch.float32).to(device)
            outputs = model(input_ids, attention_mask).to(device).squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            num_samples += len(input_ids)

            all_ood_predictions.extend(outputs.cpu().numpy())
            all_ood_labels.extend(labels.cpu().numpy())

        average_loss = total_loss / num_samples
        print(f"OOD Test Loss: {average_loss:.4f}")

    iid_test_texts, iid_test_labels = iid_df["text"].tolist(), iid_df["labels"].tolist()
    _iid_test_dataset = SmilesDataset(iid_test_texts, iid_test_labels, tokenizer)
    iid_test_dataloader = DataLoader(_iid_test_dataset, batch_size=32)

    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        num_samples = 0
        all_iid_predictions = []
        all_iid_labels = []
        for batch in iid_test_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(torch.float32).to(device)
            outputs = model(input_ids, attention_mask).to(device).squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            num_samples += len(input_ids)

            all_iid_predictions.extend(outputs.cpu().numpy())
            all_iid_labels.extend(labels.cpu().numpy())

        average_loss = total_loss / num_samples
        print(f"IID Test Loss: {average_loss:.4f}")

    iid_real_vals = denormalize_target(all_iid_labels, train_dataset)
    ood_real_vals = denormalize_target(all_ood_labels, train_dataset)

    iid_preds = denormalize_target(all_iid_predictions, train_dataset)
    ood_preds = denormalize_target(all_ood_predictions, train_dataset)

    iid_smiles = iid_df["text"].values
    ood_smiles = ood_df["text"].values

    test_results = {
        "target": target,  # "Density" or "HoF
        "mean_score": mean_score,
        "std_score": std_score,
        "iid_smiles": iid_smiles,
        "pred_iid_vals": iid_preds,
        "real_iid_vals": iid_real_vals,
        "ood_smiles": ood_smiles,
        "pred_ood_vals": ood_preds,
        "real_ood_vals": ood_real_vals,
    }
    return test_results


def process_results(results, plotter, tag):
    """
    Processes the results.
    """

    target = results["target"]
    pred_iid_vals = np.array(results["pred_iid_vals"])
    real_iid_vals = np.array(results["real_iid_vals"])
    pred_ood_vals = np.array(results["pred_ood_vals"])
    real_ood_vals = np.array(results["real_ood_vals"])

    np.save(f"./results/{target}_{tag}_iid_preds.npy", pred_iid_vals)
    np.save(f"./results/{target}_{tag}_iid_real.npy", real_iid_vals)
    np.save(f"./results/{target}_{tag}_ood_preds.npy", pred_ood_vals)
    np.save(f"./results/{target}_{tag}_ood_real.npy", real_ood_vals)

    true_labels = {
        "iid": real_iid_vals,
        "ood": real_ood_vals,
    }

    fake_labels = {
        "iid": pred_iid_vals,
        "ood": pred_ood_vals,
    }

    fig = plotter(true_labels, fake_labels, title=target + tag, model_name="ChemBERTa")
    fig.savefig(f"./results/{target}_{tag}_parity_plot.png")


def main():
    train_dataset = TrainQM9_zpveDataset()
    iid_test_dataset = IIDQM9_zpveDataset()
    ood_test_dataset = OODQM9_zpveDataset()
    for pre_trained in [True, False]:

        results = run_experiment(
            train_dataset,
            iid_test_dataset,
            ood_test_dataset,
            "QM9_zpve",
            num_epochs=5,
            pre_trained=pre_trained,
        )

        process_results(
            results, ZpveOODParityPlot, f"pre_trained={str(pre_trained)}"
        )

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    main()
