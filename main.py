import os
import argparse
import multiprocessing as mp
from sys import argv

import torch
import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader

from model import PatchT5, EfficientT5, ET5Baseline, EfficientBERT5
from dataset import DocVQADataset, collate_any, collate_one, collate_bert

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task")
    parser.add_argument("--model", type=int, default=1, help="Choose model by indice")
    parser.add_argument("--freeze", type=int, default=1, help="Freeze weights")
    parser.add_argument("--max_len", type=int, default=32, help="Transformer sequence length.")
    parser.add_argument("--lr", type=float, default=5e-4, help="ADAM Learning Rate.")
    parser.add_argument("--bs", type=float, default=16, help="Batch size.")
    parser.add_argument("--acc_grad", type=float, default=2, help="Accumulate grad Batches.")
    parser.add_argument("--precision", type=int, default=16, help="Precision.")
    parser.add_argument("--max_epochs", type=int, default=20, help="Maximum number of epochs.")
    parser.add_argument("--patience", type=int, default=3, help="How many epochs to wait for improvement in validation.")
    parser.add_argument("--nworkers", type=object, default=mp.cpu_count(), help="Number of workers to use in dataloading.")
    parser.add_argument("--experiment_name", type=str, default="baseline", help="Single word describing experiment.")
    parser.add_argument("--description", type=str, default="No description.", help="Single phrase describing experiment.")
    parser.add_argument("--device", type=int, default=4, help="Specify GPU.")
    hparams = parser.parse_args()

    if hparams.task == "train":
        train_folder = '/data/train'
        val_folder = '/data/val'
        choose_model = [ET5Baseline, EfficientT5, PatchT5, EfficientBERT5]
        
        print('Carregando dataset...')
        train_set = DocVQADataset("train", train_folder)
        val_set = DocVQADataset("val", val_folder)
        train_loader = DataLoader(train_set, batch_size=hparams.bs, shuffle=True, num_workers=4, collate_fn=collate_bert if hparams.model == 3 else collate_any)
        val_loader = DataLoader(val_set, batch_size=1, num_workers=hparams.nworkers, collate_fn=collate_bert if hparams.model == 3 else collate_one)
        
        print('Carregando modelo...')
        model = choose_model[hparams.model](hparams)

        neptune_logger = NeptuneLogger(api_key=os.getenv('NEPTUNE_API_TOKEN'),
                                       project_name="decioaguiar/ia376",
                                       experiment_name=hparams.experiment_name,
                                       tags=[hparams.description],
                                       params=vars(hparams))

        early_stopping = EarlyStopping(monitor="val_word_f1",
                                       patience=hparams.patience,
                                       verbose=False,
                                       mode='max',
                                       )

        dir_path = os.path.join("models", hparams.experiment_name)
        filename = "{epoch}-{val_loss:.2f}-{val_extact_match:.2f}-{val__word_f1:.2f}"
        checkpoint_callback = ModelCheckpoint(prefix=hparams.experiment_name,
                                              dirpath=dir_path,
                                              monitor="val_word_f1",
                                              mode="max")

        callbacks = [checkpoint_callback, early_stopping]
        logger = neptune_logger
        
        trainer = pl.Trainer(
                        gpus=[hparams.device],
                        precision=hparams.precision,
                        max_epochs=hparams.max_epochs,
                        logger=logger,
                        callbacks=callbacks,
                        accumulate_grad_batches=hparams.acc_grad,
                        limit_val_batches=.25,
                    )

        trainer.fit(model, train_loader, val_loader)
        trainer.test(model, val_loader)