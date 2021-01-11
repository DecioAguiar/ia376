import torch
import torchvision
from torch import nn
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl

from efficientnet_pytorch import EfficientNet
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer
from transformers import AdamW
from metrics import compute_exact, compute_f1
from torch.nn import CrossEntropyLoss
from transformers import DistilBertModel

tokenizer = T5Tokenizer.from_pretrained('t5-base')

class PatchT5(pl.LightningModule):

    def __init__(self, params):
        super().__init__()
        
        self.decoder = T5ForConditionalGeneration.from_pretrained('t5-base')
        
        # Patch convolution. Basically tranform the 2D image into 16x16 patches with 2048 dimensions.
        self.patch = nn.Conv2d(in_channels=3, out_channels=self.decoder.config.d_ff, kernel_size=16, stride=16)
        
        # Convert patches from 2048 to 512 dimensions.
        self.linear = nn.Linear(self.decoder.config.d_ff, self.decoder.config.d_model)
        
        self.max_len = params.max_len
        self.learning_rate = params.lr
    
    def _feature_fusion(self, img, question_ids):
        img_embeds = self._embeds_forward(img)
        question_embeds = self.decoder.encoder.embed_tokens(question_ids)
        inputs_embeds = torch.cat((img_embeds[:,:480,:], question_embeds[:,:32,:]), 1)
        return inputs_embeds

    def _embeds_forward(self, img):

        features = img
        
        # Compute patches to feed the transformer.
        features = self.patch(features)
        
        # Reshape the output to the 2D format of the tranformer.
        inputs = features \
            .permute(0, 2, 3, 1) \
            .reshape(features.shape[0], -1, self.decoder.config.d_ff)

        # Add a non-linearity (this seems to speed up training a little bit).
        inputs = F.relu(inputs)
        
        # Resize to transformer dimension.
        inputs = self.linear(inputs)
        
        return inputs

    def forward(self, img=None, question_ids=None, inputs_embeds=None, decoder_input_ids=None, labels=None):

        # Pass efficientnet hidden states as embeddings for the transformer encoder input.
        inputs_embeds = self._feature_fusion(img, question_ids) if inputs_embeds is None else inputs_embeds

        return self.decoder(
            inputs_embeds=inputs_embeds,
            decoder_input_ids=decoder_input_ids, 
            labels=labels,
            return_dict=True
        )

    def generate(self, img, question_ids):

        # We need to implement our own generate loop as transformers doesn't accept 
        # precomputed embeddings on the generate method.
        # Issue: https://github.com/huggingface/transformers/issues/7626
        
        # Precompute embeddings to speedup generation as they don't change.
        inputs_embeds = self._feature_fusion(img, question_ids)

        decoder_input_ids = torch.full(
            (1, 1), self.decoder.config.decoder_start_token_id, dtype=torch.long, device=img.device
        )

        for i in range(self.max_len):
            with torch.no_grad():
                output = self.forward(decoder_input_ids=decoder_input_ids, 
                                      inputs_embeds=inputs_embeds)
            
            logits = output[0]
            next_token_logits = logits[:, -1, :]
            next_token_id = next_token_logits.argmax(1).unsqueeze(-1)
            decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=-1).to(img.device)

            if torch.eq(next_token_id[:, -1], self.decoder.config.eos_token_id).all():
                break

        return decoder_input_ids

    def training_step(self, batch, batch_idx):
        img, question_ids, answer_ids, answers = batch
        output = self(img=img, 
                      question_ids=question_ids, 
                      labels=answer_ids)
        return output[0]

    def validation_step(self, batch, batch_idx):
        img, question_ids, answer_ids, answers = batch
        tokens = [self.generate(im.view((1,) + im.shape), question_ids)[0].cpu() for im in img]
        
        return (tokens, answers)

    def validation_epoch_end(self, validation_step_outputs):
        validation_step_outputs = list(validation_step_outputs)
    
        tokens_batch = [t for out in validation_step_outputs for t in out[0]]
        reference_batch = [r for out in validation_step_outputs for r in out[1]]
        
        generated_batch = tokenizer.batch_decode(tokens_batch)

        em = np.average([compute_exact(g, r) for g, r in zip(generated_batch, reference_batch)])
        f1 = np.average([compute_f1(g, r) for g, r in zip(generated_batch, reference_batch)])

        self.log("val_exact_match", em, prog_bar=True)
        self.log("val_word_f1", f1, prog_bar=True)

    def test_step(self, batch, batch_idx):
        img, question_ids, answer_ids, answers = batch
        tokens = [self.generate(im.view((1,) + im.shape), question_ids)[0].cpu() for im in img]
        
        return (tokens, answers)

    def test_epoch_end(self, validation_step_outputs):
        validation_step_outputs = list(validation_step_outputs)
    
        tokens_batch = [t for out in validation_step_outputs for t in out[0]]
        reference_batch = [r for out in validation_step_outputs for r in out[1]]
        
        generated_batch = tokenizer.batch_decode(tokens_batch)
        
        em = np.average([compute_exact(g, r) for g, r in zip(generated_batch, reference_batch)])
        f1 = np.average([compute_f1(g, r) for g, r in zip(generated_batch, reference_batch)])

        self.log("test_exact_match", em, prog_bar=True)
        self.log("test_word_f1", f1, prog_bar=True)
        
        return {
            "test_exact_match": em,
            "test_word_f1": f1,
        }

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)


class EfficientT5(pl.LightningModule):

    def __init__(self, params):
        super().__init__()

        self.decoder = T5ForConditionalGeneration.from_pretrained('t5-base')

        self.encoder = EfficientNet.from_pretrained('efficientnet-b7', advprop=True, include_top=False)
        if params.freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Convert features from 2560x7x7 to 512x768 dimensions.
        self.bridge = nn.Conv2d(in_channels=2560, out_channels=self.decoder.config.d_model, kernel_size=1)
        
        self.max_len = params.max_len
        self.learning_rate = params.lr

    def _feature_fusion(self, img, question_ids):
        img_embeds = self._embeds_forward(img)
        question_embeds = self.decoder.encoder.embed_tokens(question_ids)
        inputs_embeds = torch.cat((img_embeds[:,:480,:], question_embeds[:,:32,:]), 1)
        return inputs_embeds

    def _embeds_forward(self, img):

        features = self.encoder.extract_features(img)

        features = self.bridge(features)

        inputs = features \
            .permute(0, 2, 3, 1) \
            .reshape(features.shape[0], -1, self.decoder.config.d_model)
        
        return inputs

    def forward(self, img=None, question_ids=None, inputs_embeds=None, decoder_input_ids=None, labels=None):

        # Pass efficientnet hidden states as embeddings for the transformer encoder input.
        inputs_embeds = self._feature_fusion(img, question_ids) if inputs_embeds is None else inputs_embeds

        return self.decoder(
            inputs_embeds=inputs_embeds,
            decoder_input_ids=decoder_input_ids, 
            labels=labels,
            return_dict=True
        )

    def generate(self, img, question_ids):

        # We need to implement our own generate loop as transformers doesn't accept 
        # precomputed embeddings on the generate method.
        # Issue: https://github.com/huggingface/transformers/issues/7626
        
        # Precompute embeddings to speedup generation as they don't change.
        inputs_embeds = self._feature_fusion(img, question_ids)

        decoder_input_ids = torch.full(
            (1, 1), self.decoder.config.decoder_start_token_id, dtype=torch.long, device=img.device
        )

        for i in range(self.max_len):
            with torch.no_grad():
                output = self.forward(decoder_input_ids=decoder_input_ids, 
                                      inputs_embeds=inputs_embeds)
            
            logits = output[0]
            next_token_logits = logits[:, -1, :]
            next_token_id = next_token_logits.argmax(1).unsqueeze(-1)
            decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=-1).to(img.device)

            if torch.eq(next_token_id[:, -1], self.decoder.config.eos_token_id).all():
                break

        return decoder_input_ids

    def training_step(self, batch, batch_idx):
        img, question_ids, answer_ids, answers = batch
        output = self(img=img, 
                      question_ids=question_ids, 
                      labels=answer_ids)
        self.log("train_loss", output[0], prog_bar=True)
        
        return output[0]

    def validation_step(self, batch, batch_idx):
        img, question_ids, answer_ids, answers = batch
        tokens = [self.generate(im.view((1,) + im.shape), question_ids)[0].cpu() for im in img]
        return (tokens, answers)

    def validation_epoch_end(self, validation_step_outputs):
        validation_step_outputs = list(validation_step_outputs)
    
        tokens_batch = [t for out in validation_step_outputs for t in out[0]]
        reference_batch = [r for out in validation_step_outputs for r in out[1]]
        
        generated_batch = tokenizer.batch_decode(tokens_batch)

        em = np.average([compute_exact(g, r) for g, r in zip(generated_batch, reference_batch)])
        f1 = np.average([compute_f1(g, r) for g, r in zip(generated_batch, reference_batch)])

        self.log("val_exact_match", em, prog_bar=True)
        self.log("val_word_f1", f1, prog_bar=True)

    def test_step(self, batch, batch_idx):
        img, question_ids, answer_ids, answers = batch
        tokens = [self.generate(im.view((1,) + im.shape), question_ids)[0].cpu() for im in img]
        
        return (tokens, answers)
    
    def test_epoch_end(self, validation_step_outputs):
        validation_step_outputs = list(validation_step_outputs)
    
        tokens_batch = [t for out in validation_step_outputs for t in out[0]]
        reference_batch = [r for out in validation_step_outputs for r in out[1]]
        
        generated_batch = tokenizer.batch_decode(tokens_batch)
        
        em = np.average([compute_exact(g, r) for g, r in zip(generated_batch, reference_batch)])
        f1 = np.average([compute_f1(g, r) for g, r in zip(generated_batch, reference_batch)])

        self.log("test_exact_match", em, prog_bar=True)
        self.log("test_word_f1", f1, prog_bar=True)
        
        return {
            "test_exact_match": em,
            "test_word_f1": f1,
        }
        
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)


class ET5Baseline(pl.LightningModule):

    def __init__(self, params):
        super().__init__()

        self.decoder = T5ForConditionalGeneration.from_pretrained('t5-small')

        self.encoder = EfficientNet.from_pretrained('efficientnet-b0', advprop=True, include_top=False)
        if params.freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.bridge = nn.Conv2d(in_channels=1280, out_channels=self.decoder.config.d_model, kernel_size=1)
        
        self.max_len = params.max_len
        self.learning_rate = params.lr

    def _feature_fusion(self, img, question_ids):
        img_embeds = self._embeds_forward(img)
        question_embeds = self.decoder.encoder.embed_tokens(question_ids)
        inputs_embeds = torch.cat((img_embeds[:,:480,:], question_embeds[:,:32,:]), 1)
        return inputs_embeds

    def _embeds_forward(self, img):

        features = self.encoder.extract_features(img)

        features = self.bridge(features)

        inputs = features \
            .permute(0, 2, 3, 1) \
            .reshape(features.shape[0], -1, self.decoder.config.d_model)
        
        return inputs

    def forward(self, img=None, question_ids=None, inputs_embeds=None, decoder_input_ids=None, labels=None):

        # Pass efficientnet hidden states as embeddings for the transformer encoder input.
        inputs_embeds = self._feature_fusion(img, question_ids) if inputs_embeds is None else inputs_embeds

        return self.decoder(
            inputs_embeds=inputs_embeds,
            decoder_input_ids=decoder_input_ids, 
            labels=labels,
            return_dict=True
        )

    def generate(self, img, question_ids):

        # We need to implement our own generate loop as transformers doesn't accept 
        # precomputed embeddings on the generate method.
        # Issue: https://github.com/huggingface/transformers/issues/7626
        
        # Precompute embeddings to speedup generation as they don't change.
        inputs_embeds = self._feature_fusion(img, ocr_ids)

        decoder_input_ids = torch.full(
            (1, 1), self.decoder.config.decoder_start_token_id, dtype=torch.long, device=img.device
        )

        for i in range(self.max_len):
            with torch.no_grad():
                output = self.forward(decoder_input_ids=decoder_input_ids, 
                                      inputs_embeds=inputs_embeds)
            
            logits = output[0]
            next_token_logits = logits[:, -1, :]
            next_token_id = next_token_logits.argmax(1).unsqueeze(-1)
            decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=-1).to(img.device)

            if torch.eq(next_token_id[:, -1], self.decoder.config.eos_token_id).all():
                break

        return decoder_input_ids

    def training_step(self, batch, batch_idx):
        img, question_ids, answer_ids, answers = batch
        output = self(img=img, 
                      question_ids=question_ids, 
                      labels=answer_ids)
        self.log("train_loss", output[0], prog_bar=True)
        
        return output[0]

    def validation_step(self, batch, batch_idx):
        img, question_ids, answer_ids, answers = batch
        tokens = [self.generate(im.view((1,) + im.shape), question_ids)[0].cpu() for im in img]
        return (tokens, answers)

    def validation_epoch_end(self, validation_step_outputs):
        validation_step_outputs = list(validation_step_outputs)
    
        tokens_batch = [t for out in validation_step_outputs for t in out[0]]
        reference_batch = [r for out in validation_step_outputs for r in out[1]]
        
        generated_batch = tokenizer.batch_decode(tokens_batch)

        em = np.average([compute_exact(g, r) for g, r in zip(generated_batch, reference_batch)])
        f1 = np.average([compute_f1(g, r) for g, r in zip(generated_batch, reference_batch)])

        self.log("val_exact_match", em, prog_bar=True)
        self.log("val_word_f1", f1, prog_bar=True)

    def test_step(self, batch, batch_idx):
        img, question_ids, answer_ids, answers = batch
        tokens = [self.generate(im.view((1,) + im.shape), question_ids)[0].cpu() for im in img]
        
        return (tokens, answers)
    
    def test_epoch_end(self, validation_step_outputs):
        validation_step_outputs = list(validation_step_outputs)
    
        tokens_batch = [t for out in validation_step_outputs for t in out[0]]
        reference_batch = [r for out in validation_step_outputs for r in out[1]]
        
        generated_batch = tokenizer.batch_decode(tokens_batch)
        
        em = np.average([compute_exact(g, r) for g, r in zip(generated_batch, reference_batch)])
        f1 = np.average([compute_f1(g, r) for g, r in zip(generated_batch, reference_batch)])

        self.log("test_exact_match", em, prog_bar=True)
        self.log("test_word_f1", f1, prog_bar=True)
        
        return {
            "test_exact_match": em,
            "test_word_f1": f1,
        }
        
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)


class EfficientBERT5(pl.LightningModule):

    def __init__(self, params):
        super().__init__()

        self.decoder = T5ForConditionalGeneration.from_pretrained('t5-base')
        self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # Patch convolution. Basically tranform the 2D image into 16x16 patches with 2048 dimensions.
        self.encoder = EfficientNet.from_pretrained('efficientnet-b0', advprop=True, include_top=False)
        if params.freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        if params.freeze:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        # Convert patches from 2048 to 512 dimensions.
        self.bridge = nn.Conv2d(in_channels=1280, out_channels=self.decoder.config.d_model, kernel_size=1)
        
        self.max_len = params.max_len
        self.learning_rate = params.lr
    
    def _feature_fusion(self, img, question_ids):
        img_embeds = self._embeds_forward(img)
        question_embeds = self.text_encoder(question_ids).last_hidden_state
        inputs_embeds = torch.cat((img_embeds[:,:480,:], question_embeds[:,:32,:]), 1)
        return inputs_embeds

    def _embeds_forward(self, img):

        features = self.encoder.extract_features(img)

        features = self.bridge(features)

        inputs = features \
            .permute(0, 2, 3, 1) \
            .reshape(features.shape[0], -1, self.decoder.config.d_model)
        
        return inputs

    def forward(self, img=None, question_ids=None, inputs_embeds=None, decoder_input_ids=None, labels=None):

        # Pass efficientnet hidden states as embeddings for the transformer encoder input.
        inputs_embeds = self._feature_fusion(img, question_ids) if inputs_embeds is None else inputs_embeds

        return self.decoder(
            inputs_embeds=inputs_embeds,
            decoder_input_ids=decoder_input_ids, 
            labels=labels,
            return_dict=True
        )

    def generate(self, img, question_ids):

        # We need to implement our own generate loop as transformers doesn't accept 
        # precomputed embeddings on the generate method.
        # Issue: https://github.com/huggingface/transformers/issues/7626
        
        # Precompute embeddings to speedup generation as they don't change.
        inputs_embeds = self._feature_fusion(img, question_ids)

        decoder_input_ids = torch.full(
            (1, 1), self.decoder.config.decoder_start_token_id, dtype=torch.long, device=img.device
        )

        for i in range(self.max_len):
            with torch.no_grad():
                output = self.forward(decoder_input_ids=decoder_input_ids, 
                                      inputs_embeds=inputs_embeds)
            
            logits = output[0]
            next_token_logits = logits[:, -1, :]
            next_token_id = next_token_logits.argmax(1).unsqueeze(-1)
            decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=-1).to(img.device)

            if torch.eq(next_token_id[:, -1], self.decoder.config.eos_token_id).all():
                break

        return decoder_input_ids

    def training_step(self, batch, batch_idx):
        img, question_ids, answer_ids, answers = batch
        output = self(img=img, 
                      question_ids=question_ids, 
                      labels=answer_ids)
        return output[0]

    def validation_step(self, batch, batch_idx):
        img, question_ids, answer_ids, answers = batch
        tokens = [self.generate(im.view((1,) + im.shape), question_ids)[0].cpu() for im in img]
        return (tokens, answers)

    def validation_epoch_end(self, validation_step_outputs):
        validation_step_outputs = list(validation_step_outputs)
    
        tokens_batch = [t for out in validation_step_outputs for t in out[0]]
        reference_batch = [r for out in validation_step_outputs for r in out[1]]
        
        generated_batch = tokenizer.batch_decode(tokens_batch)

        em = np.average([compute_exact(g, r) for g, r in zip(generated_batch, reference_batch)])
        f1 = np.average([compute_f1(g, r) for g, r in zip(generated_batch, reference_batch)])

        self.log("val_exact_match", em, prog_bar=True)
        self.log("val_word_f1", f1, prog_bar=True)

    def test_step(self, batch, batch_idx):
        img, question_ids, answer_ids, answers = batch
        tokens = [self.generate(im.view((1,) + im.shape), question_ids)[0].cpu() for im in img]
        
        return (tokens, answers)
    
    def test_epoch_end(self, validation_step_outputs):
        validation_step_outputs = list(validation_step_outputs)
    
        tokens_batch = [t for out in validation_step_outputs for t in out[0]]
        reference_batch = [r for out in validation_step_outputs for r in out[1]]
        
        generated_batch = tokenizer.batch_decode(tokens_batch)
        
        em = np.average([compute_exact(g, r) for g, r in zip(generated_batch, reference_batch)])
        f1 = np.average([compute_f1(g, r) for g, r in zip(generated_batch, reference_batch)])

        self.log("test_exact_match", em, prog_bar=True)
        self.log("test_word_f1", f1, prog_bar=True)
        
        return {
            "test_exact_match": em,
            "test_word_f1": f1,
        }
        
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)
    
class OCRT5(pl.LightningModule):

    def __init__(self, params):
        super().__init__()

        self.model = T5ForConditionalGeneration.from_pretrained('t5-base')

        self.max_len = params.max_len
        self.learning_rate = params.lr

    def forward(self, input_token_ids, input_mask=None, target_token_ids=None):

        
        if self.training:
            # recomendação da documentação trocar pad para -100 para ser ignorado no cálculo
            target_token_ids[target_token_ids == 0] = -100
            # O fwd retorna uma tupla onde o primeiro elemento é a loss
            loss = self.model(
                        input_ids=input_token_ids,
                        attention_mask=input_mask,
                        labels=target_token_ids,
                        )[0]

            return loss
        else:
            # TODO gerar predicted_token ids
            predicted_token_ids = self.model.generate(
                                        input_ids=input_token_ids,
                                        attention_mask=input_mask,
                                        max_length=self.max_len,
                                                    )

            return predicted_token_ids
    
    def training_step(self, batch, batch_idx):
        input_ids, input_mask, answer_ids, questions, answers = batch
        loss = self(input_token_ids=input_ids,
                    input_mask = input_mask,
                    target_token_ids=answer_ids)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, input_mask, answer_ids, question, answers = batch
        tokens = self(input_token_ids=input_ids, input_mask=input_mask)
        return (tokens, answers)

    def validation_epoch_end(self, validation_step_outputs):
        validation_step_outputs = list(validation_step_outputs)
    
        tokens_batch = [t for out in validation_step_outputs for t in out[0]]
        reference_batch = [r for out in validation_step_outputs for r in out[1]]
        
        generated_batch = tokenizer.batch_decode(tokens_batch)

        em = np.average([compute_exact(g, r) for g, r in zip(generated_batch, reference_batch)])
        f1 = np.average([compute_f1(g, r) for g, r in zip(generated_batch, reference_batch)])

        self.log("val_exact_match", em, prog_bar=True)
        self.log("val_word_f1", f1, prog_bar=True)

    def test_step(self, batch, batch_idx):
        input_ids, input_mask, answer_ids, question, answers = batch
        tokens = self(input_token_ids=input_ids, input_mask=input_mask)
        return (tokens, answers)
    
    def test_epoch_end(self, validation_step_outputs):
        validation_step_outputs = list(validation_step_outputs)
    
        tokens_batch = [t for out in validation_step_outputs for t in out[0]]
        reference_batch = [r for out in validation_step_outputs for r in out[1]]
        
        generated_batch = tokenizer.batch_decode(tokens_batch)
        
        em = np.average([compute_exact(g, r) for g, r in zip(generated_batch, reference_batch)])
        f1 = np.average([compute_f1(g, r) for g, r in zip(generated_batch, reference_batch)])

        self.log("test_exact_match", em, prog_bar=True)
        self.log("test_word_f1", f1, prog_bar=True)
        
        return {
            "test_exact_match": em,
            "test_word_f1": f1,
        }
        
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)