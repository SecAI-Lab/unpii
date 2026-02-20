import torch
from torch import nn
import torch.nn.functional as F
from transformers import Trainer
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available
import datasets
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import torch.nn.functional as F
import copy, os
import deepspeed
from evaluate_util import get_dataloader, get_all_evals, get_kl_divergence
import copy
import json 
from pathlib import Path
from data_module import get_batch_loss 
from utils import merge_dicts, interleave_eval_result_dict, get_forget_quality, get_model_utility
import numpy as np
from scipy.stats import ks_2samp, hmean
import csv 
import pickle
import math
import openai

def printll(name, inp):
    #print list with 4 decimal for each item
    print(name, [round(x, 4) for x in inp])

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids, labels, attention_mask = inputs
        outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)
    

class CustomTrainerForgetting(Trainer):
    def __init__(self, *args, **kwargs):
        self.loss_type = kwargs.pop('forget_loss')
        self.oracle_model = kwargs.pop('oracle_model')
        self.eval_cfg = kwargs.pop('eval_cfg')
        self.seed = kwargs.pop('seed')

        # the coefficient of each part in the loss function. This is used in ablation study.
        self.npo_coeff=kwargs.pop('npo_coeff')
        self.grad_diff_coeff=kwargs.pop('grad_diff_coeff')
        self.KL_coeff=kwargs.pop('KL_coeff')

        self.ref_policy = kwargs.pop('ref_policy')

        self.beta = kwargs.pop('beta')

        super(CustomTrainerForgetting, self).__init__(*args, **kwargs)

        # Here, we always need the oracle model to compute the KL distance in the evaluation time.
        self.oracle_model = self.e_prepare_deepspeed(self.oracle_model)

    def get_train_dataloader(self):
        """
        Override the original get_train_dataloader function simply for debugging.
        This is identical to the get_train_dataloader function in transformer.Trainer.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.state.global_step)
        print(f'Generator........Epoch-{self.state.global_step}')

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            
            dataloader_params["generator"] = generator
            dataloader_params["shuffle"] = True # set shuffle=True with specified generator.
            # dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))


    def e_prepare_deepspeed(self, model):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = copy.deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        config_kwargs["optimizer"] = {"type": None}
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        #set the gradients to false for every parameter
        for param in model.parameters():
            param.requires_grad = False
        
        return model

    # --- Shared Utility Methods for PII Methods ---
    def get_pii_penalty(self, text):
        """Calculates PII exposure penalty using GPT-4o-mini API."""
        prompt = (
            "Analyze the text below for these PII categories and combinations.\n"
            "Penalty scale:\n"
            "- Gender: 0.173\n"
            "- Region address: 0.175\n"
            "- ZIP code: 0.179\n"
            "- Date of birth: 0.179\n"
            "- Name: 0.183\n"
            "- Detailed address: 0.224\n"
            "- Name + Region address: 0.349\n"
            "- Name + Detailed address: 0.392\n"
            "- Phone number: 0.404\n"
            "- Driver license number: 0.456\n"
            "- Region address + Gender + Date of birth: 0.488\n"
            "- Gender + ZIP code + Date of birth: 0.491\n"
            "- Social security number: 0.513\n"
            "- Passport number: 0.513\n"
            "- Name + Medical records: 0.544\n"
            "- Name + Credit card number: 0.567\n"
            "- Name + Bank account number: 0.665\n\n"
            "If combinations match (e.g., Name and Credit card), use the combined score. Otherwise, sum the individual scores. Output the total score ONLY as a single float. Output 0 if none.\n"
            f"Text: '''{text}'''"
        )
        try:
            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "##"))
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "You are a PII expert."}, {"role": "user", "content": prompt}],
                temperature=0, max_tokens=50
            )
            val = float(response.choices[0].message.content.strip())
            print(f"PII Penalty result: {val}")
            return val
        except: return 0.0

    def compute_seq_log_prob(self, model, prefix_ids, full_ids):
        """Computes sequence log probability for newly generated tokens."""
        L = prefix_ids.shape[1]
        outputs = model(full_ids, attention_mask=(full_ids != self.tokenizer.pad_token_id))
        new_logits = outputs.logits[:, L-1:-1, :]
        log_probs = F.log_softmax(new_logits, dim=-1)
        return torch.gather(log_probs, dim=-1, index=full_ids[:, L:].unsqueeze(-1)).squeeze(-1).sum(dim=-1)

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.loss_type == "grad_ascent":
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)         ##attention_mask is used to indicate which tokens to attend to ()
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1
            loss = forget_loss

        elif self.loss_type == "grad_ascent_forgetKL":
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)         
            forget_loss = -1 * outputs.loss

            with torch.no_grad():
                oracle_outputs = self.oracle_model(input_ids,labels=labels, attention_mask=attention_mask)
            oracle_probs = F.log_softmax(oracle_outputs.logits, dim=-1)
            oracle_probs = oracle_probs.view(-1, oracle_outputs.logits.shape[-1])
            current_probs = F.log_softmax(outputs.logits, dim=-1)
            current_probs = current_probs.view(-1, outputs.logits.shape[-1])

            kl_loss = nn.functional.kl_div(current_probs, oracle_probs, reduction='batchmean', log_target=True)
            loss = forget_loss + kl_loss

        elif self.loss_type == "grad_diff":
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            loss = forget_loss + retain_loss
        
        elif self.loss_type == "KL":
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            with torch.no_grad():
                retain_outputs = self.oracle_model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            
            retain_probs = F.log_softmax(retain_outputs.logits, dim=-1)
            retain_probs = retain_probs.view(-1, retain_outputs.logits.shape[-1])

            current_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            current_probs = F.log_softmax(current_outputs.logits, dim=-1)
            current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])

            #minimum KL divergence
            retain_loss = nn.functional.kl_div(current_probs, retain_probs, reduction='batchmean', log_target=True)
            loss = forget_loss + retain_loss

        elif self.loss_type == "idk":
            idk_inputs, retain_inputs = inputs
            print("aaaaaaaaaitcheck1================================================")
            idk_input_ids, idk_labels, idk_attention_mask = idk_inputs
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            ######
            decode_idk = self.tokenizer.decode(idk_input_ids[0].tolist()) 
            decode_retain = self.tokenizer.decode(retain_input_ids[0].tolist()) 
            print(f'retain_input_ids  : {decode_retain}' )
            print(f'idk_input_ids  : {decode_idk}' )
            ######
            #concatenate the inputs. single forward pass is much more efficient
            input_ids = torch.cat((idk_input_ids, retain_input_ids), dim=0)
            labels = torch.cat((idk_labels, retain_labels), dim=0)
            attention_mask = torch.cat((idk_attention_mask, retain_attention_mask), dim=0)
            
            #decode_input = self.tokenizer.decode(input_ids[0].tolist()) 
            #####
            decode_input = [self.tokenizer.decode(seq.tolist(), skip_special_tokens=True) for seq in input_ids]
            print(f'input_ids  : {decode_input}' )
            #####

            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            #####
            logits = outputs.logits 
            generated_tokens = torch.argmax(logits, dim=-1)
            generated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            print(f'generated_texts : {generated_texts}' )
            #####
            loss = outputs.loss
        
        elif self.loss_type in ["dpo","dpo_grad_diff","dpo_KL"]:
            idk_inputs, forget_inputs, retain_inputs = inputs
            idk_input_ids, idk_labels, idk_attention_mask = idk_inputs
            forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
            idk_outputs = model(idk_input_ids,labels=idk_labels, attention_mask=idk_attention_mask)
            forget_outputs = model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask)
            with torch.no_grad():
                idk_outputs_oracle = self.oracle_model(idk_input_ids,labels=idk_labels, attention_mask=idk_attention_mask)
                forget_outputs_oracle = self.oracle_model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask)
                idk_logits_oracle = idk_outputs_oracle.logits
                forget_logits_oracle = forget_outputs_oracle.logits

                idk_loss_oracle = -1 * get_batch_loss(idk_logits_oracle, idk_labels)
                forget_loss_oracle = -1 * get_batch_loss(forget_logits_oracle, forget_labels)
            
            idk_loss_current = -1 * get_batch_loss(idk_outputs.logits, idk_labels)
            forget_loss_current = -1 * get_batch_loss(forget_outputs.logits, forget_labels)

            pi_logratios = idk_loss_current - forget_loss_current
            ref_logratios = idk_loss_oracle - forget_loss_oracle
            loss = -F.logsigmoid(self.beta * (pi_logratios - ref_logratios)).mean()*2/self.beta

            if self.loss_type == 'dpo_grad_diff':
                retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
                retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
                retain_loss = retain_outputs.loss
                loss = loss + retain_loss

            elif self.loss_type == 'dpo_KL':
                retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
                with torch.no_grad():
                    retain_outputs = self.oracle_model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
                retain_probs = F.log_softmax(retain_outputs.logits, dim=-1)
                retain_probs = retain_probs.view(-1, retain_outputs.logits.shape[-1])

                current_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
                current_probs = F.log_softmax(current_outputs.logits, dim=-1)
                current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])

                #minimum KL divergence
                retain_loss = nn.functional.kl_div(current_probs, retain_probs, reduction='batchmean', log_target=True)
                loss = loss + retain_loss

        ### Implement the NPO
        elif self.loss_type == 'npo':
            forget_inputs, _ = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)

            forget_loss_current = get_batch_loss(outputs.logits, labels) 

            if self.ref_policy == 'fine_tuned':
                with torch.no_grad():
                    forget_outputs_oracle = self.oracle_model(input_ids,labels=labels, attention_mask=attention_mask)
                    forget_logits_oracle = forget_outputs_oracle.logits
                    forget_loss_oracle = get_batch_loss(forget_logits_oracle, labels)
                neg_log_ratios = forget_loss_current - forget_loss_oracle
            else:
                raise NotImplementedError
            loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta 

        elif self.loss_type == 'npo_grad_diff':
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss_current = get_batch_loss(outputs.logits, labels) 

            if self.ref_policy == 'fine_tuned':
                with torch.no_grad():
                    forget_outputs_oracle = self.oracle_model(input_ids,labels=labels, attention_mask=attention_mask)
                    forget_logits_oracle = forget_outputs_oracle.logits
                    forget_loss_oracle = get_batch_loss(forget_logits_oracle, labels)
                neg_log_ratios = forget_loss_current - forget_loss_oracle
            else:
                raise NotImplementedError
            forget_loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            loss = self.npo_coeff * forget_loss + self.grad_diff_coeff * retain_loss
            
        elif self.loss_type == 'npo_KL':
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss_current = get_batch_loss(outputs.logits, labels) 
            if self.ref_policy == 'fine_tuned':
                with torch.no_grad():
                    forget_outputs_oracle = self.oracle_model(input_ids,labels=labels, attention_mask=attention_mask)
                    forget_logits_oracle = forget_outputs_oracle.logits
                    forget_loss_oracle = get_batch_loss(forget_logits_oracle, labels)
                neg_log_ratios = forget_loss_current - forget_loss_oracle
            else:
                raise NotImplementedError
            forget_loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            with torch.no_grad():
                retain_outputs = self.oracle_model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_probs = F.log_softmax(retain_outputs.logits, dim=-1)
            retain_probs = retain_probs.view(-1, retain_outputs.logits.shape[-1])

            current_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            current_probs = F.log_softmax(current_outputs.logits, dim=-1)
            current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])

            #minimum KL divergence
            retain_loss = nn.functional.kl_div(current_probs, retain_probs, reduction='batchmean', log_target=True)
            loss = self.npo_coeff * forget_loss + self.KL_coeff * retain_loss

        elif self.loss_type == 'kto_sigmoid':
            idk_inputs, forget_inputs, retain_inputs = inputs
            idk_input_ids, idk_labels, idk_attention_mask = idk_inputs
            forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
            
            with torch.no_grad():
                idk_outputs = model(idk_input_ids,labels=idk_labels, attention_mask=idk_attention_mask)
                idk_outputs_oracle = self.oracle_model(idk_input_ids,labels=idk_labels, attention_mask=idk_attention_mask)
                idk_loss_log = -1 * get_batch_loss(idk_outputs.logits, idk_labels)
                idk_loss_log_oracle = -1 * get_batch_loss(idk_outputs_oracle.logits, idk_labels)
                
                KL_term = (idk_loss_log - idk_loss_log_oracle).mean()

                forget_outputs_oracle = self.oracle_model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask)
                forget_loss_oracle = -1 * get_batch_loss(forget_outputs_oracle.logits, forget_labels)

            forget_outputs = model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask)
            forget_loss = -1 * get_batch_loss(forget_outputs.logits, forget_labels)
            log_ratios = forget_loss - forget_loss_oracle
            loss = 1.0 - F.sigmoid(KL_term - self.beta * log_ratios).mean() * 2 / self.beta

        elif self.loss_type == 'kto_logsigmoid':
            idk_inputs, forget_inputs, retain_inputs = inputs
            idk_input_ids, idk_labels, idk_attention_mask = idk_inputs
            forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
            
            with torch.no_grad():
                idk_outputs = model(idk_input_ids,labels=idk_labels, attention_mask=idk_attention_mask)
                idk_outputs_oracle = self.oracle_model(idk_input_ids,labels=idk_labels, attention_mask=idk_attention_mask)
                idk_loss_log = -1 * get_batch_loss(idk_outputs.logits, idk_labels)
                idk_loss_log_oracle = -1 * get_batch_loss(idk_outputs_oracle.logits, idk_labels)
                
                KL_term = (idk_loss_log - idk_loss_log_oracle).mean()

                forget_outputs_oracle = self.oracle_model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask)
                forget_loss_oracle = -1 * get_batch_loss(forget_outputs_oracle.logits, forget_labels)

            forget_outputs = model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask)
            forget_loss = -1 * get_batch_loss(forget_outputs.logits, forget_labels)
            log_ratios = forget_loss - forget_loss_oracle
            loss = 1.0 - F.logsigmoid(KL_term - self.beta * log_ratios).mean() * 2 / self.beta

        elif self.loss_type == 'kto_logsigmoid_grad_diff':
            idk_inputs, forget_inputs, retain_inputs = inputs
            idk_input_ids, idk_labels, idk_attention_mask = idk_inputs
            forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
            
            with torch.no_grad():
                idk_outputs = model(idk_input_ids,labels=idk_labels, attention_mask=idk_attention_mask)
                idk_outputs_oracle = self.oracle_model(idk_input_ids,labels=idk_labels, attention_mask=idk_attention_mask)
                idk_loss_log = -1 * get_batch_loss(idk_outputs.logits, idk_labels)
                idk_loss_log_oracle = -1 * get_batch_loss(idk_outputs_oracle.logits, idk_labels)
                
                KL_term = (idk_loss_log - idk_loss_log_oracle).mean()

                forget_outputs_oracle = self.oracle_model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask)
                forget_loss_oracle = -1 * get_batch_loss(forget_outputs_oracle.logits, forget_labels)

            forget_outputs = model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask)
            forget_loss = -1 * get_batch_loss(forget_outputs.logits, forget_labels)
            log_ratios = forget_loss - forget_loss_oracle
            forget_loss = 1.0 - F.logsigmoid(KL_term - self.beta * log_ratios).mean() * 2 / self.beta
            print(KL_term)

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss

            loss = forget_loss + retain_loss

        # --- Request 3 methodologies Integration (Renamed & Optimized) ---

        elif self.loss_type == 'grad_unpii':
            """Weighted Gradient Ascent by PII risk scores."""
            forget_inputs, _ = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            
            loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
            base_losses = loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
            texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            # Use internal GPT penalty function
            pii_penalties = torch.tensor([self.get_pii_penalty(text) for text in texts], device=base_losses.device)
            weighted_losses = base_losses * (1 + pii_penalties.repeat_interleave(labels.size(1)))
            
            loss = -torch.mean(weighted_losses)

		elif self.loss_type == 'npo_unpii':
            """
            Methodology: npo_unpii (Standard NPO with Risk Scaling)
            Calculates NPO loss (based on Oracle reference) and scales it by PII risk.
            """
            import torch
            import torch.nn.functional as F

            forget_inputs, _ = inputs 
            input_ids, labels, attention_mask = forget_inputs

            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            with torch.no_grad():
                oracle_outputs = self.oracle_model(input_ids, labels=labels, attention_mask=attention_mask)

            current_nll = get_batch_loss(outputs.logits, labels) # P(theta)
            oracle_nll = get_batch_loss(oracle_outputs.logits, labels) # P(ref)

            log_ratios = oracle_nll - current_nll
            npo_loss_per_sample = -F.logsigmoid(self.beta * log_ratios) * 2 / self.beta

            texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            pii_penalties = torch.tensor(
                [self.get_pii_penalty(text) for text in texts],
                device=npo_loss_per_sample.device
            )

            weighted_loss = npo_loss_per_sample * (1 + pii_penalties)
            
            loss = weighted_loss.mean()
			
			
		elif self.loss_type == 'dpo_unpii':
            """
            Methodology: dpo_unpii (formerly dpo_pii_rl4)
            Implements DPO where the log-likelihood is penalized by PII risk (Implicit Reward).
            """
            import torch
            import torch.nn.functional as F

            idk_inputs, forget_inputs, _ = inputs
            idk_input_ids, idk_labels, idk_mask = idk_inputs
            forget_input_ids, forget_labels, forget_mask = forget_inputs

            idk_outputs = model(idk_input_ids, labels=idk_labels, attention_mask=idk_mask)
            forget_outputs = model(forget_input_ids, labels=forget_labels, attention_mask=forget_mask)

            with torch.no_grad():
                idk_oracle = self.oracle_model(idk_input_ids, labels=idk_labels, attention_mask=idk_mask)
                forget_oracle = self.oracle_model(forget_input_ids, labels=forget_labels, attention_mask=forget_mask)

            idk_loss_pi = -1 * get_batch_loss(idk_outputs.logits, idk_labels)
            forget_loss_pi = -1 * get_batch_loss(forget_outputs.logits, forget_labels)

            idk_loss_ref = -1 * get_batch_loss(idk_oracle.logits, idk_labels)
            forget_loss_ref = -1 * get_batch_loss(forget_oracle.logits, forget_labels)

            forget_texts = self.tokenizer.batch_decode(forget_input_ids, skip_special_tokens=True)
            raw_penalties = [self.get_pii_penalty(text) for text in forget_texts]
            pii_penalties = torch.tensor(raw_penalties, device=forget_loss_pi.device, dtype=forget_loss_pi.dtype)

            scaled_penalties = torch.log1p(torch.clamp(pii_penalties, 0.0, 2.0))
            step_ratio = min(1.0, float(self.state.global_step) / (self.state.max_steps * 0.3 + 1e-6))
            final_penalties = 5.0 * step_ratio * scaled_penalties

            forget_loss_pi = forget_loss_pi - final_penalties

            pi_logratios = idk_loss_pi - forget_loss_pi
            ref_logratios = idk_loss_ref - forget_loss_ref
            
            loss = -F.logsigmoid(self.beta * (pi_logratios - ref_logratios)).mean() * 2.0 / self.beta

        return (loss, outputs) if return_outputs else loss
    
    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)

    def evaluate(
        self,
        eval_dataset = None,
        ignore_keys = None,
        metric_key_prefix = "eval",
    ):
        '''
        RZ: Call this function in Trainer.train() when evakluating the performace of each checkpoint.
        '''

        args = self.args
        model = self._wrap_model(self.model, training=False, dataloader=None)

        print('####### Evaluating the model...... #######')
        print(self.is_in_train, args.device, model.dtype, self.args.dataloader_num_workers, self.eval_cfg.split_list)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)
        
        model.eval()
        curr_step = self.state.global_step
        eval_cfg = self.eval_cfg

        curr_save_dir = os.path.join(eval_cfg.save_dir, f"checkpoint-{curr_step}")
        Path(curr_save_dir).mkdir(parents=True, exist_ok=True)

        forget_rate = eval_cfg.split_list[-1].split('_')[0]

        with torch.no_grad():
            for i, (folder, split, question_key, answer_key, eval_task, base_answer_key, perturbed_answer_key) in enumerate(zip(eval_cfg.data_path, eval_cfg.split_list, eval_cfg.question_key, eval_cfg.answer_key, eval_cfg.eval_task, eval_cfg.base_answer_key, eval_cfg.perturbed_answer_key)):
                
                world_size = self.accelerator.num_processes

                # For some reason, Hydra is not interprating the split correctly
                if eval_task == 'eval_log_forget':
                    split = eval_cfg.split
                print(f'Working on eval task {eval_task} with split {split}')
                save_filename = os.path.join(curr_save_dir, f"{eval_task}.json")

                save_filename = save_filename if world_size == 1 else os.path.join(curr_save_dir, f"{eval_task}_{self.accelerator.local_process_index}.json")
                if os.path.exists(save_filename) and not eval_cfg.overwrite:
                    print(f"Skipping {eval_task} because {save_filename} already exists")
                    continue

                eval_dataloader, base_eval_dataloader, perturb_dataloader = get_dataloader(eval_cfg, eval_task, self.tokenizer, folder, split, question_key, answer_key, base_answer_key, perturbed_answer_key)

                eval_dataloader = self.accelerator.prepare(eval_dataloader)
                # print('dataset condition: ', len(eval_dataloader.dataset), self.accelerator.local_process_index)
                base_eval_dataloader = self.accelerator.prepare(base_eval_dataloader)
                perturb_dataloader = self.accelerator.prepare(perturb_dataloader)

                # if int(os.environ.get('RANK', '0')) == 0:
                #    import pdb; pdb.set_trace()

                eval_logs = get_all_evals(eval_cfg, model, self.tokenizer, eval_task, eval_dataloader, base_eval_dataloader, perturb_dataloader)
                
                kl_divergence_log = get_kl_divergence(model, self.oracle_model, eval_dataloader)
                eval_logs['kl_divergence'] = kl_divergence_log

                with open(save_filename, "w") as f:
                    # pretty write json to f
                    json.dump(eval_logs, f, indent=4)
            
                #wait for all process to finish
            self.accelerator.wait_for_everyone()
            aggregated_eval_logs = {}
            for eval_task in eval_cfg.eval_task:
                #read the saved file as json and merge them using merge_dicts

                if world_size > 1:
                    if self.accelerator.is_local_main_process:
                        eval_logs = json.load(open(os.path.join(curr_save_dir, f"{eval_task}_0.json")))

                        for i in range(1, world_size):
                            filename = os.path.join(curr_save_dir, f"{eval_task}_{i}.json")
                            eval_logs = merge_dicts(eval_logs, json.load(open(filename)))
                        
                        aggregated_eval_logs[f'{eval_task}.json'] = eval_logs

                        new_save_filename = os.path.join(curr_save_dir, f"{eval_task}.json")
                        with open(new_save_filename, "w") as f:
                            # pretty write json to f
                            json.dump(eval_logs, f, indent=4)
                            #delete old files use shutil
                            for i in range(world_size):
                                filename = os.path.join(curr_save_dir, f"{eval_task}_{i}.json")
                                os.remove(filename)

                else:
                    if self.accelerator.is_local_main_process:
                        eval_logs = json.load(open(os.path.join(curr_save_dir, f"{eval_task}.json")))
                        aggregated_eval_logs[f'{eval_task}.json'] = eval_logs
                                
            if self.accelerator.is_local_main_process:

                aggregated_eval_logs = interleave_eval_result_dict(aggregated_eval_logs, forget_rate, large_bsz=eval_cfg.batch_size, num_processes=world_size)
                aggregated_eval_log_filename = os.path.join(curr_save_dir, "eval_log_aggregated.json")

                with open(aggregated_eval_log_filename, 'w') as f:
                    json.dump(aggregated_eval_logs, f, indent=4)
                
                model_utility = get_model_utility(aggregated_eval_logs)
                retain_result = json.load(open(eval_cfg.retain_result, 'r'))
                forget_quality, trust_ratio = get_forget_quality(aggregated_eval_logs, retain_result)
                aaggregate_stat = {**model_utility, **forget_quality}

                aaggregate_stat['curr_step'] = curr_step
                aaggregate_stat['seed'] = self.seed
                aaggregate_stat['loss_type'] = self.loss_type

                with open(os.path.join(curr_save_dir, "aggregate_stat.txt"), 'w') as txtfile:
                    for key, value in aaggregate_stat.items():
                        txtfile.write(f"{key}: {value}\n")

                with open(os.path.join(curr_save_dir, "truth_ratio.pkl"), 'wb') as picklefile:
                    pickle.dump(trust_ratio, picklefile)

class CustomTrainerRetraining(Trainer):
    def __init__(self, *args, **kwargs):
        self.eval_cfg = kwargs.pop('eval_cfg')
        self.seed = kwargs.pop('seed')
        super(CustomTrainerRetraining, self).__init__(*args, **kwargs)

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.state.global_step)
        print(f'Generator........Epoch-{self.state.global_step}')

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["generator"] = generator
            dataloader_params["shuffle"] = True # set shuffle=True with specified generator.
            # dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids, labels, attention_mask = inputs
        outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)

    def evaluate(
        self,
        eval_dataset = None,
        ignore_keys = None,
        metric_key_prefix = "eval",
    ):

        args = self.args
        model = self._wrap_model(self.model, training=False, dataloader=None)

        print('####### Evaluating the model...... #######')
        print(self.is_in_train, args.device, model.dtype, self.args.dataloader_num_workers, self.eval_cfg.split_list)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)
        
        model.eval()
        curr_step = self.state.global_step
        eval_cfg = self.eval_cfg

        curr_save_dir = os.path.join(eval_cfg.save_dir, f"checkpoint-{curr_step}")
        Path(curr_save_dir).mkdir(parents=True, exist_ok=True)

        forget_rate = eval_cfg.split.split('_')[0]

        with torch.no_grad():
            for i, (folder, split, question_key, answer_key, eval_task, base_answer_key, perturbed_answer_key) in enumerate(zip(eval_cfg.data_path, eval_cfg.split_list, eval_cfg.question_key, eval_cfg.answer_key, eval_cfg.eval_task, eval_cfg.base_answer_key, eval_cfg.perturbed_answer_key)):

                world_size = self.accelerator.num_processes

                # For some reason, Hydra is not interprating the split correctly
                if eval_task == 'eval_log_forget':
                    split = eval_cfg.split
                print(f'Working on eval task {eval_task} with split {split}')
                save_filename = os.path.join(curr_save_dir, f"{eval_task}.json")

                save_filename = save_filename if world_size == 1 else os.path.join(curr_save_dir, f"{eval_task}_{self.accelerator.local_process_index}.json")
                # print(save_filename)
                if os.path.exists(save_filename) and not eval_cfg.overwrite:
                    print(f"Skipping {eval_task} because {save_filename} already exists")
                    continue

                eval_dataloader, base_eval_dataloader, perturb_dataloader = get_dataloader(eval_cfg, eval_task, self.tokenizer, folder, split, question_key, answer_key, base_answer_key, perturbed_answer_key)
                eval_dataloader = self.accelerator.prepare(eval_dataloader)
                # print('dataset condition: ', len(eval_dataloader.dataset), self.accelerator.local_process_index)
                base_eval_dataloader = self.accelerator.prepare(base_eval_dataloader)
                perturb_dataloader = self.accelerator.prepare(perturb_dataloader)

                eval_logs = get_all_evals(eval_cfg, model, self.tokenizer, eval_task, eval_dataloader, base_eval_dataloader, perturb_dataloader)
                with open(save_filename, "w") as f:
                    # pretty write json to f
                    json.dump(eval_logs, f, indent=4)
            
                #wait for all process to finish
            self.accelerator.wait_for_everyone()
            aggregated_eval_logs = {}
            for eval_task in eval_cfg.eval_task:
                #read the saved file as json and merge them using merge_dicts

                if world_size > 1:
                    if self.accelerator.is_local_main_process:
                        eval_logs = json.load(open(os.path.join(curr_save_dir, f"{eval_task}_0.json")))

                        for i in range(1, world_size):
                            filename = os.path.join(curr_save_dir, f"{eval_task}_{i}.json")
                            eval_logs = merge_dicts(eval_logs, json.load(open(filename)))
                        
                        aggregated_eval_logs[f'{eval_task}.json'] = eval_logs

                        new_save_filename = os.path.join(curr_save_dir, f"{eval_task}.json")
                        with open(new_save_filename, "w") as f:
                            # pretty write json to f
                            json.dump(eval_logs, f, indent=4)
                            #delete old files use shutil
                            for i in range(world_size):
                                filename = os.path.join(curr_save_dir, f"{eval_task}_{i}.json")
                                os.remove(filename)

                else:
                    if self.accelerator.is_local_main_process:
                        eval_logs = json.load(open(os.path.join(curr_save_dir, f"{eval_task}.json")))
                        aggregated_eval_logs[f'{eval_task}.json'] = eval_logs
                                
            if self.accelerator.is_local_main_process:

                aggregated_eval_logs = interleave_eval_result_dict(aggregated_eval_logs, forget_rate, large_bsz=eval_cfg.batch_size, num_processes=world_size)
                aggregated_eval_log_filename = os.path.join(curr_save_dir, "eval_log_aggregated.json")

                with open(aggregated_eval_log_filename, 'w') as f:
                    json.dump(aggregated_eval_logs, f, indent=4)
                
def custom_data_collator_forget(samples):
    rets = []
    if len(samples[0]) == 3:
        idk_samples, forget_samples, retain_samples = [sample[0] for sample in samples], [sample[1] for sample in samples], [sample[2] for sample in samples]
        data_types = ["idk", "forget", "retain"]
    elif len(samples[0]) == 2:
        forget_samples, retain_samples = [sample[0] for sample in samples], [sample[1] for sample in samples]
        data_types = ["forget", "retain"]
    for data_type in data_types:
        if data_type == "forget":
            data = forget_samples 
        elif data_type == "retain":
            data = retain_samples 
        elif data_type == "idk":
            data = idk_samples
         
        input_ids = [s[0] for s in data]
        labels = [s[1] for s in data]
        attention_mask = [s[2] for s in data]
        rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))
    return rets

def compute_metrics(pred):
    logits, labels = torch.from_numpy(pred.predictions), torch.from_numpy(pred.label_ids)
    preds = torch.from_numpy(pred.predictions.argmax(-1))
    shifted_labels = labels[..., 1:].contiguous()
    acc = torch.mean((preds[..., :-1] == shifted_labels).float())
    loss  = get_loss(logits, labels)
    return {"eval accuracy": acc, "eval loss": loss.item()}

def get_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_function(output.view(-1, output.size(-1)), shifted_labels.view(-1))

    return loss
