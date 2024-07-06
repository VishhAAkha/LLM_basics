from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
import torch
import pandas as pd
import datasets

class LLM_Trainer:
    
    def __init__(self, filename, max_seq_length = 2048, load_in_4bit = True):
        self.max_seq_length = max_seq_length # Choose any! We auto support RoPE Scaling internally!
        self.dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        self.load_in_4bit = load_in_4bit # Use 4bit quantization to reduce memory usage. Can be False.

        # alpaca formatting template
        self.alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = "unsloth/llama-3-8b-bnb-4bit",
            max_seq_length = self.max_seq_length,
            dtype = self.dtype,
            load_in_4bit = self.load_in_4bit,
        )
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 16,
            lora_dropout = 0, # Supports any, but 0 is optimized
            bias = "none",    # Supports any, but "none" is optimized
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 3407,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )

        self.EOS_TOKEN = self.tokenizer.eos_token # Must add EOS_TOKEN
        self.load_data(filename)
        self.setup_trainer()

    def setup_trainer(self):
        # set up the model trainer
        self.trainer = SFTTrainer(
            model = self.model,
            tokenizer = self.tokenizer,
            train_dataset = self.dataset,
            dataset_text_field = "text",
            max_seq_length = self.max_seq_length,
            dataset_num_proc = 2,
            packing = False, # Can make training 5x faster for short sequences.
            args = TrainingArguments(
                per_device_train_batch_size = 2,
                gradient_accumulation_steps = 4,
                warmup_steps = 5,
                # max_steps = 60,
                max_steps = None,
                num_train_epochs = 1, # should change
                learning_rate = 2e-4,
                fp16 = not torch.cuda.is_bf16_supported(),
                bf16 = torch.cuda.is_bf16_supported(),
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                seed = 3407,
                output_dir = "outputs",
            ),
        )

    # format data to work with unsloth and alpaca training
    def formatting_prompts_func(self, examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]

        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = self.alpaca_prompt.format(str(instruction), str(input), str(output)) + self.EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }

    def load_data(self, filename):
        df = pd.concat(pd.read_excel(filename, engine = "openpyxl", sheet_name = None).values())
        df = df.astype(str)

        # modify training data to work with the alpaca format
        instructions, inputs, responses = [], [], []
        for idx, row in df.iterrows():
            instructions.append(row["QUESTION"])
            inputs.append("")
            responses.append(row["ANSWER"])
        self.dataset = datasets.Dataset.from_pandas(pd.DataFrame({"instruction": instructions, "input": inputs, "output": responses}))
        self.dataset = self.dataset.map(self.formatting_prompts_func, batched = True,)

    def train_model(self):
        print(self.trainer.train())

    def save_model(self, hf_link):
        # push the lora model to hugging face
        self.model.push_to_hub_merged(hf_link, self.tokenizer, save_method = "lora")
