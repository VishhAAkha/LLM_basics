from huggingface_hub import login
from thai_llm_trainer import LLM_Trainer
import warnings

login()

# suppress warnings
warnings.filterwarnings("ignore")

# load, train, and save thai llm
llm_trainer = LLM_Trainer("all_data.xlsx")
llm_trainer.train_model()
llm_trainer.save_model("raghav2005/thai_llm_lora")
