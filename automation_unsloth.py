import json
import os

import requests
import yaml
from loguru import logger
from huggingface_hub import HfApi

from train_unsloth import train_unsloth
from utils.constants_unsloth import model2base_model, model2size
from utils.flock_api import get_task, submit_task
from utils.gpu_utils import get_gpu_type

HF_USERNAME = os.environ["HF_USERNAME"]

if __name__ == "__main__":
    task_id = os.environ["TASK_ID"]
    # load trainin args
    # define the path of the current file
    current_folder = os.path.dirname(os.path.realpath(__file__))
    with open(f"{current_folder}/training_args.yaml", "r") as f:
        all_training_args = yaml.safe_load(f)

    task = get_task(task_id)
    # log the task info
    logger.info(json.dumps(task, indent=4))
    # download data from a presigned url
    data_url = task["data"]["training_set_url"]
    context_length = task["data"]["context_length"]
    max_params = task["data"]["max_params"]

    # filter out the model within the max_params
    model2size = {k: v for k, v in model2size.items() if v <= max_params}
    all_training_args = {k: v for k, v in all_training_args.items() if k in model2size}
    logger.info(f"Models within the max_params: {all_training_args.keys()}")
    # download in chunks
    response = requests.get(data_url, stream=True)
    with open("data/demo_data.jsonl", "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # train all feasible models and merge
    for model_id in all_training_args.keys():
        logger.info(f"Start to train the model {model_id}...")
    # if OOM, proceed to the next model
    
        max_seq_length = context_length = 4096 # Choose any! We auto support RoPE Scaling internally!
        dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
        #model_id = "unsloth/Qwen2.5-7B-instruct"
        model_id = "mistralai/Mistral-7B-v0.1"
        #model_id = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
        args = all_training_args[model_id]
        train_unsloth(model_id, max_seq_length, context_length, load_in_4bit, full_train = True, **args)


        # generate a random repo id based on timestamp
        gpu_type = get_gpu_type()

        try:
            logger.info("Start to push the lora weight to the hub...")
            api = HfApi(token=os.environ["HF_TOKEN"])
            repo_name = f"{HF_USERNAME}/task-{task_id}-{model_id.replace('/', '-').replace('2.5', '1.5')}"
            # check whether the repo exists
            try:
                api.create_repo(
                    repo_name,
                    exist_ok=False,
                    repo_type="model",
                )
            except Exception as e:
                print(e)
                logger.info(
                    f"Repo {repo_name} already exists. Will commit the new version."
                )

            commit_message = api.upload_folder(
                folder_path="outputs",
                repo_id=repo_name,
                repo_type="model",
            )
            # get commit hash
            commit_hash = commit_message.oid
            logger.info(f"Commit hash: {commit_hash}")
            logger.info(f"Repo name: {repo_name}")
            # submit
            submit_task(
                task_id, repo_name, model2base_model[model_id], gpu_type, commit_hash
            )
            logger.info("Task submitted successfully")
        except Exception as e:
            logger.error(f"Error: {e}")
            logger.info("Proceed to the next model...")
        finally:
            # cleanup merged_model and output
            os.system("rm -rf merged_model")
            os.system("rm -rf outputs")
            continue
