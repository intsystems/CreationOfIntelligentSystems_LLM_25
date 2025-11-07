import yaml
import os

def create_config_dict(topic : str):
    config_dict = {
        'topic' : topic,
        'is_positive' : True,
        'output_dir' : f'/home/jovyan/rusakov/dim_lm/activations_topics/open_llama_3b/{topic}',
        'path_to_topic' : f'/home/jovyan/zorin/GCS/datasets/openai/{topic}.pkl',
        'max_length':256,
        'batch_size': 256
    }   
    return config_dict

def save_config(topic : str):
    config_path = f'/home/jovyan/rusakov/dim_lm/configs/{topic}_topic_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(create_config_dict(topic), f)

for topic in os.listdir('/home/jovyan/zorin/GCS/datasets/openai'):
    save_config(topic[:-4])