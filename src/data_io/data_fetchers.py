import os
import logging
import requests


logger = logging.getLogger(__name__)


def fetch_txt_data(data_url, output_dir_path):
    """download a .txt dataset"""
    input_file_path = os.path.join(output_dir_path, 'input.txt')
    if not os.path.exists(input_file_path):
        with open(input_file_path, 'w') as f:
            f.write(requests.get(data_url).text)

    with open(input_file_path, 'r') as f:
        data = f.read()
    
    logger.info(f"length of dataset in characters: {len(data):,}")
    
    return data
