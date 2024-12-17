from omnigibson.utils.asset_utils import decrypt_file

encrypted_filename = "/home/arpit/projects/OmniGibson/omnigibson/data/datasets/og_dataset/objects/fridge/hivvdf/usd/hivvdf.encrypted.usd"
usd_path = "/home/arpit/projects/OmniGibson/omnigibson/data/datasets/og_dataset/objects/fridge/hivvdf/usd/hivvdf.usd"
decrypt_file(encrypted_filename, usd_path)