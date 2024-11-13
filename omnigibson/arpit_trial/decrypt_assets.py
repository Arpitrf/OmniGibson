from omnigibson.utils.asset_utils import decrypt_file

encrypted_filename = "/home/arpit/test_projects/OmniGibson/omnigibson/data/og_dataset/objects/fridge/hivvdf/usd/hivvdf.encrypted.usd"
usd_path = "/home/arpit/test_projects/OmniGibson/omnigibson/data/og_dataset/objects/fridge/hivvdf/usd/hivvdf.usd"
decrypt_file(encrypted_filename, usd_path)