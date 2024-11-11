from omnigibson.utils.asset_utils import decrypt_file

encrypted_filename = "/home/arpit/test_projects/OmniGibson/omnigibson/data/og_dataset/objects/fridge/dszchb/usd/dszchb.encrypted.usd"
usd_path = "/home/arpit/test_projects/OmniGibson/omnigibson/data/og_dataset/objects/fridge/dszchb/usd/dszchb.usd"
decrypt_file(encrypted_filename, usd_path)