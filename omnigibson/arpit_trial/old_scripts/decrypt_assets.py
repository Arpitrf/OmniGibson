from omnigibson.utils.asset_utils import decrypt_file, encrypt_file

# encrypted_filename = "/home/arpit/projects/OmniGibson/omnigibson/data/datasets/og_dataset/objects/fridge/hivvdf/usd/hivvdf.encrypted.usd"
# usd_path = "/home/arpit/projects/OmniGibson/omnigibson/data/datasets/og_dataset/objects/fridge/hivvdf/usd/hivvdf.usd"
# decrypt_file(encrypted_filename, usd_path)

# encrypt
decrypted_filename = "/home/arpit/projects/OmniGibson/omnigibson/data/datasets/og_dataset/objects/fridge/hivvdf/usd/hivvdf_new.usd"
new_encrypted_filename = "/home/arpit/projects/OmniGibson/omnigibson/data/datasets/og_dataset/objects/fridge/hivvdf/usd/hivvdf_new.encrypted.usd"
encrypt_file(decrypted_filename, encrypted_filename=new_encrypted_filename)