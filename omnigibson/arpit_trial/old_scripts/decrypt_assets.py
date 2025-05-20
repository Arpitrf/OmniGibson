from omnigibson.utils.asset_utils import decrypt_file, encrypt_file

encrypted_filename = "/home/arpit/projects/OmniGibson/omnigibson/data/datasets/og_dataset/objects/sink/ojjqku/usd/ojjqku.encrypted.usd"
usd_path = "/home/arpit/projects/OmniGibson/omnigibson/data/datasets/og_dataset/objects/sink/ojjqku/usd/ojjqku.usd"
decrypt_file(encrypted_filename, usd_path)

# # encrypt
# decrypted_filename = "/home/arpit/projects/OmniGibson/omnigibson/data/datasets/og_dataset/objects/fridge/hivvdf/usd/hivvdf_new.usd"
# new_encrypted_filename = "/home/arpit/projects/OmniGibson/omnigibson/data/datasets/og_dataset/objects/fridge/hivvdf/usd/hivvdf_new.encrypted.usd"
# encrypt_file(decrypted_filename, encrypted_filename=new_encrypted_filename)