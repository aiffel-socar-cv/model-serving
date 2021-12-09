import os
import zipfile


ROOT_DIR = os.getcwd()
MODELS_DIR = os.path.join(os.getcwd(), "models")
TARGET_DIR = os.path.join(os.getcwd(), "model_store")
# MODELS_DIR = "/Users/tseo/Documents/Github/viai-serving/serve/models"
# TARGET_DIR = "/Users/tseo/Documents/Github/viai-serving/serve/model_store"
DATA_TYPE_LIST = ["dent", "scratch", "spacing"]


class Colors:
    RED = "\033[31m"
    GREEN = "\033[32m"
    RESET = "\033[0m"


for data_type in DATA_TYPE_LIST:
    file_list = []
    model_path = os.path.join(MODELS_DIR, data_type)

    assert os.path.exists(model_path), Colors.RED + f"{data_type} model doesn't exist!!" + Colors.RESET
    # assert os.stat(model_path) == 0, Colors.RED + f"{data_type} folder is empty!!" + Colors.RESET

    for root, dirnames, filenames in os.walk(model_path):
        for file in filenames:
            file_list.append(os.path.join(root, file))

    mar_file = zipfile.ZipFile(os.path.join(TARGET_DIR, f"{data_type}.mar"), "w")
    for file in file_list:
        file_path = model_path + "/"
        mar_file.open(file.replace(file_path, ""), "w").write(open(file, "rb").read())

    mar_file.close()
    print(Colors.GREEN, f"{data_type} model archiving done!", Colors.RESET)
