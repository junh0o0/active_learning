import os
import glob
import fnmatch

def search_file(model_path):
    file_list = []
    for file in os.listdir(model_path):
        if fnmatch.fnmatch(file,'*_mace_swa.model'):
            file_list.append(os.path.join(model_path,file))

    return file_list

def remove_file(model_path):
    for file in glob.glob(os.path.join(model_path,"*.pt")):
        os.remove(file)
