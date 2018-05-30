import os
import collections
from subprocess import call
from tqdm import tqdm

files_list=os.listdir("data/")
print(files_list[0:5])
a = [x.split(".")[0] for x in  files_list]
incomplete_files = [item for item, count in collections.Counter(a).items() if count < 2]
print("before removal incompleted files: ")
print(incomplete_files)
for incomplete_file in incomplete_files:
    os.system("rm ~/github/invoice_ocr/code/ocr/data/{}*".format(incomplete_file))
print("after removal incompleted files: ")
files_list=os.listdir("data/")
a = [x.split(".")[0] for x in  files_list]
incomplete_files = [item for item, count in collections.Counter(a).items() if count < 2]
print(incomplete_files)
