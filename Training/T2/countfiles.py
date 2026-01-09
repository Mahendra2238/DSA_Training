import os

def count_txt_files(root_path=None):
    if root_path is None:
        root_path = os.path.join(os.path.expanduser("~"), "Desktop")
    
    count = 0
    for dirpath, dirnames, filenames in os.walk(root_path):
        #print(dirpath, dirnames, filenames)
        count += sum(1 for file in filenames if file.endswith(".txt"))
    
    return count

print("Number of .txt files:", count_txt_files())
