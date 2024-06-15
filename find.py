import os

def find_files(directory, filename):
    result = []
    for root, dirs, files in os.walk(directory):
        if filename in files:
            result.append(os.path.join(root, filename))
    return result

directory = "./"  # 検索を開始するディレクトリのパス
filename = "tf_model.h5"   # 探したいファイル名

files_found = find_files(directory, filename)

if files_found:
    print(f"Found {filename} at the following locations:")
    for file in files_found:
        print(file)
else:
    print(f"{filename} not found in {directory}")