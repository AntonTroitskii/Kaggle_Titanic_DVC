from pathlib import Path
import shutil
import pickle


def delete_folder(path: Path):
    if path.exists() and path.is_dir():
        try:
            shutil.rmtree(path)
        except Exception as ex:
            print(ex)


def delete_files_in_folder(path: Path):
    if path.exists() and path.is_dir():
        for item in path.iterdir():
            if item.is_file():
                item.unlink()


def clear_all(dir_path: Path, ex_path):
    for item in dir_path.iterdir():
        if item not in ex_path:
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            elif item.is_file():
                item.unlink()

def make_folder(fpath: Path):
    if fpath.exists() and fpath.is_dir():
        delete_files_in_folder(fpath)
    else:
        fpath.mkdir(parents=False, exist_ok=True)
        
        
def load_X_y_from_pkl(path: Path):
    with open(path, 'rb') as f:
        train_data = pickle.load(f)        
        
    X = train_data.iloc[:, :-1]
    y = train_data.iloc[:,-1]
    
    return X, y

