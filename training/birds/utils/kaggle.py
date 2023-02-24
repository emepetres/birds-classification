import os
from pathlib import Path
from zipfile import ZipFile
from kaggle import api


def running_on_kaggle() -> bool:
    """
    Checks if script is running on kaggle
    :return: true if the script is running on kaggle, false otherwise
    """
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE", ""):
        return True
    else:
        return False


def download_competition_data(competition: str, input_path: str | Path) -> None:
    """
    Downloads data from kaggle competition only if input folder is empty
    :param comptetition: string with the competition name id of kaggle
    :param input_path: path of the input folder
    """
    data_path = Path(input_path)
    if not data_path.exists():
        data_path.mkdir(parents=True)
    if not any(data_path.iterdir()):
        api.competition_download_cli(competition, path=data_path)
        with ZipFile(data_path / (competition + ".zip"), "r") as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall(path=data_path)
        os.remove(data_path / (competition + ".zip"))

        print(os.listdir(data_path))


def download_dataset(owner: str, dataset: str, input_path: str | Path) -> None:
    """
    Downloads data from kaggle competition only if input folder is empty
    :param comptetition: string with the competition name id of kaggle
    :param input_path: path of the input folder
    """
    data_path = Path(input_path)
    if not data_path.exists():
        data_path.mkdir(parents=True)
    if not any(data_path.iterdir()):
        api.dataset_download_files(f"{owner}/{dataset}", path=data_path)
        with ZipFile(data_path / (dataset + ".zip"), "r") as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall(path=data_path)
        os.remove(data_path / (dataset + ".zip"))

        print(os.listdir(data_path))
