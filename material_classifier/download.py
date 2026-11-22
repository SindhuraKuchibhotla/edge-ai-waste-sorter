"""Download Kaggle's TrashNet image dataset into the ignored local data folder."""

from pathlib import Path

import kagglehub


def main() -> None:
    destination = Path(__file__).resolve().parent / "data"
    path = kagglehub.dataset_download("feyzazkefe/trashnet", output_dir=str(destination))
    print(f"TrashNet dataset: {path}")


if __name__ == "__main__":
    main()
