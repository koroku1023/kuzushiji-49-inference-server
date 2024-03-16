# refs: https://github.com/rois-codh/kmnist/blob/master/download_data.py

import requests
import os
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = (
        lambda x, total, unit: x
    )  # If tqdm doesn't exist, replace it with a function that does nothing
    print(
        "**** Could not import tqdm. Please install tqdm for download progressbars! (pip install tqdm) ****"
    )

url_list = [
    "http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-imgs.npz",
    "http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-labels.npz",
    "http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-imgs.npz",
    "http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-labels.npz",
    "http://codh.rois.ac.jp/kmnist/dataset/k49/k49_classmap.csv",
]


# Download a list of files
def download_list(url_list):
    for url in url_list:
        file_name = url.split("/")[-1]
        path = os.path.join("data/raw/", file_name)
        if not file_name.endswith(".csv"):
            r = requests.get(url, stream=True)
            with open(path, "wb") as f:
                total_length = int(r.headers.get("content-length"))
                print(
                    "Downloading {} - {:.1f} MB".format(
                        path, (total_length / 1024000)
                    )
                )

                for chunk in tqdm(
                    r.iter_content(chunk_size=1024),
                    total=int(total_length / 1024) + 1,
                    unit="KB",
                ):
                    if chunk:
                        f.write(chunk)
        else:
            df = pd.read_csv(url)
            df.to_csv(path, index=False)
    print("All dataset files downloaded!")


download_list(url_list)
