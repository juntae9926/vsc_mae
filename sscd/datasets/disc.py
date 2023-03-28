# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path
from typing import Callable, Dict, Optional
from torchvision.datasets.folder import default_loader

from sscd.datasets.image_folder import get_image_paths
from sscd.datasets.isc.descriptor_matching import (
    knn_match_and_make_predictions,
    match_and_make_predictions,
)
from sscd.datasets.isc.io import read_ground_truth
from sscd.datasets.isc.metrics import evaluate, Metrics
import pandas as pd


class DISCEvalDataset:
    """DISC2021 evaluation dataset."""

    SPLIT_REF = 0
    SPLIT_QUERY = 1
    SPLIT_TRAIN = 2

    def __init__(
        self,
        path: str,
        transform: Callable = None,
        include_train: bool = True,
        # Specific paths for each part of the dataset. If not set, inferred from `path`.
        query_path: Optional[str] = None,
        ref_path: Optional[str] = None,
        train_path: Optional[str] = None,
        gt_path: Optional[str] = None,
    ):
        def get_path(override_path, relative_path):
            return override_path if override_path else os.path.join(path, relative_path)

        query_path = os.path.join(query_path, "query")
        ref_path = os.path.join(ref_path, "reference")

        train_path = get_path(train_path, "training") if include_train else None
        gt_path = "./validation_matching.csv"

        anno = pd.read_csv(gt_path)
        query_ids = anno['query_id']
        ref_ids = anno['ref_id']
        self.files = self.get_list(ref_path, query_ids)
        query_files = self.get_list(query_path, ref_ids)

        # self.files = self.read_files(self.refs, self.SPLIT_REF)
        # query_files = self.read_files(query_path, self.SPLIT_QUERY)
        self.files.extend(query_files)
        # self.gt = read_ground_truth(gt_path) # ground_truth pairs

        self.gt = [[anno.iloc[i]['query_id'], anno.iloc[i]['ref_id']] for i in range(len(anno))]
        self.transform = transform

    def __getitem__(self, idx: int):
        filename = self.files[idx] + ".png"
        img = default_loader(filename)
        if self.transform:
            img = self.transform(img)
        sample = {"input": img, "instance_id": idx}
        # sample.update(self.metadata[idx])
        return sample

    def __len__(self):
        return len(self.files)

    @classmethod
    def get_list(cls, path, ids):
        file_list = [os.path.join(path, f) for f in ids]
        return file_list

    @classmethod
    def read_files(cls, path, split):
        files = path
        # files = get_image_paths(path)
        names = [os.path.splitext(os.path.basename(file))[0] for file in files]
        # metadata = [
        #     dict(name=name, split=split, image_num=int(name[1:]), target=-1)
        #     for name in names
        # ]
        return files

    def retrieval_eval(
        self, embedding_array, targets, split, **kwargs # embedding, image number, split
    ) -> Dict[str, float]:
        query_mask = split == self.SPLIT_QUERY
        ref_mask = split == self.SPLIT_REF
        query_ids = targets[query_mask]
        query_embeddings = embedding_array[query_mask, :]
        ref_ids = targets[ref_mask]
        ref_embeddings = embedding_array[ref_mask, :]
        return self.retrieval_eval_splits(
            query_ids, query_embeddings, ref_ids, ref_embeddings, **kwargs
        )

    def retrieval_eval_splits(
        self,
        query_ids,
        query_embeddings,
        ref_ids,
        ref_embeddings,
        use_gpu=False,
        k=10,
        global_candidates=False,
        **kwargs
    ) -> Dict[str, float]:
        query_names = ["Q%05d" % i for i in query_ids]
        ref_names = ["R%06d" % i for i in ref_ids]
        if global_candidates:
            predictions = match_and_make_predictions(
                query_embeddings,
                query_names,
                ref_embeddings,
                ref_names,
                num_results=k * len(query_names),
                ngpu=-1 if use_gpu else 0,
                **kwargs,
            )
        else:
            predictions = knn_match_and_make_predictions(
                query_embeddings,
                query_names,
                ref_embeddings,
                ref_names,
                k=k,
                ngpu=-1 if use_gpu else 0,
                **kwargs,
            )
        results: Metrics = evaluate(self.gt, predictions)
        return {
            "uAP": results.average_precision,
            "accuracy-at-1": results.recall_at_rank1,
            "recall-at-p90": results.recall_at_p90 or 0.0,
        }
