import numpy as np

from longtail.data import img_num_per_cls, make_synthetic_cifar100lt_arrays
from longtail.metrics import compute_metrics
from longtail.train_classical import extract_hog_features_from_arrays


def test_exp_long_tail_counts_are_monotonic():
    counts = img_num_per_cls(cls_num=100, max_num=500, imb_type="exp", imb_ratio=100)
    assert counts[0] == 500
    assert counts[-1] == 5
    assert all(a >= b for a, b in zip(counts, counts[1:]))


def test_step_long_tail_counts_have_two_plateaus():
    counts = img_num_per_cls(cls_num=100, max_num=500, imb_type="step", imb_ratio=100)
    assert counts[:50] == [500] * 50
    assert counts[50:] == [5] * 50


def test_synthetic_arrays_and_metrics_are_well_formed():
    arrays = make_synthetic_cifar100lt_arrays(
        cls_num=10,
        max_train=8,
        max_test=3,
        imb_type="exp",
        imb_ratio=10,
        seed=1,
    )
    x_train, y_train, x_bal, y_bal, _, _, counts, _, _ = arrays
    assert x_train.shape[1:] == (32, 32, 3)
    assert len(counts) == 10

    x_hog, y_hog = extract_hog_features_from_arrays(x_train, y_train, max_items=20, desc="test hog")
    assert x_hog.shape[0] == y_hog.shape[0]
    assert x_hog.shape[1] > 0

    random_scores = np.random.default_rng(1).normal(size=(len(y_bal), 10))
    metrics = compute_metrics(y_bal, y_score=random_scores, cls_num_list=counts, num_classes=10)
    assert 0.0 <= metrics["top1"] <= 1.0
    assert "per_class_acc" in metrics
