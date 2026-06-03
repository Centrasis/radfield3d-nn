import torch
import sys
import os
sys.path.append(os.path.join(os.path.basename(__file__), ".."))
from metrics import HistogramOverlapAccuracy, TotalVariationDistanceAccuracy, CosineSimilarityAccuracy


def test_overlap_accuracy_equality():
    accuracy = HistogramOverlapAccuracy()
    hist1 = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1])
    hist1 = hist1 / hist1.sum()
    hist1 = hist1.unsqueeze(0)
    hist2 = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1])
    hist2 = hist2 / hist2.sum()
    hist2 = hist2.unsqueeze(0)
    assert accuracy(hist1, hist2) == 1.0, "Accuracy should be 1.0"


def test_overlap_accuracy_disjunctivity():
    accuracy = HistogramOverlapAccuracy()
    hist1 = torch.tensor([0.5, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0])
    hist1 = hist1 / hist1.sum()
    hist1 = hist1.unsqueeze(0)
    hist2 = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.2, 0.3, 0.5])
    hist2 = hist2 / hist2.sum()
    hist2 = hist2.unsqueeze(0)
    assert accuracy(hist1, hist2) == 0.0, "Accuracy should be 0.0"


def test_overlap_with_zeros():
    accuracy = HistogramOverlapAccuracy()
    hist1 = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1])
    hist1 = hist1 / hist1.sum()
    hist1 = hist1.unsqueeze(0)
    hist2 = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    hist2 = hist2.unsqueeze(0)
    assert accuracy(hist1, hist2) == 0.0, "Accuracy should be 0.0"


def test_half_overlap():
    accuracy = HistogramOverlapAccuracy()
    hist1 = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1])
    hist1 = hist1 / hist1.sum()
    hist1 = hist1.unsqueeze(0)
    hist2 = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.3, 0.2, 0.1])
    hist2 = hist2 / hist2.sum()
    hist2 = hist2.unsqueeze(0)
    val = accuracy(hist1, hist2)
    assert val < 0.5 and val > 0.1, "Accuracy should be between 0.1 and 0.5"


def test_totalvar_accuracy_equality():
    accuracy = TotalVariationDistanceAccuracy()
    hist1 = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1])
    hist1 = hist1 / hist1.sum()
    hist1 = hist1.unsqueeze(0)
    hist2 = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1])
    hist2 = hist2 / hist2.sum()
    hist2 = hist2.unsqueeze(0)
    assert accuracy(hist1, hist2) == 1.0, "Accuracy should be 1.0"


def test_totalvar_accuracy_disjunctivity():
    accuracy = TotalVariationDistanceAccuracy()
    hist1 = torch.tensor([0.5, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0])
    hist1 = hist1 / hist1.sum()
    hist1 = hist1.unsqueeze(0)
    hist2 = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.2, 0.3, 0.5])
    hist2 = hist2 / hist2.sum()
    hist2 = hist2.unsqueeze(0)
    assert accuracy(hist1, hist2) == 0.0, "Accuracy should be 0.0"


def test_totalvar_with_zeros():
    accuracy = TotalVariationDistanceAccuracy()
    hist1 = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1])
    hist1 = hist1 / hist1.sum()
    hist1 = hist1.unsqueeze(0)
    hist2 = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    hist2 = hist2.unsqueeze(0)
    assert accuracy(hist1, hist2) == 0.5, "Accuracy should be 0.5"


def test_totalvar_half_overlap():
    accuracy = TotalVariationDistanceAccuracy()
    hist1 = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1])
    hist1 = hist1 / hist1.sum()
    hist1 = hist1.unsqueeze(0)
    hist2 = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.3, 0.2, 0.1])
    hist2 = hist2 / hist2.sum()
    hist2 = hist2.unsqueeze(0)
    val = accuracy(hist1, hist2)
    assert val < 0.5 and val > 0.1, "Accuracy should be between 0.1 and 0.5"


def test_cosine_accuracy_equality():
    accuracy = CosineSimilarityAccuracy()
    hist1 = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1])
    hist1 = hist1 / hist1.sum()
    hist1 = hist1.unsqueeze(0)
    hist2 = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1])
    hist2 = hist2 / hist2.sum()
    hist2 = hist2.unsqueeze(0)
    assert accuracy(hist1, hist2) == 1.0, "Accuracy should be 1.0"


def test_cosine_accuracy_disjunctivity():
    accuracy = CosineSimilarityAccuracy()
    hist1 = torch.tensor([0.5, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0])
    hist1 = hist1 / hist1.sum()
    hist1 = hist1.unsqueeze(0)
    hist2 = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.2, 0.3, 0.5])
    hist2 = hist2 / hist2.sum()
    hist2 = hist2.unsqueeze(0)
    accuracy = accuracy(hist1, hist2)
    assert accuracy == 0.0, "Accuracy should be 0.0 and not " + str(accuracy)


def test_cosine_with_zeros():
    accuracy = CosineSimilarityAccuracy()
    hist1 = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1])
    hist1 = hist1 / hist1.sum()
    hist1 = hist1.unsqueeze(0)
    hist2 = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    hist2 = hist2.unsqueeze(0)
    accuracy = accuracy(hist1, hist2)
    assert accuracy == 0.0, "Accuracy should be 0.0 and not " + str(accuracy)


def test_cosine_half_overlap():
    accuracy = CosineSimilarityAccuracy()
    hist1 = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1])
    hist1 = hist1 / hist1.sum()
    hist1 = hist1.unsqueeze(0)
    hist2 = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.3, 0.2, 0.1])
    hist2 = hist2 / hist2.sum()
    hist2 = hist2.unsqueeze(0)
    accuracy = accuracy(hist1, hist2)
    assert accuracy < 0.6 and accuracy > 0.1, "Accuracy should be between 0.1 and 0.5, but is " + str(accuracy)
