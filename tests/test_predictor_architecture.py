import torch.nn as nn

from streamlit_app.services.raw_late_fusion_predictor import build_convnext_base


def test_convnext_head_matches_training_checkpoint_layout():
    model = build_convnext_base(num_classes=27)

    assert isinstance(model.classifier[2], nn.Sequential)
    assert isinstance(model.classifier[2][0], nn.LayerNorm)
    assert isinstance(model.classifier[2][1], nn.Dropout)
    assert isinstance(model.classifier[2][2], nn.Linear)
    assert model.classifier[2][2].out_features == 27
    assert "classifier.2.2.weight" in model.state_dict()
