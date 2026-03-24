import torch


def build_model():
    model = torch.nn.Linear(16, 16)
    model = torch.compile(model)
    model = model.to("cuda")
    return model


def run_once():
    model = build_model()
    x = torch.randn(2, 16).to("cuda")
    return model(x)
