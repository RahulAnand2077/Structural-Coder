import torch


def build_model():
    model = torch.nn.Linear(16, 16)
    model = torch.compile(model)
    return model


def run_once():
    model = build_model()
    x = torch.randn(2, 16)
    y = model(x)
    torch._dynamo.graph_break()
    return y
