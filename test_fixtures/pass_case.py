import torch


def build_model():
    model = torch.nn.Linear(16, 16)
    model = torch.compile(model)
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model


def run_once():
    model = build_model()
    x = torch.randn(2, 16)
    if torch.cuda.is_available():
        x = x.to("cuda")
    return model(x)
