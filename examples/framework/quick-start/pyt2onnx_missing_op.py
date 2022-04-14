import torch

import model_navigator as nav


def dataloader():
    yield torch.randn((1, 1, 10))


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        _layer_size = 23
        self.net = torch.nn.Sequential(torch.nn.Linear(10, 100), torch.nn.AdaptiveMaxPool1d(_layer_size))

    def forward(self, x):
        return self.net(x)


model = MyModule()

package = nav.torch.export(
    model=model,
    dataloader=dataloader,
    # atol=0.01,
    # rtol=0.01,
    override_workdir=True,
)

exported_model = package.get_model(format=nav.Format.TORCHSCRIPT_TRACE)

results = exported_model(next(dataloader()))
