import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        # super().__init__()
        super(MyModel, self).__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        # Input: 3x224x224
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=(5,3), stride = (1,1), padding=(2,1)),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 24x112x112
            
            nn.Conv2d(in_channels=24, out_channels=62, kernel_size=(5,3), stride = (1,1), padding=(2,1)),
            # nn.BatchNorm2d(62),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 62x56x56
                
            nn.Conv2d(in_channels=62, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 64x28x28
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2),
            # 128x14x14
            
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2),
            # 256x14x14
            
            
            nn.Flatten(),
            nn.Linear(512*28*28, 1000), 
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Linear(1000, num_classes),
            nn.Dropout(dropout),

        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        return self.model(x)


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
