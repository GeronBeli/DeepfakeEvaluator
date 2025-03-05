import pytest
import torch
import pandas as pd
import os
import tempfile
from unittest.mock import MagicMock, patch
from evaluate import evaluate_model

@pytest.fixture
def mock_model():
    """Creates a mock model with a fake `predict` method."""
    class MockModel:
        def __init__(self):
            pass

        def load_state_dict(self, weights):
            pass  # Pretend to load weights

        def eval(self):
            pass  # Pretend to switch to evaluation mode

        def predict(self, image):
            return 1, 0.85  # Always predict FAKE with confidence 0.85

    return MockModel()

@pytest.fixture
def mock_dataloader():
    """Creates a mock DataLoader that yields fake image data."""
    class MockDataset:
        def __getitem__(self, index):
            return {
                "image": torch.rand((3, 224, 224)),  # Random tensor simulating an image
                "label": 1,  # Fake image label
                "path": f"sample_images/FAKE/image_{index}.jpg"
            }

        def __len__(self):
            return 5  # Simulate a dataset with 5 images

    return torch.utils.data.DataLoader(MockDataset(), batch_size=1, shuffle=False)

@pytest.fixture
def temp_output_csv():
    """Creates a temporary file path for the output CSV."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
        yield temp_file.name
    os.remove(temp_file.name)  # Cleanup after test

@patch("evaluate.load_model_class")
@patch("data.dataloader.get_dataloader")
def test_evaluate_model(mock_get_dataloader, mock_load_model_class, mock_model, mock_dataloader, temp_output_csv):
    """Tests the evaluate function end-to-end."""
    
    # Mock the model loading function to return our mock model
    mock_load_model_class.return_value = lambda: mock_model

    # Mock the DataLoader function to return our fake DataLoader
    mock_get_dataloader.return_value = mock_dataloader

    # Run evaluation
    evaluate_model(
        model_file="mock_model.py",
        model_class_name="MockModel",
        weights_path="weights/mock.pth",
        data_dir="sample_images",
        output_csv=temp_output_csv
    )

    # Verify that the CSV file was created
    assert os.path.exists(temp_output_csv), "Output CSV file was not created"

    # Load the results
    df = pd.read_csv(temp_output_csv)

    # Check if the expected columns exist
    expected_columns = ["FilePath", "CorrectLabel", "PredictedLabel", "PredictedValue"]
    assert all(col in df.columns for col in expected_columns), "Missing columns in CSV output"

    # Check if predictions are consistent with the mock model
    assert all(df["PredictedLabel"] == "FAKE"), "All predictions should be FAKE"
    assert all(df["PredictedValue"] == 0.85), "All predictions should have a confidence of 0.85"

    # Check accuracy calculation
    assert df["CorrectLabel"].equals(df["PredictedLabel"]), "All predictions should be correct"
    assert len(df) == 5, "CSV should have 5 rows"

    print("✅ Evaluation test passed!")

