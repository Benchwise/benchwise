from benchwise import load_dataset
import os

# Define the path to the dummy data file
# Assuming data.json is in the same directory as this script for testing purposes
data_file_path = "data.json"

def test_load_dataset_from_json():
    # Load the dataset
    dataset = load_dataset(data_file_path)

    # Assertions to verify the loaded dataset's content
    assert dataset is not None, "Dataset should not be None"
    assert len(dataset.data) == 2, "Dataset should contain 2 items"
    assert dataset.prompts == ["What is the capital of France?", "Who wrote 'Hamlet'?"]
    assert dataset.references == ["Paris", "William Shakespeare"]

    print("Successfully loaded dataset and assertions passed!")

if __name__ == "__main__":
    # Create a dummy data.json file for testing if it doesn't exist
    if not os.path.exists(data_file_path):
        with open(data_file_path, "w") as f:
            f.write("""[
  {
    "prompt": "What is the capital of France?",
    "reference": "Paris"
  },
  {
    "prompt": "Who wrote 'Hamlet'?",
    "reference": "William Shakespeare"
  }
]""")
        print(f"Created dummy {data_file_path} for testing.")

    test_load_dataset_from_json()

    # Clean up the dummy file after test (optional, but good practice)
    # os.remove(data_file_path)
    # print(f"Removed dummy {data_file_path}.")
