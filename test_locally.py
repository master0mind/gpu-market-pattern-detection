import json
from main import timeseries_analysis
import io

class MockRequest:
    def __init__(self, json_data=None, files=None, args=None):
        self.json_data = json_data or {}
        self.mock_files = files or {}
        self.args = args or {}
    
    def get_json(self, silent=False):
        return self.json_data
    
    @property
    def files(self):
        if not self.mock_files:
            return {}
        return self.mock_files

def test_function():
    # Load test data
    with open('test_data.json', 'r') as f:
        test_data = json.load(f)
    
    # Create a mock request with JSON data
    mock_request = MockRequest(json_data=test_data)
    
    # Call the function
    response, status_code = timeseries_analysis(mock_request)
    
    # Print the results
    print(f"Status Code: {status_code}")
    print("Response:")
    pretty_response = json.loads(response)
    print(json.dumps(pretty_response, indent=2))
    
    print("\n" + "="*50 + "\n")
    
    # Create a CSV version for file upload testing
    class MockFile:
        def __init__(self, content):
            self.content = content
            
        def read(self):
            return self.content.encode('utf-8')
    
    # Test with file upload
    mock_files = {'file': MockFile(test_data['csv_data'])}
    file_args = {'model_type': 'exponential_smoothing', 'target_column': 'Close', 'forecast_periods': '12'}
    mock_request_with_file = MockRequest(files=mock_files, args=file_args)
    
    # Call the function with file upload
    print("Testing with file upload (Exponential Smoothing):")
    response, status_code = timeseries_analysis(mock_request_with_file)
    
    # Print the results
    print(f"Status Code: {status_code}")
    print("Response:")
    pretty_response = json.loads(response)
    print(json.dumps(pretty_response, indent=2))

if __name__ == "__main__":
    test_function()
