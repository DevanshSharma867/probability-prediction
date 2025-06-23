import unittest
import pandas as pd
from src.data_preprocessing import load_and_process_data

class TestDataPreprocessing(unittest.TestCase):
    def test_load_and_process_data(self):
        # This is a placeholder test; you can add more detailed tests with sample data
        try:
            df = load_and_process_data("../data/KJS_data.csv")
            self.assertIn('NetFamilyIncome', df.columns)
            self.assertIn('Scholarship', df.columns)
        except Exception as e:
            self.fail(f"load_and_process_data raised an exception: {e}")

if __name__ == "__main__":
    unittest.main()
