import unittest
import json
import os
from app import app

class TestApp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Cargar el modelo desde la ubicación correcta
        cls.model_path = './mle-intv-main/model.pkl'
        os.environ['MODEL_PATH'] = cls.model_path

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_predict(self):
        payload = [{
            "x1": 0.5,
            "x2": 1.2,
            "x3": "category1",
            "x4": 3.4,
            "x5": 2.1,
            "x6": "category2",
            "x7": "category3"
        }]
        response = self.app.post('/predict', data=json.dumps(payload), content_type='application/json')
        data = json.loads(response.get_data(as_text=True))

        self.assertEqual(response.status_code, 200)
        self.assertIn('predictions', data)
        self.assertIn('probabilities', data)

if __name__ == '__main__':
    unittest.main()
