import math
import json
import logging
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report, RocCurveDisplay
import joblib

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuraciones Globales
RANDOM_STATE = 1337

class ModelTraining:
    def __init__(self, data_path, model_path):
        """
        Clase para entrenar un modelo de regresión logística con preprocesamiento.

        Args:
            data_path (str): Ruta al archivo CSV con los datos de entrenamiento.
            model_path (str): Ruta para guardar el modelo entrenado.
        """
        self.data_path = data_path
        self.model_path = model_path
        self.df = None
        self.model = None

    def load_data(self):
        """Carga los datos desde un archivo CSV."""
        try:
            self.df = pd.read_csv(self.data_path)
            logging.info("Datos cargados exitosamente.")
        except Exception as e:
            logging.error(f"Error cargando los datos: {e}")
            raise

    def preprocess_data(self):
        """Preprocesa los datos y define el pipeline de transformación."""
        try:
            df_X = self.df.drop("y", axis=1)
            df_label = self.df["y"]
            
            numeric_features = ["x1", "x2", "x4", "x5"]
            numeric_transformer = Pipeline(
                steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
            )
            
            categorical_features = ["x3", "x6", "x7"]
            categorical_transformer = OneHotEncoder(handle_unknown="ignore")
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, numeric_features),
                    ("cat", categorical_transformer, categorical_features),
                ]
            )
            
            self.model = Pipeline(
                steps=[("preprocessor", preprocessor),
                       ("classifier", LogisticRegression(max_iter=10000))]
            )
            
            return df_X, df_label
        except Exception as e:
            logging.error(f"Error en el preprocesamiento de datos: {e}")
            raise

    def train_model(self, X_train, y_train):
        """Entrena el modelo con los datos de entrenamiento."""
        try:
            self.model.fit(X_train, y_train)
            logging.info("Modelo entrenado exitosamente.")
        except Exception as e:
            logging.error(f"Error entrenando el modelo: {e}")
            raise

    def save_model(self):
        """Guarda el modelo entrenado en un archivo."""
        try:
            joblib.dump(self.model, self.model_path)
            logging.info(f"Modelo guardado exitosamente en {self.model_path}")
        except Exception as e:
            logging.error(f"Error guardando el modelo: {e}")
            raise

    def evaluate_model(self, X_test, y_test):
        """Evalúa el modelo con los datos de prueba."""
        try:
            score = self.model.score(X_test, y_test)
            tprobs = self.model.predict_proba(X_test)[:, 1]
            logging.info(f"Model score: {score:.3f}")
            logging.info("Classification Report:")
            logging.info("\n" + classification_report(y_test, self.model.predict(X_test)))
            logging.info("Confusion Matrix:")
            logging.info("\n" + str(confusion_matrix(y_test, self.model.predict(X_test))))
            logging.info(f"AUC: {roc_auc_score(y_test, tprobs):.3f}")

            # Gráficas
            RocCurveDisplay.from_estimator(estimator=self.model, X=X_test, y=y_test)
            plt.title('ROC Curve')
            plt.show()

            cm = confusion_matrix(y_test, self.model.predict(X_test))
            fig, ax = plt.subplots()
            cax = ax.matshow(cm, cmap=plt.cm.Blues)
            fig.colorbar(cax)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.show()
        except Exception as e:
            logging.error(f"Error evaluando el modelo: {e}")
            raise

    def run(self):
        """Método principal para ejecutar el flujo de trabajo de entrenamiento y evaluación."""
        self.load_data()
        df_X, df_label = self.preprocess_data()
        X_train, X_test, y_train, y_test = train_test_split(df_X, df_label, random_state=RANDOM_STATE)
        self.train_model(X_train, y_train)
        self.save_model()
        self.evaluate_model(X_test, y_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Entrena y evalúa un modelo de regresión logística.')
    parser.add_argument('--data_path', type=str, required=True, help='Ruta al archivo CSV con los datos de entrenamiento.')
    parser.add_argument('--model_path', type=str, required=True, help='Ruta para guardar el modelo entrenado.')

    args = parser.parse_args()

    trainer = ModelTraining(data_path=args.data_path, model_path=args.model_path)
    trainer.run()