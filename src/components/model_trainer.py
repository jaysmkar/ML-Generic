import os 
import sys 
from dataclasses import dataclass 
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging 
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            
            logging.info('splitting training and test input data started') 
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            logging.info('splitting training and test input data completed')

            models = {
                "Random Forest" : RandomForestRegressor(),
                "Linear Regression" : LinearRegression(),
                "Decision Tree" : DecisionTreeRegressor() ,
                "Gradient Boosting" : GradientBoostingRegressor(),
                "AdaBoost Regressor" : AdaBoostRegressor(),
                "XGBoost Regressor" : XGBRegressor(),
                "Catboost Classifier" : CatBoostRegressor(),
                "K-Nearest Neighbors" : KNeighborsRegressor()
            }

            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            # getting the best modell from the dictionary 
            best_model_score = max(sorted(model_report.values()))

            # getting the score of the best model from the dictionary 
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
