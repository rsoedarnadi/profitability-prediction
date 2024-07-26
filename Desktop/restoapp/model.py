import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from sklearn.model_selection import RandomizedSearchCV
import catboost as cb
from sklearn.metrics import f1_score, make_scorer
from imblearn.over_sampling import SMOTE as smote
from collections import Counter
from sklearn.model_selection import train_test_split

df = pd.read_csv('restaurant_profitability.csv')

label_encoder = LabelEncoder()
scaler = StandardScaler()

#initialize encoder and scaler
profitability_encoder = LabelEncoder()
ingredients_encoder = LabelEncoder()
menu_category_encoder = LabelEncoder()
menu_item_encoder = LabelEncoder()
scaler = StandardScaler()

#standardize the price data values and encode categorical variables
df['EnProfitability'] = profitability_encoder.fit_transform(df['Profitability'])
df['EnIngredients'] = ingredients_encoder.fit_transform(df['Ingredients'])
df['EnMenuCategory'] = menu_category_encoder.fit_transform(df['MenuCategory'])
df['EnMenuItem'] = menu_item_encoder.fit_transform(df['MenuItem'])
df['ScPrice'] = scaler.fit_transform(df[['Price']])

joblib.dump(scaler, 'scaler.pkl')
joblib.dump(ingredients_encoder, 'ingredients_encoder.pkl')
joblib.dump(menu_category_encoder, 'menu_category_encoder.pkl')
joblib.dump(menu_item_encoder, 'menu_item_encoder.pkl')
joblib.dump(profitability_encoder, 'profitability_encoder.pkl')

X_colnames = ['ScPrice', 'EnIngredients', 'EnMenuCategory', 'EnMenuItem']
X = df[X_colnames]
Y = df['EnProfitability']

sm = smote(sampling_strategy='not majority')
X, Y = sm.fit_resample(X, Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
del X, Y

custom_scorer=make_scorer(f1_score, average='weighted')

param_grid = {
    'iterations': [50, 100],
    'depth': [4, 6, 10],
    'learning_rate': [1, 0.8, 0.6, 0.4],
    'l2_leaf_reg': [1, 3, 5, 7, 9]
}

# Set up the RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator= cb.CatBoostClassifier(verbose=0),
    param_distributions=param_grid,
    n_iter=50,  # Number of parameter settings that are sampled
    scoring=custom_scorer,  # Metric to evaluate the models
    cv=5,  # Number of cross-validation folds
    verbose=1,
    n_jobs=-1,  # Use all available cores
    random_state=42  # Set a random state for reproducibility
)

# Fit the RandomizedSearchCV object to your data
random_search.fit(X_train, y_train)

# Get the best parameters and model
best_params = random_search.best_params_
best_model = random_search.best_estimator_

joblib.dump(best_model, 'catboost_model.pkl')