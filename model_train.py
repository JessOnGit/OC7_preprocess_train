"""
Model training script
- Scoring functions (mlflow version)
- Preprocess Data (call to functions in data_preprocessing script)
- Reformat file for LightGBM
- Split (80/20)
- Pipeline train (SMOTE + LightGBM Classifier)
- Log results and model in Mlflow (2 run lines :
    --  1st = model with env and params (to be registered for predict usage)
    -- 2nd = pipeline performances, including SMOTE step

v2 : add a shap value explainer + clients scores archiving for comparison in web app
"""

import re
from data_preprocessing import *
import mlflow
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
import shap
import joblib


# Scorer "MlFlow evaluation"
def score_metier(eval_df, builtin_metrics):
    tn, fp, fn, tp = metrics.confusion_matrix(eval_df['target'], eval_df['prediction']).ravel()
    score_false = 10 * fn + fp
    score = score_false / (tn + fp + fn + tp)
    return score


# Customs metrics declaration mlflow
score_metier_mlf = mlflow.models.make_metric(eval_fn=score_metier,
                                             greater_is_better=False)


# Function evaluation, to be called after model fit
def eval_fn(model):
    mlflow.sklearn.log_model(model, 'model')
    model_uri = mlflow.get_artifact_uri('model')
    result = mlflow.evaluate(
        model=model_uri,
        data=eval_data,
        targets='target',
        model_type='classifier',
        evaluators=['default'],
        custom_metrics=[score_metier_mlf],
    )


# Evaluation run (hors gscv > for GSCV versions, see modelisation_and_feature-importance notebook)
def evaluation_run(run_name, model):
    with mlflow.start_run(run_name=run_name) as run:
        eval_fn(model)


# Preprocessing
train_preprocessed = preprocess(data_path='../data/', entry_file='application_train.csv', is_training_file=True,
                                with_NA_treatment=True, pc_NA_max=0.3)

# Reformat and round X values
X = train_preprocessed.drop('TARGET', axis=1).round(2)
# Reformat columns pour LightGBM
X = X.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
y = train_preprocessed['TARGET'].tolist()

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y, random_state=123)

# Evaluation dataframe for MLflow
eval_data = X_test.copy()
eval_data['target'] = y_test

# Pipeline for SMOTE synthetic values generation in train set before LGBMClassif
steps = [('oversampling', SMOTE()), ('model', LGBMClassifier())]
lgbm = Pipeline(steps=steps)

# Set Params
params = {
    'oversampling__k_neighbors': 30,
    'model__num_iterations': 500,
    'model__min_child_samples': 50,
    'model__reg_alpha': 0.5,
    'model__reg_lambda': 0.5
}
lgbm.set_params(**params)

# MlFlow Autolog
mlflow.lightgbm.autolog()

# Train
lgbm.fit(X_train, y_train)

# Evaluate & Log (NB : 2 run lines : 1 model params logged, 2 Pipeline results, including SMOTE req)
# Register model from Run line 1
evaluation_run('LGBM_rounded_values', lgbm)


# --- SHAP values recording
model = lgbm['model']
explainer = shap.TreeExplainer(model)
data = X_train.reset_index().drop('SK_ID_CURR', axis=1)
sv = explainer(data)
# shap values dataframe
shap_df = pd.DataFrame(data=sv[:, :, 1].values, columns=data.columns)
# score calculation
base_value = sv[:, :, 1][0].base_values
scores_clients = shap_df.sum(axis=1).apply(lambda x : x + base_value)
# indexes of pos and neg clients
data['target'] = y_train
idxs_0 = data.loc[data['target'] == 0].index.values.tolist()
idxs_1 = data.loc[data['target'] == 1].index.values.tolist()
# scores for each group
scores_0 = scores_clients[idxs_0]
scores_1 = scores_clients[idxs_1]
# csv export in refs file
scores_1.to_csv('refs/scores1.csv', index=False)
scores_0.to_csv('refs/scores0.csv', index=False)

# Save explainer for further usage
save_directory = 'refs/shap_explainer.sav'
joblib.dump(explainer, save_directory)


# --- Record min, max, mean, median for X, X target 0 and X target 1
# min, max, mean, median for X, X target 0 and X target 1

def min_max_values(X):
    cols, mini, maxi, mean, median = [], [], [], [], []
    for col in X.columns:
        cols.append(col)
        mini.append(X[col].min())
        maxi.append(X[col].max())
        mean.append(X[col].mean())
        median.append(X[col].median())
    return cols, mini, maxi, mean, median


X['target'] = y
X0 = X.loc[X.target == 0, :]
X1 = X.loc[X.target == 1, :]
X.drop('target', axis=1, inplace=True)
X0.drop('target', axis=1, inplace=True)
X1.drop('target', axis=1, inplace=True)

cols, mini, maxi, mean, median = min_max_values(X)
cols0, mini0, maxi0, mean0, median0 = min_max_values(X0)
cols1, mini1, maxi1, mean1, median1 = min_max_values(X1)

min_max_mean_med = pd.DataFrame({'features': cols,
                                 'min': mini,
                                 'max': maxi,
                                 'mean': mean,
                                 'median': median,
                                 'min0': mini0,
                                 'max0': maxi0,
                                 'mean0': mean0,
                                 'median0': median0,
                                 'min1': mini1,
                                 'max1': maxi1,
                                 'mean1': mean1,
                                 'median1': median1
                                 })

min_max_mean_med.to_csv('refs/min_max_mean_med.csv', index=False)


# Record SAMPLE X_test and y_test for simulation (same number of TP, FP, TN, FN)
def confusion_data_test(X_test, y_test):
    y_pred = model.predict(X_test)
    data_test = X_test.copy()
    data_test['true'] = y_test
    data_test['pred'] = y_pred
    def confusion(x,y) :
        conf = ''
        if (x,y) == (0,0):
            conf = 'tn'
        elif (x,y) == (0,1):
            conf = 'fp'
        elif (x,y) == (1,1):
            conf = 'tp'
        elif (x,y) == (1,0):
            conf = 'fn'
        return conf
    data_test['conf'] = data_test.apply(lambda row: confusion(row['true'], row['pred']), axis=1)
    return data_test


def conf_sample(X_test, y_test, n=500):
    data_test = confusion_data_test(X_test, y_test)
    n_maxi = data_test.conf.value_counts().min()
    if n_maxi < n:
        n = n_maxi
    data_test_sample = data_test.loc[data_test.conf == 'tn'].sample(n)
    for conf in ['fp', 'tp', 'fn']:
        conf_df = data_test.loc[data_test.conf == conf].sample(n)
        data_test_sample = pd.concat((data_test_sample, conf_df), axis=0)
    y = data_test_sample.true
    X = data_test_sample.drop(['true', 'pred', 'conf'], axis=1)
    return X, y


Xtest, ytest = conf_sample(X_test, y_test, n=500)
Xtest.to_csv('data_test_clean/X_test_sample.csv')
ytest.to_csv('data_test_clean/y_test_sample.csv', index=False)

# # --- Record COMPLETE X_test and y_test for simulations
# X_test.to_csv('data_test_clean/X_test.csv')
# y_test.rename({'0' : 'true'}, axis=1)
# pd.DataFrame(y_test).to_csv('data_test_clean/y_test.csv', index=False)