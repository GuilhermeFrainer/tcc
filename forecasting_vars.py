# Parâmetros gerais
TEST_SIZE = 79
LINEAR_MODELS = ['ridge', 'lasso'] # Modelos cujos dados precisam ser normalizados
WINDOW_LENGTH = 12
N_JOBS = 4
FIGSIZE = (10, 5)

# Parâmetros de modelos
lasso = {
    'max_iter': 2000,
    'n_jobs': N_JOBS
}

rf = {
    'n_estimators': 1000,
    'max_features': 1.0,
    'n_jobs': N_JOBS,
    'criterion': 'absolute_error',
}

lgbm = {
    'n_jobs': N_JOBS,
    'verbose': -1,
    'force_col_wise': True,
    'n_estimators': 1000,
    'learning_rate': 0.01,
    'objective': 'regression_l1',
}