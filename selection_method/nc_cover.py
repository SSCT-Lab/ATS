#####cover
from selection_method.necov_method.neural_cov import CovInit, CovRank



def get_cov_initer(X_train, Y_train, data_name, model_name):
    params = {
        "data_name": data_name,
        "model_name": model_name
    }
    cov_initer = CovInit(X_train, Y_train, params)
    return cov_initer




