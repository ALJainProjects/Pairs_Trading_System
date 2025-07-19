import multiprocessing
from multiprocessing import Pool
from typing import List, Tuple, Callable, Any, Dict
from config.logging_config import logger


def train_model(model_func: Callable, *args, **kwargs) -> Any:
    """
    Wrapper for parallel execution of a model training function.
    """
    try:
        model = model_func(*args, **kwargs)
        logger.debug(f"Model function '{model_func.__name__}' trained successfully.")
        return model
    except Exception as e:
        logger.error(f"Error training model {model_func.__name__}: {e}")
        return None


def train_models_in_parallel(model_funcs: List[Callable],
                             args_list: List[Tuple],
                             kwargs_list: List[Dict],
                             n_jobs: int = None) -> List[Any]:
    """
    Train multiple models in parallel.
    """
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count()
    logger.info(f"Training {len(model_funcs)} models in parallel with {n_jobs} processes.")

    tasks = zip(model_funcs, args_list, kwargs_list)
    with Pool(processes=n_jobs) as pool:
        results = pool.starmap(train_model, tasks)

    logger.info("Parallel model training completed.")
    return results


def parallel_grid_search(model_train_func: Callable,
                         param_grid: List[Dict],
                         X_train: Any,
                         y_train: Any,
                         X_val: Any,
                         y_val: Any,
                         n_jobs: int = None) -> List[Dict]:
    """
    Perform a parallelized grid search over a list of parameter dictionaries.
    """
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count()

    logger.info(f"Starting parallel grid search with {len(param_grid)} parameter sets on {n_jobs} processes.")

    def evaluate_params(params: Dict) -> Dict:
        try:
            model, metrics = model_train_func(X_train, y_train, X_val, y_val, **params)
            logger.debug(f"Parameters {params} => metrics {metrics}.")
            return {'params': params, 'metrics': metrics}
        except Exception as e:
            logger.error(f"Error evaluating params {params}: {e}")
            return {'params': params, 'metrics': None}

    with Pool(processes=n_jobs) as pool:
        results = pool.map(evaluate_params, param_grid)

    logger.info("Parallel grid search completed.")
    return results