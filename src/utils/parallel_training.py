"""
Parallel Training Module

Enhances computational efficiency for model training tasks by using multiprocessing.
"""

import multiprocessing
from multiprocessing import Pool
from typing import List, Tuple, Callable, Any, Dict
from functools import partial
from config.logging_config import logger


def train_model(model_func: Callable, *args, **kwargs) -> Any:
    """
    Wrapper for parallel execution of a model training function.

    Args:
        model_func (Callable): A function that returns a trained model or output.
        *args: Positional arguments for 'model_func'.
        **kwargs: Keyword arguments for 'model_func'.

    Returns:
        Any: The trained model or relevant result from 'model_func', or None on error.
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
    Train multiple models_data in parallel, each with potentially different functions and arguments.

    Args:
        model_funcs (List[Callable]): List of callables to train models_data.
        args_list (List[Tuple]): Positional arguments for each model_func.
        kwargs_list (List[Dict]): Keyword arguments for each model_func.
        n_jobs (int): Number of processes to use. Defaults to all CPU cores.

    Returns:
        List[Any]: List of model results. Each entry corresponds to the result
                   of calling 'model_funcs[i]' with 'args_list[i]' and 'kwargs_list[i]'.
    """
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count()
    logger.info(f"Training {len(model_funcs)} models_data in parallel with {n_jobs} processes.")

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

    Args:
        model_train_func (Callable): A function that trains a model given parameters and data, returning (model, metrics).
        param_grid (List[Dict]): Each dict is a param set to try.
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        n_jobs (int): Number of processes. Defaults to all cores.

    Returns:
        List[Dict]: Each item has {'params': param_dict, 'metrics': metrics_or_None}.
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
