import multiprocessing
from multiprocessing import Pool
from typing import List, Tuple, Callable, Any, Dict, Optional
from tqdm import tqdm # Import tqdm for progress bars
from config.logging_config import logger


# Define a type alias for clarity if needed, though simple tuples are also fine.
# Task = Tuple[Callable, Tuple, Dict]

def _train_single_model_task(model_func: Callable, args_tuple: Tuple, kwargs_dict: Dict) -> Any:
    """
    Internal helper function for parallel model training.
    Executes a single model training function with its arguments and keyword arguments.
    Designed to be picklable (e.g., top-level function).
    """
    try:
        model = model_func(*args_tuple, **kwargs_dict)
        logger.debug(f"Model function '{model_func.__name__}' trained successfully in parallel task.")
        return model
    except Exception as e:
        logger.error(f"Error training model '{model_func.__name__}' in parallel task: {e}")
        return None # Return None on failure


def train_models_in_parallel(model_funcs: List[Callable],
                             args_list: List[Tuple],
                             kwargs_list: List[Dict],
                             n_jobs: Optional[int] = None) -> List[Any]:
    """
    Train multiple models in parallel using a multiprocessing Pool.

    Args:
        model_funcs (List[Callable]): A list of model training functions to execute.
                                      Each function should accept positional and keyword arguments.
        args_list (List[Tuple]): A list of tuples, where each tuple contains positional arguments
                                 for the corresponding model_func. Must have same length as model_funcs.
        kwargs_list (List[Dict]): A list of dictionaries, where each dict contains keyword arguments
                                  for the corresponding model_func. Must have same length as model_funcs.
        n_jobs (Optional[int]): Number of processes to use.
                                If None or -1, uses `multiprocessing.cpu_count()`.
                                If > 0, uses that many processes.

    Returns:
        List[Any]: A list of results from each model training function. None for failed tasks.
                   The order of results corresponds to the order of input functions.
    """
    if len(model_funcs) != len(args_list) or len(model_funcs) != len(kwargs_list):
        raise ValueError("model_funcs, args_list, and kwargs_list must have the same length.")

    if n_jobs is None or n_jobs < 1:
        n_jobs_actual = multiprocessing.cpu_count()
    else:
        n_jobs_actual = n_jobs

    logger.info(f"Training {len(model_funcs)} models in parallel with {n_jobs_actual} processes.")

    # Prepare tasks for starmap: each task is a tuple (model_func, args_tuple, kwargs_dict)
    tasks = [(mf, args, kwargs) for mf, args, kwargs in zip(model_funcs, args_list, kwargs_list)]

    with Pool(processes=n_jobs_actual) as pool:
        # Use starmap to unpack the arguments for _train_single_model_task
        # Wrap with tqdm for a progress bar
        results = list(tqdm(pool.starmap(_train_single_model_task, tasks), total=len(tasks), desc="Parallel Training"))

    logger.info("Parallel model training completed.")
    return results


def _evaluate_single_param_set_task(params: Dict, model_train_func: Callable, X_train: Any, y_train: Any, X_val: Any, y_val: Any) -> Dict:
    """
    Internal helper function for parallel grid search.
    Evaluates a single set of parameters by training a model and collecting its metrics.
    Designed to be picklable (e.g., top-level function).
    """
    try:
        # model_train_func is expected to return (model, metrics_dict)
        # Or just metrics_dict if the model object itself is not needed back.
        model, metrics = model_train_func(X_train, y_train, X_val, y_val, **params)
        logger.debug(f"Parameters {params} => metrics {metrics}.")
        return {'params': params, 'metrics': metrics}
    except Exception as e:
        logger.error(f"Error evaluating params {params}: {e}")
        return {'params': params, 'metrics': None}


def parallel_grid_search(model_train_func: Callable,
                         param_grid: List[Dict],
                         X_train: Any,
                         y_train: Any,
                         X_val: Any,
                         y_val: Any,
                         n_jobs: Optional[int] = None) -> List[Dict]:
    """
    Perform a parallelized grid search over a list of parameter dictionaries.

    Args:
        model_train_func (Callable): A function that trains a model and returns its
                                     trained instance and a dictionary of performance metrics.
                                     Expected signature: `(X_train, y_train, X_val, y_val, **params) -> Tuple[Any, Dict]`.
        param_grid (List[Dict]): A list of dictionaries, where each dictionary represents
                                 a set of parameters to test.
        X_train (Any): Training features.
        y_train (Any): Training targets.
        X_val (Any): Validation features.
        y_val (Any): Validation targets.
        n_jobs (Optional[int]): Number of processes to use.
                                If None or -1, uses `multiprocessing.cpu_count()`.

    Returns:
        List[Dict]: A list of dictionaries, each containing 'params' (the parameter set tested)
                    and 'metrics' (the performance metrics, or None if evaluation failed).
    """
    if not param_grid:
        logger.warning("Parameter grid is empty. Returning empty results.")
        return []

    if n_jobs is None or n_jobs < 1:
        n_jobs_actual = multiprocessing.cpu_count()
    else:
        n_jobs_actual = n_jobs

    logger.info(f"Starting parallel grid search with {len(param_grid)} parameter sets on {n_jobs_actual} processes.")

    # Prepare partial function for `_evaluate_single_param_set_task`
    # This captures the common arguments (model_train_func, data)
    from functools import partial
    partial_evaluate = partial(_evaluate_single_param_set_task,
                               model_train_func=model_train_func,
                               X_train=X_train, y_train=y_train,
                               X_val=X_val, y_val=y_val)

    with Pool(processes=n_jobs_actual) as pool:
        # Use map to apply the partial function to each parameter set.
        # Wrap with tqdm for a progress bar.
        results = list(tqdm(pool.map(partial_evaluate, param_grid), total=len(param_grid), desc="Parallel Grid Search"))

    logger.info("Parallel grid search completed.")
    return results