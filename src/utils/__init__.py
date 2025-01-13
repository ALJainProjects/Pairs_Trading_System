# utils/__init__.py

"""
Utilities Module

This package contains various utility functions and classes that support
different aspects of the Equity Pair Trading Research Project, including
performance metrics, visualization tools, data validation, and parallel training
facilities.
"""

# Importing necessary functions and classes for easier access
from .metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_calmar_ratio,
    calculate_beta,
    calculate_alpha
)
from .visualization import (
    plot_equity_curve,
    plot_performance_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_training_history
)
from .validation import (
    check_missing_values,
    check_data_types,
    check_outliers,
    validate_dataframe
)
from .parallel_training import (
    train_models_in_parallel,
    parallel_grid_search
)
