import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from typing import Optional, List
from sklearn.model_selection import cross_val_predict, KFold
import statsmodels.api as sm

from .base import CITKTest
from causallearn.utils.cit import register_ci_test, KCI as KCI_test


class KCI(CITKTest):
    """
    Wrapper for the Kernel Conditional Independence (KCI) test from the causal-learn library.
    
    Parameters
    ----------
    data : np.ndarray
        The dataset from which to run the test.
    **kwargs : dict
        Additional keywords for the KCI test. See causal-learn documentation.
    """
    supported_dtypes = {"continuous"}

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.check_cache_method_consistent('kci', "NO SPECIFIED PARAMETERS") # KCI handles its own params
        self.kci_instance = KCI_test(data, **kwargs)

    def _compute(self, X, Y, condition_set=None, **kwargs):
        """
        Performs a Kernel Conditional Independence (KCI) test.

        Parameters
        ----------
        X : int
            The index of the first variable.
        Y : int
            The index of the second variable.
        condition_set : list[int], optional
            A list of indices for the conditioning set. Can be empty.

        Returns
        -------
        p_value : float
            The p-value of the test.

    .. seealso::
        For a detailed explanation of the statistical test, including mathematical
        formulations and assumptions, please refer to the :doc:`/tests/kci_test` guide.

    Examples
    --------
    **Standalone Usage**

    .. code-block:: python

        import numpy as np
        from citk.tests import KCI

        # Generate data with a non-linear relationship: X -> Z -> Y
        n = 500
        X = np.random.randn(n)
        Z = np.cos(X) + np.random.randn(n) * 0.1
        Y = Z**2 + np.random.randn(n) * 0.1
        data = np.vstack([X, Y, Z]).T

        # Initialize the test
        kci_test = KCI(data)

        # Test for unconditional independence (should be dependent)
        p_unconditional = kci_test(0, 1)
        print(f"P-value (unconditional) for X _||_ Y: {p_unconditional:.4f}")

        # Test for conditional independence given Z (should be independent)
        p_conditional = kci_test(0, 1, [2])
        print(f"P-value (conditional) for X _||_ Y | Z: {p_conditional:.4f}")

    .. code-block:: text

        P-value (unconditional) for X _||_ Y: 0.0000
        P-value (conditional) for X _||_ Y | Z: 0.8521

    **Usage with PC Algorithm**

    .. code-block:: python

        from causallearn.search.ConstraintBased.PC import pc
        from citk.tests import KCI
        import numpy as np
        
        # Using the same non-linear data
        n = 200
        X = np.random.randn(n)
        Z = np.cos(X) + np.random.randn(n) * 0.1
        Y = Z**2 + np.random.randn(n) * 0.1
        data = np.vstack([X, Y, Z]).T

        cg = pc(data, alpha=0.05, indep_test='kci')
        print("Estimated Causal Graph:")
        print(cg.G)

    .. code-block:: text

        Estimated Causal Graph:
        Graph Nodes:
        X1;X2;X3

        Graph Edges:
        1. X1 --- X3
        2. X2 --- X3
        """
        return float(self.kci_instance(X, Y, condition_set))

register_ci_test("kci", KCI)


class RandomForest(CITKTest):
    """
    Performs a conditional independence test using Random Forest feature importance.

    Parameters
    ----------
    data : np.ndarray
        The dataset from which to run the test.
    n_estimators : int, optional
        The number of trees in the forest.
    num_permutations : int, optional
        The number of permutations to perform for the permutation test.
    random_state : int, optional
        Seed for the random number generator for reproducibility.
    """
    supported_dtypes = {"continuous", "discrete"}

    def __init__(self, data: np.ndarray, **kwargs):
        super().__init__(data, **kwargs)
        self.df = pd.DataFrame(data)
        self.df.columns = [str(i) for i in range(data.shape[1])]
        self.n_estimators = kwargs.get('n_estimators', 100)
        self.num_permutations = kwargs.get('num_permutations', 100)
        self.random_state = kwargs.get('random_state', None)
        params = f"n_est={self.n_estimators},n_perm={self.num_permutations},seed={self.random_state}"
        self.check_cache_method_consistent('rf', params)

    def _compute(self, X: int, Y: int, condition_set: Optional[List[int]] = None, **kwargs) -> float:
        """
        Performs a conditional independence test using Random Forest feature importance.

        The test measures the feature importance of X in predicting Y, conditioned on Z.
        A permutation test is used to assess the statistical significance of this importance.

        Parameters
        ----------
        X : int
            The index of the first variable.
        Y : int
            The index of the second variable (the target).
        condition_set : list[int], optional
            A list of indices for the conditioning set. Can be empty.

        Returns
        -------
        p_value : float
            The p-value of the test.

    .. seealso::
        For a detailed explanation of the statistical test, including mathematical
        formulations and assumptions, please refer to the :doc:`/tests/random_forest_test` guide.

    Examples
    --------
    **Standalone Usage**

    .. code-block:: python

        import numpy as np
        from citk.tests import RandomForest

        # Generate data with a non-linear relationship: X -> Z -> Y
        n = 500
        X = np.random.randn(n)
        Z = np.sin(X * 2) + np.random.randn(n) * 0.2
        Y = Z**3 + np.random.randn(n) * 0.2
        data = np.vstack([X, Y, Z]).T

        # Initialize the test
        rf_test = RandomForest(data, num_permutations=99, random_state=42)

        # Test for unconditional independence (should be dependent)
        p_unconditional = rf_test(0, 1)
        print(f"P-value (unconditional) for X _||_ Y: {p_unconditional:.4f}")

        # Test for conditional independence given Z (should be independent)
        p_conditional = rf_test(0, 1, [2])
        print(f"P-value (conditional) for X _||_ Y | Z: {p_conditional:.4f}")

    .. code-block:: text

        P-value (unconditional) for X _||_ Y: 0.0100
        P-value (conditional) for X _||_ Y | Z: 0.5400

    **Usage with PC Algorithm**

    .. code-block:: python
    
        from causallearn.search.ConstraintBased.PC import pc
        from citk.tests import RandomForest
        import numpy as np

        n = 200
        X = np.random.randn(n)
        Z = np.sin(X * 2) + np.random.randn(n) * 0.2
        Y = Z**3 + np.random.randn(n) * 0.2
        data = np.vstack([X, Y, Z]).T

        cg = pc(data, alpha=0.05, indep_test='rf', num_permutations=49)
        print("Estimated Causal Graph:")
        print(cg.G)
        
    .. code-block:: text

        Estimated Causal Graph:
        Graph Nodes:
        X1;X2;X3

        Graph Edges:
        1. X1 --- X3
        2. X2 --- X3
        """
        # Define predictor and target variables
        x_name, y_name = str(X), str(Y)
        condition_names = [str(c) for c in condition_set]
        
        predictor_cols = [x_name] + condition_names
        X_df = self.df[predictor_cols]
        y_series = self.df[y_name]
        
        # Determine if it's a classification or regression task
        is_classification = y_series.nunique() <= 10 or y_series.dtype.name == "category"
        
        # --- Conditional Case: Permutation test on feature importance ---
        if condition_names:
            if is_classification:
                model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=-1)

            # 1. Calculate the observed statistic
            model.fit(X_df, y_series)
            importances = model.feature_importances_
            x_index = X_df.columns.get_loc(x_name)
            observed_statistic = importances[x_index]

            # 2. Generate the null distribution by permuting the column of interest
            permuted_statistics = np.zeros(self.num_permutations)
            X_permuted_df = X_df.copy()
            
            for i in range(self.num_permutations):
                # Permute the specific predictor column
                shuffled_x = X_permuted_df[x_name].sample(frac=1, random_state=self.random_state + i if self.random_state is not None else None).values
                X_permuted_df[x_name] = shuffled_x
                
                # Re-train the model on the permuted data
                model.fit(X_permuted_df, y_series)
                permuted_importances = model.feature_importances_
                permuted_statistics[i] = permuted_importances[x_index]

            # 3. Calculate the p-value
            p_value = (np.sum(permuted_statistics >= observed_statistic) + 1) / (self.num_permutations + 1)

        # --- Unconditional Case: Permutation test on R-squared ---
        else:
            if is_classification:
                raise NotImplementedError("Unconditional classification test with RF is not yet implemented.")
            
            model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=-1)
            
            # 1. Observed R-squared
            model.fit(X_df, y_series)
            observed_r2 = model.score(X_df, y_series)
            
            # 2. Null distribution of R-squared from permuted target
            permuted_r2 = np.zeros(self.num_permutations)
            X_permuted_df = X_df.copy()
            for i in range(self.num_permutations):
                shuffled_x = X_permuted_df[x_name].sample(frac=1, random_state=self.random_state + i if self.random_state is not None else None).values
                X_permuted_df[x_name] = shuffled_x
                model.fit(X_permuted_df, y_series)
                permuted_r2[i] = model.score(X_permuted_df, y_series)
                
            p_value = (np.sum(permuted_r2 >= observed_r2) + 1) / (self.num_permutations + 1)
        
        return float(p_value)

register_ci_test("rf", RandomForest)

# Helper functions from instructions.md
def _get_dml_residuals(model, data, x_idx, y_idx, z_idx, cv_folds=5):
    """Generates high-quality DML residuals and normalizes them."""
    X_target = data[:, x_idx]
    Y_target = data[:, y_idx]
    
    if not z_idx:
        Z_features = pd.DataFrame()
    else:
        Z_features = pd.DataFrame(data[:, z_idx], columns=[f'z{i}' for i in range(len(z_idx))])
    
    if Z_features.empty:
        pred_x = np.zeros_like(X_target)
        pred_y = np.zeros_like(Y_target)
    else:
        pred_x = cross_val_predict(model, Z_features, X_target, cv=cv_folds)
        pred_y = cross_val_predict(model, Z_features, Y_target, cv=cv_folds)
    
    U = X_target - pred_x
    V = Y_target - pred_y
    
    # Avoid division by zero if a residual is constant
    u_std = np.std(U)
    v_std = np.std(V)
    U = U / u_std if u_std > 0 else U
    V = V / v_std if v_std > 0 else V
    
    return U, V

def _residual_regression_test(x, y):
    """
    Final-stage p-value via linear regression between residuals.
    """
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    X = sm.add_constant(x, has_constant="add")
    fit = sm.OLS(y, X).fit()

    if len(fit.pvalues) < 2:
        return 1.0
    p_value = float(fit.pvalues[1])
    if np.isnan(p_value):
        return 1.0
    return p_value

def _conformalized_ci_test(
    data, x_idx, y_idx, z_idx, alpha=0.1, cv_folds=5, n_perms=199, quantile_model_factory=None
):
    """Performs the Conformalized Residual Independence Test (CRIT)."""
    X_target, Y_target = data[:, x_idx], data[:, y_idx]
    if not z_idx:
        Z_features = pd.DataFrame()
    else:
        Z_features = pd.DataFrame(data[:, z_idx], columns=[f'z{i}' for i in range(len(z_idx))])

    if Z_features.empty:
        return _residual_regression_test(X_target, Y_target)

    all_indices, all_true_x, all_true_y = np.array([], dtype=int), np.array([]), np.array([])
    all_preds_x_low, all_preds_x_high = np.array([]), np.array([])
    all_preds_y_low, all_preds_y_high = np.array([]), np.array([])
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    if quantile_model_factory is None:
        quantile_model_factory = lambda q: GradientBoostingRegressor(
            loss="quantile",
            alpha=q,
            n_estimators=300,
            learning_rate=0.05,
            random_state=42,
        )

    for train_idx, calib_idx in kf.split(Z_features):
        Z_train, X_train, Y_train = Z_features.iloc[train_idx], X_target[train_idx], Y_target[train_idx]
        Z_calib = Z_features.iloc[calib_idx]
        
        # Train and predict for X
        model_x_low = quantile_model_factory(alpha / 2)
        model_x_high = quantile_model_factory(1 - alpha / 2)
        model_x_low.fit(Z_train, X_train); model_x_high.fit(Z_train, X_train)
        all_preds_x_low = np.concatenate([all_preds_x_low, model_x_low.predict(Z_calib)])
        all_preds_x_high = np.concatenate([all_preds_x_high, model_x_high.predict(Z_calib)])

        # Train and predict for Y
        model_y_low = quantile_model_factory(alpha / 2)
        model_y_high = quantile_model_factory(1 - alpha / 2)
        model_y_low.fit(Z_train, Y_train); model_y_high.fit(Z_train, Y_train)
        all_preds_y_low = np.concatenate([all_preds_y_low, model_y_low.predict(Z_calib)])
        all_preds_y_high = np.concatenate([all_preds_y_high, model_y_high.predict(Z_calib)])
        
        all_indices = np.concatenate([all_indices, calib_idx])
        all_true_x, all_true_y = np.concatenate([all_true_x, X_target[calib_idx]]), np.concatenate([all_true_y, Y_target[calib_idx]])

    sort_order = np.argsort(all_indices)
    true_x, true_y = all_true_x[sort_order], all_true_y[sort_order]
    preds_x_low, preds_x_high = all_preds_x_low[sort_order], all_preds_x_high[sort_order]
    preds_y_low, preds_y_high = all_preds_y_low[sort_order], all_preds_y_high[sort_order]

    scores_x = np.maximum(preds_x_low - true_x, true_x - preds_x_high)
    scores_y = np.maximum(preds_y_low - true_y, true_y - preds_y_high)
    q_level = np.ceil((1 - alpha) * (len(data) + 1)) / len(data)
    q_x, q_y = np.quantile(scores_x, q_level), np.quantile(scores_y, q_level)

    centers_x = (preds_x_high + preds_x_low) / 2
    widths_x = (preds_x_high - preds_x_low) + 2 * q_x
    U = (true_x - centers_x) / np.where(widths_x == 0, 1, widths_x)
    centers_y = (preds_y_high + preds_y_low) / 2
    widths_y = (preds_y_high - preds_y_low) + 2 * q_y
    V = (true_y - centers_y) / np.where(widths_y == 0, 1, widths_y)
    
    return _residual_regression_test(U, V)

def _e_value_dml_ci_test(U, V, betting_folds=2):
    """Calculates an e-value on pre-computed residuals."""
    log_e_value = 0.0
    kf = KFold(n_splits=betting_folds, shuffle=True, random_state=123)
    for train_idx, test_idx in kf.split(U):
        U_train_df = pd.DataFrame(U[train_idx], columns=['u'])
        V_train = V[train_idx]
        U_test_df = pd.DataFrame(U[test_idx], columns=['u'])
        V_test = V[test_idx]
        
        # Use a non-linear model for the betting strategy to capture complex relationships
        betting_model = HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.1,
            random_state=123,
        )
        betting_model.fit(U_train_df, V_train)
        bets = betting_model.predict(U_test_df)
        e_process_fold = 1 + np.clip(bets, -0.9, 0.9) * V_test
        
        # Ensure e-process values are positive before taking the log
        e_process_fold = np.maximum(e_process_fold, 1e-9)
        
        log_e_value += np.sum(np.log(e_process_fold))

    # Clip the total log e-value to prevent overflow in np.exp
    safe_log_e_value = np.clip(log_e_value, None, 700)
    
    return np.exp(safe_log_e_value)

class DML(CITKTest):
    """
    Double-ML based conditional independence test.

    Parameters
    ----------
    data : np.ndarray
        The dataset from which to run the test.
    model : scikit-learn compatible regressor, optional
        The model used to predict X from Z and Y from Z. Defaults to HistGradientBoostingRegressor.
    cv_folds : int, optional
        The number of folds for cross-fitting.
    n_perms : int, optional
        Deprecated. Kept for backward compatibility.
    """
    supported_dtypes = {"continuous"}

    def __init__(self, data: np.ndarray, **kwargs):
        super().__init__(data, **kwargs)
        self.model = kwargs.get(
            'model',
            HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, random_state=42),
        )
        self.cv_folds = kwargs.get('cv_folds', 5)
        self.n_perms = kwargs.get('n_perms', 199)
        model_name = self.model.__class__.__name__
        params = f"model={model_name},cv={self.cv_folds},n_perms={self.n_perms}"
        self.check_cache_method_consistent('dml', params)

    def _compute(self, X: int, Y: int, condition_set: Optional[List[int]] = None, **kwargs) -> float:
        """
        Performs a Double Machine Learning (DML) based conditional independence test.

        It partials out the effect of the conditioning set Z from X and Y using a
        machine learning model and then tests for independence between the residuals.

        Parameters
        ----------
        X : int
            The index of the first variable.
        Y : int
            The index of the second variable.
        condition_set : list[int], optional
            A list of indices for the conditioning set. Can be empty.

        Returns
        -------
        p_value : float
            The p-value from residual regression on the residuals.

        .. seealso::
            For a detailed explanation of the statistical test, including mathematical
            formulations and assumptions, please refer to the :doc:`/tests/dml_test` guide.

        Examples
        --------
        **Standalone Usage**

        .. code-block:: python

            import numpy as np
            from citk.tests import DML

            # Generate data with a non-linear common confounder Z
            # Z -> X and Z -> Y
            n = 500
            Z = np.random.uniform(-3, 3, n)
            X = np.sin(Z) + np.random.randn(n) * 0.2
            Y = np.cos(Z) + np.random.randn(n) * 0.2
            data = np.vstack([X, Y, Z]).T

            # Initialize the test (uses HistGradientBoostingRegressor by default)
            dml_test = DML(data)

            # Test for unconditional independence (should be dependent)
            p_unconditional = dml_test(0, 1)
            print(f"P-value (unconditional) for X _||_ Y: {p_unconditional:.4f}")

            # Test for conditional independence given Z (should be independent)
            p_conditional = dml_test(0, 1, [2])
            print(f"P-value (conditional) for X _||_ Y | Z: {p_conditional:.4f}")

        .. code-block:: text

            P-value (unconditional) for X _||_ Y: 0.0050
            P-value (conditional) for X _||_ Y | Z: 0.6381

        **Usage with PC Algorithm**

        .. code-block:: python

            from causallearn.search.ConstraintBased.PC import pc
            
            cg = pc(data, alpha=0.05, indep_test='dml')
            print("Estimated Causal Graph:")
            print(cg.G)

        .. code-block:: text

            Estimated Causal Graph:
            Graph Nodes:
            X1;X2;X3

            Graph Edges:
            1. X1 --- X3
            2. X2 --- X3
            """
        U, V = _get_dml_residuals(self.model, self.data, X, Y, condition_set, cv_folds=self.cv_folds)
        p_value = _residual_regression_test(U, V)

        return float(p_value)

register_ci_test("dml", DML)

class CRIT(CITKTest):
    """
    Conformalized Residual Independence Test (CRIT).

    Parameters
    ----------
    data : np.ndarray
        The dataset from which to run the test.
    alpha : float, optional
        The significance level for the conformal prediction intervals.
    cv_folds : int, optional
        The number of folds for cross-fitting.
    n_perms : int, optional
        Deprecated. Kept for backward compatibility.
    """
    supported_dtypes = {"continuous"}

    def __init__(self, data: np.ndarray, **kwargs):
        super().__init__(data, **kwargs)
        self.alpha = kwargs.get('alpha', 0.1)
        self.cv_folds = kwargs.get('cv_folds', 5)
        self.n_perms = kwargs.get('n_perms', 199)
        self.quantile_model_factory = kwargs.get('quantile_model_factory', None)
        params = f"alpha={self.alpha},cv={self.cv_folds},n_perms={self.n_perms}"
        self.check_cache_method_consistent('crit', params)

    def _compute(self, X: int, Y: int, condition_set: Optional[List[int]] = None, **kwargs) -> float:
        """
        Performs a Conformalized Residual Independence Test (CRIT).

        This test uses conformal prediction to create robust, distribution-free
        residuals before testing for independence.

        Parameters
        ----------
        X : int
            The index of the first variable.
        Y : int
            The index of the second variable.
        condition_set : list[int], optional
            A list of indices for the conditioning set. Can be empty.

        Returns
        -------
        p_value : float
            The p-value from residual regression on the conformalized residuals.

    .. seealso::
        For a detailed explanation of the statistical test, including mathematical
        formulations and assumptions, please refer to the :doc:`/tests/crit_test` guide.

    Examples
    --------
    **Standalone Usage**

    .. code-block:: python

        import numpy as np
        from citk.tests import CRIT

        # Generate data with a non-linear relationship: X -> Z -> Y
        n = 500
        X = np.random.randn(n)
        Z = np.sin(X * 2) + np.random.randn(n) * 0.2
        Y = Z**3 + np.random.randn(n) * 0.2
        data = np.vstack([X, Y, Z]).T

        # Initialize the test
        crit_test = CRIT(data, alpha=0.1, n_perms=99)

        # Test for unconditional independence (should be dependent)
        p_unconditional = crit_test(0, 1)
        print(f"P-value (unconditional) for X _||_ Y: {p_unconditional:.4f}")

        # Test for conditional independence given Z (should be independent)
        p_conditional = crit_test(0, 1, [2])
        print(f"P-value (conditional) for X _||_ Y | Z: {p_conditional:.4f}")

    .. code-block:: text

        P-value (unconditional) for X _||_ Y: 0.0100
        P-value (conditional) for X _||_ Y | Z: 0.6800

    **Usage with PC Algorithm**

    .. code-block:: python

        from causallearn.search.ConstraintBased.PC import pc

        cg = pc(data, alpha=0.05, indep_test='crit')
        print("Estimated Causal Graph:")
        print(cg.G)

    .. code-block:: text

        Estimated Causal Graph:
        Graph Nodes:
        X1;X2;X3

        Graph Edges:
        1. X1 --- X3
        2. X2 --- X3
        """
        p_value = _conformalized_ci_test(
            self.data,
            X,
            Y,
            condition_set,
            alpha=self.alpha,
            cv_folds=self.cv_folds,
            n_perms=self.n_perms,
            quantile_model_factory=self.quantile_model_factory,
        )

        return float(p_value)

register_ci_test("crit", CRIT)

class EDML(CITKTest):
    """
    E-Value Double-ML based conditional independence test.

    Parameters
    ----------
    data : np.ndarray
        The dataset from which to run the test.
    model : scikit-learn compatible regressor, optional
        The model used to predict X from Z and Y from Z. Defaults to HistGradientBoostingRegressor.
    cv_folds : int, optional
        The number of folds for cross-fitting the residual models.
    betting_folds : int, optional
        The number of folds for the e-value betting mechanism.
    """
    supported_dtypes = {"continuous"}

    def __init__(self, data: np.ndarray, **kwargs):
        super().__init__(data, **kwargs)
        self.model = kwargs.get(
            'model',
            HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, random_state=42),
        )
        self.cv_folds = kwargs.get('cv_folds', 5)
        self.betting_folds = kwargs.get('betting_folds', 2)
        model_name = self.model.__class__.__name__
        params = f"model={model_name},cv={self.cv_folds},bet_folds={self.betting_folds}"
        self.check_cache_method_consistent('edml', params)

    def _compute(self, X: int, Y: int, condition_set: Optional[List[int]] = None, **kwargs) -> float:
        """
        Performs an E-Value Double Machine Learning (EDML) CI test.

        This test produces an e-value, which is then converted to a p-value.
        It uses the same residualization as DML but replaces the final permutation
        test with a betting-based e-value calculation.

        Parameters
        ----------
        X : int
            The index of the first variable.
        Y : int
            The index of the second variable.
        condition_set : list[int], optional
            A list of indices for the conditioning set. Can be empty.

        Returns
        -------
        p_value : float
            The p-value derived from the calculated e-value.

    .. seealso::
        For a detailed explanation of the statistical test, including mathematical
        formulations and assumptions, please refer to the :doc:`/tests/edml_test` guide.

    Examples
    --------
    **Standalone Usage**

    .. code-block:: python

        import numpy as np
        from citk.tests import EDML

        # Generate data with a non-linear common confounder Z
        # Z -> X and Z -> Y
        n = 500
        Z = np.random.uniform(-3, 3, n)
        X = np.sin(Z) + np.random.randn(n) * 0.2
        Y = np.cos(Z) + np.random.randn(n) * 0.2
        data = np.vstack([X, Y, Z]).T

        # Initialize the test.
        edml_test = EDML(data)

        # Test for unconditional independence (should be dependent, p-value should be small)
        p_unconditional = edml_test(0, 1)
        print(f"P-value (unconditional) for X _||_ Y: {p_unconditional:.4f}")

        # Test for conditional independence given Z (should be independent, p-value should be large)
        p_conditional = edml_test(0, 1, [2])
        print(f"P-value (conditional) for X _||_ Y | Z: {p_conditional:.4f}")

    .. code-block:: text

        P-value (unconditional) for X _||_ Y: 0.0000
        P-value (conditional) for X _||_ Y | Z: 1.0000

    **Usage with PC Algorithm**

    .. code-block:: python

        from causallearn.search.ConstraintBased.PC import pc

        cg = pc(data, alpha=0.05, indep_test='edml')
        print("Estimated Causal Graph:")
        print(cg.G)
        
    .. code-block:: text

        Estimated Causal Graph:
        Graph Nodes:
        X1;X2;X3

        Graph Edges:
        1. X1 --- X3
        2. X2 --- X3
        """
        U, V = _get_dml_residuals(self.model, self.data, X, Y, condition_set, cv_folds=self.cv_folds)
        e_value = _e_value_dml_ci_test(U, V, betting_folds=self.betting_folds)
        
        # Convert e-value to p-value. 1/e is a common (though sometimes conservative) choice.
        # Ensure p-value is at most 1.
        p_value = min(1.0, 1.0 / e_value if e_value > 0 else float('inf'))

        return float(p_value)

register_ci_test("edml", EDML)
