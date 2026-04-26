import numpy as np
from typing import Optional
import pandas as pd

from causallearn.utils.cit import register_ci_test, NO_SPECIFIED_PARAMETERS_MSG, Chisq_or_Gsq, CIT
from .base import CITKTest, inner_test_kwargs


class FisherZ(CITKTest):
    """
    This class is a wrapper around the `fisherz` test from the `causal-learn` library.

    Parameters
    ----------
    data : np.ndarray
        The dataset from which to run the test.
    """
    supported_dtypes = {"continuous"}

    def __init__(self, data: np.ndarray, **kwargs):
        super().__init__(data, **kwargs)
        self.data = data
        self.check_cache_method_consistent('fisherz_citk', "NO SPECIFIED PARAMETERS")
        self.test_instance = CIT(data, method_name='fisherz', **inner_test_kwargs(kwargs))

    def _compute(self, X: int, Y: int, condition_set: Optional[list[int]] = None, **kwargs) -> float:
        """
        Performs a Fisher-Z conditional independence test.

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
        formulations and assumptions, please refer to the :doc:`/tests/fisher_z_test` guide.


    Examples
    --------
    **Standalone Usage**

    .. code-block:: python

        import numpy as np
        from citk.tests import FisherZ

        # Generate data where X and Y are independent given Z
        # X -> Z -> Y
        n = 500
        X = np.random.randn(n)
        Z = 2 * X + np.random.randn(n)
        Y = 3 * Z + np.random.randn(n)
        data = np.vstack([X, Y, Z]).T

        # Initialize the test
        fisher_z_test = FisherZ(data)

        # Test if X and Y are independent
        p_value_unconditional = fisher_z_test(0, 1)
        print(f"P-value (unconditional) for X _||_ Y: {p_value_unconditional:.4f}")

        # Test if X and Y are independent given Z
        p_value_conditional = fisher_z_test(0, 1, [2])
        print(f"P-value (conditional) for X _||_ Y | Z: {p_value_conditional:.4f}")

    .. code-block:: text

        P-value (unconditional) for X _||_ Y: 0.0000
        P-value (conditional) for X _||_ Y | Z: 0.9876



    **Usage with PC Algorithm**

    .. code-block:: python

        from causallearn.search.ConstraintBased.PC import pc

        # The same data from the standalone example
        cg = pc(data, alpha=0.05, indep_test='fisherz_citk')

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
        return float(self.test_instance(X, Y, condition_set))

register_ci_test("fisherz_citk", FisherZ)


class Spearman(CITKTest):
    """
    This class is a wrapper around the `fisherz` test from the `causal-learn` library on ranked data.

    Parameters
    ----------
    data : np.ndarray
        The dataset from which to run the test.
    """
    supported_dtypes = {"continuous"}

    def __init__(self, data: np.ndarray, **kwargs):
        # Rank the data to convert Pearson's correlation to Spearman's.
        ranked_data = pd.DataFrame(data).rank().to_numpy()

        # Initialize the base class with the RANKED data to ensure hash consistency.
        super().__init__(ranked_data, **kwargs)

        self.check_cache_method_consistent('spearman', NO_SPECIFIED_PARAMETERS_MSG)
        self.test_instance = CIT(self.data, method_name='fisherz', **inner_test_kwargs(kwargs))

    def _compute(self, X: int, Y: int, condition_set: Optional[list[int]] = None, **kwargs) -> float:
        """
        Performs a Spearman partial correlation conditional independence test.

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
        formulations and assumptions, please refer to the :doc:`/tests/spearman` guide.

        Examples
        --------
        **Standalone Usage**

    .. code-block:: python

        import numpy as np
        from citk.tests import Spearman

        # Generate data with a non-linear, monotonic relationship
        # X -> Z -> Y, where the relationships are not linear
        n = 500
        X = np.random.rand(n) * 5
        Z = np.exp(X / 2) + np.random.randn(n) * 0.1
        Y = np.log(Z**2) + np.random.randn(n) * 0.1
        data = np.vstack([X, Y, Z]).T

        # Initialize the test
        spearman_test = Spearman(data)

        # Test if X and Y are independent
        p_value_unconditional = spearman_test(0, 1)
        print(f"P-value (unconditional) for X _||_ Y: {p_value_unconditional:.4f}")

        # Test if X and Y are independent given Z
        p_value_conditional = spearman_test(0, 1, [2])
        print(f"P-value (conditional) for X _||_ Y | Z: {p_value_conditional:.4f}")

    .. code-block:: text 

        P-value (unconditional) for X _||_ Y: 0.0000
        P-value (conditional) for X _||_ Y | Z: 0.4640

    **Usage with PC Algorithm**

    .. code-block:: python

        from causallearn.search.ConstraintBased.PC import pc

        # The same data from the standalone example
        cg = pc(data, alpha=0.05, indep_test='spearman')

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
        return float(self.test_instance(X, Y, condition_set))

register_ci_test("spearman", Spearman)


class GSq(CITKTest):
    """
    This class is a wrapper around the `gsq` test from the `causal-learn` library.
    
    Parameters
    ----------
    data : np.ndarray
        The dataset from which to run the test.
    """
    supported_dtypes = {"discrete"}

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.check_cache_method_consistent('gsq', "NO SPECIFIED PARAMETERS")
        self.test_instance = Chisq_or_Gsq(data, method_name='gsq', **inner_test_kwargs(kwargs))

    def _compute(self, X: int, Y: int, condition_set: Optional[list[int]] = None, **kwargs) -> float:
        """
        Performs a G-Square conditional independence test for discrete data.

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
        formulations and assumptions, please refer to the :doc:`/tests/g_sq_test` guide.

        Examples
        --------
        **Standalone Usage**

    .. code-block:: python

        import numpy as np
        from citk.tests import GSq

        # Generate discrete data for a chain: X -> Z -> Y
        # X, Y, and Z have 3, 3, and 2 levels respectively.
        n = 500
        X = np.random.randint(0, 3, size=n)
        Z = (X + np.random.randint(0, 2, size=n)) % 3
        Y = (Z + np.random.randint(0, 2, size=n)) % 3
        data = np.vstack([X, Y, Z]).T

        # Initialize the test
        g_sq_test = GSq(data)

        # Test for unconditional independence
        p_value_unconditional = g_sq_test(0, 1)
        print(f"P-value (unconditional) for X _||_ Y: {p_value_unconditional:.4f}")

        # Test for conditional independence given Z
        p_value_conditional = g_sq_test(0, 1, [2])
        print(f"P-value (conditional) for X _||_ Y | Z: {p_value_conditional:.4f}")

    .. code-block:: text 

        P-value (unconditional) for X _||_ Y: 0.0000
        P-value (conditional) for X _||_ Y | Z: 0.6069

        **Usage with PC Algorithm**

    .. code-block:: python

        from causallearn.search.ConstraintBased.PC import pc

        # The same discrete data from the standalone example
        cg = pc(data, alpha=0.05, indep_test='gsq')

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
        return float(self.test_instance(X, Y, condition_set))

register_ci_test("gsq", GSq)


class ChiSq(CITKTest):
    """
    This class is a wrapper around the `chisq` test from the `causal-learn` library.

    Parameters
    ----------
    data : np.ndarray
        The dataset from which to run the test.
    """
    supported_dtypes = {"discrete"}

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.check_cache_method_consistent('chisq', "NO SPECIFIED PARAMETERS")
        self.test_instance = Chisq_or_Gsq(data, method_name='chisq', **inner_test_kwargs(kwargs))

    def _compute(self, X: int, Y: int, condition_set: Optional[list[int]] = None, **kwargs) -> float:
        """
        Performs a Chi-Square conditional independence test for discrete data.

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
        formulations and assumptions, please refer to the :doc:`/tests/chi_sq_test` guide.

        Examples
        --------
        **Standalone Usage**

    .. code-block:: python

        import numpy as np
        from citk.tests import ChiSq

        # Generate discrete data for a chain: X -> Z -> Y
        # X, Y, and Z have 3, 3, and 2 levels respectively.
        n = 500
        X = np.random.randint(0, 3, size=n)
        Z = (X + np.random.randint(0, 2, size=n)) % 3
        Y = (Z + np.random.randint(0, 2, size=n)) % 3
        data = np.vstack([X, Y, Z]).T

        # Initialize the test
        g_sq_test = ChiSq(data)

        # Test for unconditional independence
        p_value_unconditional = g_sq_test(0, 1)
        print(f"P-value (unconditional) for X _||_ Y: {p_value_unconditional:.4f}")

        # Test for conditional independence given Z
        p_value_conditional = g_sq_test(0, 1, [2])
        print(f"P-value (conditional) for X _||_ Y | Z: {p_value_conditional:.4f}")

    .. code-block:: text 

        P-value (unconditional) for X _||_ Y: 0.0000
        P-value (conditional) for X _||_ Y | Z: 0.0870

        **Usage with PC Algorithm**

    .. code-block:: python

        from causallearn.search.ConstraintBased.PC import pc

        # The same discrete data from the standalone example
        cg = pc(data, alpha=0.05, indep_test='chisq')

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
        return float(self.test_instance(X, Y, condition_set))

register_ci_test("chisq", ChiSq)
