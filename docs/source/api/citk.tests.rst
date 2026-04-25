.. _citk-tests:

citk.tests package
==================

This package contains the implementations of various conditional independence tests.

Partial Correlation Tests
-------------------------

.. autoclass:: citk.tests.partial_correlation_tests.FisherZ
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __call__

.. autoclass:: citk.tests.partial_correlation_tests.Spearman
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __call__

Contingency Table Tests
-----------------------

.. autoclass:: citk.tests.contingency_table_tests.GSq
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __call__

.. autoclass:: citk.tests.contingency_table_tests.ChiSq
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __call__

Regression-Based Tests
----------------------

.. autoclass:: citk.tests.regression_tests.RegressionCI
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __call__

Nearest Neighbor Tests
----------------------

.. autoclass:: citk.tests.nearest_neighbor_tests.CMIknn
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __call__

.. autoclass:: citk.tests.nearest_neighbor_tests.CMIknnMixed
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __call__

.. autoclass:: citk.tests.nearest_neighbor_tests.MCMIknn
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __call__

Kernel Tests
------------

.. autoclass:: citk.tests.kernel_tests.KCI
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __call__

.. autoclass:: citk.tests.kernel_tests.RCoT
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __call__

.. autoclass:: citk.tests.kernel_tests.RCIT
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __call__

Adapter Tests
-------------

.. autoclass:: citk.tests.adapter_tests.DiscChiSq
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __call__

.. autoclass:: citk.tests.adapter_tests.DiscGSq
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __call__

.. autoclass:: citk.tests.adapter_tests.DummyFisherZ
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __call__

.. autoclass:: citk.tests.adapter_tests.HarteminkChiSq
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __call__
