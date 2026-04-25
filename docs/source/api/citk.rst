API Reference
=============

This page provides the API reference for the conditional independence tests
in `citk`. Sections follow the survey taxonomy of CI test families.

Partial Correlation Tests
-------------------------

.. autoclass:: citk.tests.FisherZ
   :members:
   :show-inheritance:
   :special-members: __call__

.. autoclass:: citk.tests.Spearman
   :members:
   :show-inheritance:
   :special-members: __call__

Contingency Table Tests
-----------------------

.. autoclass:: citk.tests.ChiSq
   :members:
   :show-inheritance:
   :special-members: __call__

.. autoclass:: citk.tests.GSq
   :members:
   :show-inheritance:
   :special-members: __call__

Regression-Based Tests
----------------------

.. autoclass:: citk.tests.RegressionCI
   :members:
   :show-inheritance:
   :special-members: __call__

.. autoclass:: citk.tests.CiMM
   :members:
   :show-inheritance:
   :special-members: __call__

Nearest Neighbor Tests
----------------------

.. autoclass:: citk.tests.CMIknn
   :members:
   :show-inheritance:
   :special-members: __call__

.. autoclass:: citk.tests.CMIknnMixed
   :members:
   :show-inheritance:
   :special-members: __call__

.. autoclass:: citk.tests.MCMIknn
   :members:
   :show-inheritance:
   :special-members: __call__

Kernel Tests
------------

.. autoclass:: citk.tests.KCI
   :members:
   :show-inheritance:
   :special-members: __call__

.. autoclass:: citk.tests.RCoT
   :members:
   :show-inheritance:
   :special-members: __call__

.. autoclass:: citk.tests.RCIT
   :members:
   :show-inheritance:
   :special-members: __call__

ML-Based Tests
--------------

.. autoclass:: citk.tests.GCM
   :members:
   :show-inheritance:
   :special-members: __call__

.. autoclass:: citk.tests.WGCM
   :members:
   :show-inheritance:
   :special-members: __call__

.. autoclass:: citk.tests.PCM
   :members:
   :show-inheritance:
   :special-members: __call__

Adapter Tests
-------------

.. autoclass:: citk.tests.DiscChiSq
   :members:
   :show-inheritance:
   :special-members: __call__

.. autoclass:: citk.tests.DiscGSq
   :members:
   :show-inheritance:
   :special-members: __call__

.. autoclass:: citk.tests.DummyFisherZ
   :members:
   :show-inheritance:
   :special-members: __call__

.. autoclass:: citk.tests.HarteminkChiSq
   :members:
   :show-inheritance:
   :special-members: __call__

Base Class
----------

.. automodule:: citk.tests.base
   :members:
   :show-inheritance:
