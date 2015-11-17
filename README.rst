============================================
``causalinfo``: Information on Causal Graphs 
============================================

.. image:: https://badge.fury.io/py/causalinfo.png
    :target: http://badge.fury.io/py/causalinfo

`causalinfo` is a Python library to aid in experimenting with different
*information measures on causal graphs*---a combination of information
theory with recent work on causal graphs [Pearl2000]_. These information
measures can used to ascertain the degree to which one variable controls or
explains other variables in the graph. The use of these measures has important
connections to work on causal explanation in philosophy of science, and to
understanding information processing in biological networks. 

The library is a work in progress, and will be extended as research continues.

What does it do?
----------------

`causalinfo` has been written primarily for interactive use within `IPython
Notebook`_. You can create variables and assign probability distributions to
them, or relate them to other variables using conditional probabilities.
Several related variables can be combined into a directed acyclic graph, which
can generate a joint distribution for all variables under observation, or
under controlled interventions on certain variables. You can also calculate
various information measures between variables in the graph whilst controlling
other variables. These include correlative measures, such as Mutual
Information, but also causal measures, such as Information Flow
[AyPolani2008]_, and Causal Specificity [GriffithsEtAl2015]_.

For some examples of how to use the library, please see the IPython Notebooks
that are included:

* Introduction_. A short introduction to some of the things you can do with
  the library.

* Rain_. Performing some interventions on a causal graph from Pearl's book.

.. TODO: Add the signaling stuff in.
.. * Signaling_. Looking at the measures of multiple pathways.

.. _Introduction: https://github.com/brettc/causalinfo/blob/master/notebooks/introduction.ipynb

.. _Rain: https://github.com/brettc/causalinfo/blob/master/notebooks/rain.ipynb

.. Signaling: https://github.com/brettc/causalinfo/blob/master/notebooks/signaling.ipynb -->


.. TODO: Add a getting started guide
.. Getting Started
    ---------------
    .. code:: bash 
    pip install causalinfo
    curl https://raw.githubusercontent.com/brettc/causalinfo/master/notebooks/introduction.ipynb 

Some Caveats
------------

The library is not meant for large scale analysis. The code has been written
to offload as much as possible on to other libraries (such as Pandas_ and
Networkx_), and to allow easy inspection of what is going on within `IPython
Notebook`_, thus it is not optimized for speed. Calculating the joint
distribution for a causal graph with many variables can become very *slow*
(especially if the variables have many states). 


Authorship
----------

All code was written by `Brett Calcott`_.


Acknowledgments
---------------

This work is part of the research project on the `Causal Foundations of
Biological Information`_ at the `University of Sydney`_, Australia. The work
was made possible through the support of a grant from the Templeton World
Charity Foundation. The opinions expressed are those of the author and do not
necessarily reflect the views of the Templeton World Charity Foundation. 

License
-------

MIT licensed. See the bundled LICENSE_ file for more details.


.. Miscellaneous Links------------

.. _LICENSE: https://github.com/brettc/causalinfo/blob/master/LICENSE

.. _`Brett Calcott`: http://brettcalcott.com

.. _`University of Sydney`: http://sydney.edu.au/ 

.. _`IPython Notebook`: http://ipython.org/notebook.html 

.. _Pandas: http://pandas.pydata.org/

.. _Networkx: https://networkx.github.io/ 

.. _`Causal Foundations of Biological Information`: http://sydney.edu.au/foundations_of_science/research/causal_foundations_biological_information.shtml 


References
----------

.. [AyPolani2008] Ay, N., & Polani, D. (2008). Information flows in causal
    networks. Advances in Complex Systems, 11(01), 17–41.

.. [GriffithsEtAl2015] Griffiths, P. E., Pocheville, A., Calcott, B., Stotz, K., 
    Kim, H., & Knight, R. (2015). Measuring Causal Specificity. Philosophy of Science, 82(October), 529–555.

.. [Pearl2000] Pearl, J. (2000). Causality. Cambridge University Press. 


.. vim: fo=tcroqn tw=78
