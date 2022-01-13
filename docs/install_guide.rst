.. _install_guide:

Install
=======

There are different ways to install python-psignifit:

- :ref:`Install the latest release <pip install>` from the python package index. This is the recommended approach for most users.
- Download and install the source from the Github :ref:`website <zip install>` or :ref:`git repository <git install>`.
  Use this approach to inspect and modify the source code or to use a psignifit version, that was not published in the package index.

python-psignifit depends on few external python packages:
- NumPy
- SciPy
- matplotlib

These packages are often shipped with scientific python distributions or, if not already there,
are installed automatically during the installation of python-psignifit.
We `automatically check <https://github.com/wichmann-lab/python-psignifit/actions/workflows/ci-tests.yml>`
compatibility with different python 3 versions on Ubuntu Linux and Windows. We do not support python 2.

.. _pip install:

Installing the latest release
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install python-psignifit with all dependencies:

::

    pip install psignifit


.. _zip install:

Installing from source
~~~~~~~~~~~~~~~~~~~~~~

To install psignifit go to
https://github.com/wichmann-lab/python-psignifit and click on the “Clone
or Download” button on the right or click here: `Download
ZIP <https://github.com/wichmann-lab/python-psignifit/archive/master.zip>`__.
Then unpack the ZIP file and run the contained setup.py file:

::

   python setup.py install
   # OR to make use of changes in the code:
   python setup.py develop

This should work on Linux, MAC and Windows equally well.

If you “installed” this way and you want to update your psignifit,
simply download again and replace the folder.


.. _git install:

Using Git to be kept up to Date
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instructions on how to install git can be found
`here <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`__.

If you have git installed you can also use this to get the newest
version and to keep yourself up to date:

To clone the repository for the first time, change to the directory
where psignifit should be placed. There use the following command in a
terminal:

``git clone https://github.com/wichmann-lab/python-psignifit.git``

Now you should have a directory called “python-psignifit” there. This
contains all the code you need and you can proceed to adding the code to
your path as well.

To update your local copy you only need to change to the
“python-psignifit” folder and type:

``git pull``

Install as described in :ref:`zip install`.
