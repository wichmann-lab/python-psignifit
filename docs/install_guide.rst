.. _install_guide:

Install Guide
=============

We aimed to make this clone depend on as few as possible additional
packages. The standard ones we could not avoid are: Numpy (>1.11.0),
Scipy (>0.18.1), matplotlib (>2.0) for the plots and datetime, warnings
and copy from the standard library. These packages are usually shipped
with standard python distributions. All tests of the code were run in
python 3.5/3.6/3.7. We do not support python 2.

Once you have the required packages installed psignifit can be
downloaded:

Downloading a ZIP File
~~~~~~~~~~~~~~~~~~~~~~

To install psignifit go to
https://github.com/wichmann-lab/python-psignifit and click on the “Clone
or Download” button on the right or click here: `Download
ZIP <https://github.com/wichmann-lab/python-psignifit/archive/master.zip>`__.
Then unpack the ZIP file and run the contained setup.py file:

::

   python setup.py install

This should work on Linux, MAC and Windows equally well.

If you “installed” this way and you want to update your psignifit,
simply download again and replace the folder.

Using Git to be Kept up to Date
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
