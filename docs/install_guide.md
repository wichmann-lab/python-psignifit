# Install

There are different ways to install *psignifit*:

- [Install the latest release](#install-pip) using `pip`. 
This is the recommended approach for most users.

- Download and install the source from the [Github repository](https://github.com/wichmann-lab/python-psignifit/).
Use this approach to inspect and modify the source code or to use a
psignifit version that has not been released yet.

*psignifit* depends on a few standard scientific libraries:
`numpy`, `scipy` and `matplotlib`. If they are not already there, these
dependencies are installed automatically during the
installation of psignifit.

(install-pip)=
## Installing the latest release (preferred method)

Install psignifit with all dependencies:

```bash
pip install psignifit
```

## Installing from source

Use this approach to inspect and modify the source code.
To install psignifit, go to 
the [Github repository](https://github.com/wichmann-lab/python-psignifit/).
and click on the "Clone or Download" button on the right.
Then unpack the ZIP file, navigate with the command line inside the
folder `python-psignifit`, and then run:

```
pip install -e .
```

## Using Git to live on the bleeding edge

If you know how to use `git` you can use the newest version of the
psignifit code before it gets released officially.

To clone the repository for the first time, change to the directory
where psignifit should be placed. There use the following command in a
terminal:

```
git clone https://github.com/wichmann-lab/python-psignifit.git
```

Now you should have a directory called `python-psignifit` there. Install
psignifit using the editable installation method with `pip`:

```
pip install -e .
```

Anytime you want to update your local copy change to the
`python-psignifit` folder and type:

```
git pull
```

