.. highlight:: shell

============
Contributing
============

All contributions to the diffcalc-core project are very welcome and greatly appreciated!

There are multiple ways to contribute:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/DiamondLightSource/diffcalc-core/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

diffcalc-core project could always use more documentation, whether as part of the
official diffcalc-core docs or docstrings.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/DiamondLightSource/diffcalc-core/issues.

Get Started!
------------

Ready to contribute? Here's how to set up `diffcalc-core` for local development.

1. Fork the `diffcalc-core` repo on GitHub.

2. Clone your fork locally::

    $ git clone git@github.com:DiamondLightSource/diffcalc-core.git

3. Install your local copy into a virtual environment. This is how you set up your fork for local development::

    $ python3 -m venv diffcalc-core
    $ source diffcalc-core/bin/activate
    $ pip install -e diffcalc-core
    $ cd diffcalc-core/
    $ pip install -r requirements_dev.txt

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. Install precommit hooks which will help keep the code maintainable::

    $ pre-commit install

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

Deploying
---------

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in HISTORY.rst).
Then run::

    $ bump2version patch # possible: major / minor / patch
    $ git push
    $ git push --tags

Azure will then deploy to PyPI if tests pass.
