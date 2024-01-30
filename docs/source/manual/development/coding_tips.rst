Coding Tips
-----------

The following are guidelines and suggestion that might help produce code that
is easier to integrate.  These are based on and may be biased by Jon Crall's
opinions.


1. Carefully think about inputs and outputs. Try to write functions in such a
   way that the inputs are easy to construct. If there is a bug, it helps to be
   able to quickly define a simple **entrypoint** into the code that exercies
   the behavior in question. Doctests are great for this!


2. Don't try to do too much in a single step. The outputs should be a
   reasonably simple transformation of the inputs. If you are doing a lot of
   different things to the inputs and have many intermediate forms, perhaps
   it's better to split up the functionality into multiple stages. This is not
   always true. Use best judgement.


3. Think about serializability. If you have a lot of nested custom Python
   classes, it may be difficult to reconstruct states in debugging, or refactor
   later if a better way of doing something is discovered.


4. Try make code easy for static analysis. Try to avoid decorators and other
   code that is run at import time. When possible it's best to simply define
   classes and functions at import time, and then leave any complex logic to
   the functions themselves. As always, there are exceptions to this rule.


5. Try to avoid polymorphism in cases where mixin classes would work well.
   It's usually better to start with something non-polymorphic and then
   refactor it to be polymorphic later if it turns out that that strategy is
   useful. Often mixin classes are enough. In general, try to keep code
   structures flat when possible.


6. Use pathlib (or ubelt.Path) instead of os.path methods.


7. Don't import pyplot in the global scope. It has runtime side effects we
   would like to avoid.


8. Use ``python -m`` to invoke scripts via their module name, instead of
   specifying the path to them directly.



Valuable Code Properties
------------------------


Code Patterns
~~~~~~~~~~~~~

    1. Idempotence - https://en.wikipedia.org/wiki/Idempotence

    2. Examples and Entry Points - Try to make it easy to test small units of
       code. See the doctests section for one good way to do this.

    3. When writing configurations use default values of "auto", if you can
       infer a reasonable default behavior at runtime. Using None also works,
       depending on the context.


Testing
~~~~~~~

   1. Doctests - https://github.com/Erotemic/xdoctest

   2. "Demo Data" - see kwcoco demodata

   3. Continuous Integration - CI Server


Filesystems
~~~~~~~~~~~

    1. User-agnostic or relative paths - Dont hard code your paths. Use things
           like "~" or "$HOME" environment variables. Use the ``python -m``
           mechanism to invoke code by module name rather than using the
           absolute path (some exceptions apply).

    2. Content addressable data - IPFS, DVC, Hasing - Make dedup easy, this may
           be the future of data storage.

    3. ``fpath`` for file-paths, ``dpath`` for directory-paths, ``path`` for an
           unspecified path type.  If possible, avoid ``dir`` because it
           conflicts with ``builtins.dir``.



Misc
----

1. Statically parsable declarative configurations are useful, this is why I
   like scriptconfig over argparse. I would like to add jsonargparse
   integration as the mechanism of handling nested configurations, so it
   should be compatible with jsonargparse powered things like LightningCLI in
   the future.


2. When you are writing a python package, give your module a distinctive name.
   Don't name it "lib" or "model" or "net", do something that wont conflict
   with other python packages. This is a tip based off of observing this
   anti-pattern in research repos.
