The following are guidelines and suggestion that might help produce code that
is easier to integrate.


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
