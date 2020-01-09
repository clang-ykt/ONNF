#DocCheck

## Goal

DocCheck provides a set of utilities to enforce artifacts (e.g., code snippets or output of execution) used in the documentation is compatible and up-to-date with the state of software development they describe.

## Directives

DocCheck provides a set of directives that can be used in documentations to enforce desired invariants.

### `same-as-file`:

Use `same-as-file` directive to ensure that the code section following this directive is exactly the same as a source file tested in a unit test.

For example, to ensure the following code snippet is consistent as a unit-tested file `reference.cpp`, use the following directive as shown in the documentation snippet:

[same-as-file]: <> (doc/doc_check/test/same-as-file/simple/README.md)
````markdown
Lorem ipsum dolor sit amet, consectetur adipiscing elit.

[same-as-file]: <> (reference.cpp)
```cpp
#include<iostream>

using namespace std;

int main() {
    cout<<"Hello World";
    return 0;
}
```
````
