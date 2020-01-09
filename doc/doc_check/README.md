#DocCheck

## Goal

DocCheck provides a set of utilities to enforce artifacts (e.g., code snippets or output of execution) used in the documentation is compatible and up-to-date with the state of software development they describe.

## Directives

DocCheck provides a set of directives that can be used in documentations to enforce desired invariants.
A directive is a comment with a specific format/syntax to communicate the intent to check certain invariants to the DocCheck checker.
Generally, a directive has the following syntax in markdown:

```markdown
[{directive_command}]: <> ({directive_configuration})
```

Where directive_command indicates the type of directive/checking intended and directive_configuration expresses the specific parameters to use with this directive command.
In general, a directive configuration is expressed using a python dictionary literal, but special shorthand versions of each directive's configuration are supported.
They should be documented in the comment of each directive individually.

### `same-as-file`:

Use `same-as-file` directive to ensure that the code section following this directive is the same as a source file tested in some unit test.

`same-as-file` directive supports a convenient short-hand configuration format where the directive configuration can be fully specified using the name of the reference file to check against.
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

In the canonical form of directive configuration (as a python dictionary literal), this directive supports these parameters in it:

`ref` (string): reference file to check against.

`skip-doc` (int): number of lines to skip when checking the documentation.

`skip-ref` (int): number of lines to skip when scanning the reference file.

For example, to ensure the following code snippet is the same as a unit-tested file `reference.cpp`, except for the first 2 lines of the code used in documentation, and the first 3 lines of code used in the reference file, the following directive configuration can be used:

[same-as-file]: <> (doc/doc_check/test/same-as-file/skip-doc-ref/README.md)
````markdown
Lorem ipsum dolor sit amet, consectetur adipiscing elit.

[same-as-file]: <> ({"ref": "reference.cpp", "skip-doc": 2, "skip-ref": 3})
```cpp
// First line unique to documentation
// Second line unique to documentation
#include<iostream>

using namespace std;

int main() {
    cout<<"Hello World";
    return 0;
}
```
````