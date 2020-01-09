from typing import List
from utils import *

from directive import Directive


def parse_code_section_delimiter(ctx):
    if not ctx.doc_file.next_non_empty_line().strip().startswith("```"):
        raise ValueError("Did not parse a code section delimiter")


def try_parse_and_handle_directive(ctx):
    from directive_impl import same_as_file

    all_directives: List[Directive] = [
        Directive(same_as_file.ext_to_patterns, same_as_file.parse, same_as_file.handle)
    ]

    for directive in all_directives:
        directive_config = []
        if succeeded(directive.try_parse_directive(ctx, directive_config)):
            directive.handle(directive_config.pop(), ctx)
            return success(directive_config)

    return failure()
