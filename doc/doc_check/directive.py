import re, ast
from typing import List, Dict, Callable, Any, Pattern, Tuple

from doc_parser import failure, success, succeeded
from utils import DocCheckerCtx

DirectiveConfigList = List[Dict[str, Any]]
ConfigParseResult = Tuple[str, Dict[str, Any]]


class Directive(object):
    def __init__(self, ext_to_regexes: Dict[str, str],
                 custom_parsers: List[Callable[[str, DirectiveConfigList],
                                               ConfigParseResult]],
                 handler: Callable[[Dict[str, Any], DocCheckerCtx], None]):
        self.ext_to_patterns: Dict[str, Pattern] = {}
        for ext, pattern in ext_to_regexes.items():
            self.ext_to_patterns[ext] = re.compile(pattern)

        self.custom_parsers: List[Callable[[str, DirectiveConfigList],
                                           ConfigParseResult]] = custom_parsers
        self.handler = handler

    def try_parse_directive(
            self, ctx: DocCheckerCtx,
            directive_config: DirectiveConfigList) -> Tuple[str, Any]:
        line = ctx.doc_file.next_non_empty_line()
        matches = self.ext_to_patterns[ctx.doc_file_ext()].findall(line)
        if len(matches) > 1:
            raise ValueError("more than one directives in a line")

        match = matches[0] if len(matches) else None
        if match:
            for parser in self.custom_parsers:
                if succeeded(parser(match, directive_config)):
                    return success()

            raise ValueError("Failed to parse configuration.")
        else:
            return failure()

    def handle(self, config, ctx):
        self.handler(config, ctx)


def generic_config_parser(
        match: str, directive_config: DirectiveConfigList) -> Tuple[str, Any]:
    try:
        directive_config.append(ast.literal_eval(match))
        return success()
    except (SyntaxError, ValueError):
        return failure()
