import rich

from rich import inspect
from ipdb import set_trace


def print_stack():
    import traceback

    stacks = traceback.extract_stack()
    stacks = [s for s in stacks if "miniconda" not in s.filename][:-2]

    # Hierarchical representation
    for s in stacks:
        fn = s.filename.replace("/data/yanxia/code/PHC/", "")
        rich.print(
            f"[bold blue]{fn}[/bold blue]:[green]{s.lineno}[/green] in [yellow]{s.name}[/yellow]"
        )
        content = s.line
        if content == "":
            content = "<empty>"
        rich.print(f"    [bright_black]{content}[/bright_black]")
