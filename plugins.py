"""Plugins the chat app can call automatically.

Semantic Kernel reads the type hints and descriptions below and exposes
these methods to the model as callable tools.
"""

import ast
import operator
from datetime import datetime
from typing import Annotated

from semantic_kernel.functions import kernel_function


class TimePlugin:
    """Gives the model access to the current date and time."""

    @kernel_function(name="current_datetime", description="Get the current local date and time.")
    def current_datetime(self) -> Annotated[str, "The current date and time as text"]:
        return datetime.now().strftime("%A, %d %B %Y, %H:%M:%S")


class MathPlugin:
    """Safely evaluates basic arithmetic so the model doesn't guess at maths."""

    _OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
    }

    def _eval(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in self._OPS:
            return self._OPS[type(node.op)](self._eval(node.left), self._eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._OPS:
            return self._OPS[type(node.op)](self._eval(node.operand))
        raise ValueError("Unsupported expression")

    @kernel_function(
        name="calculate",
        description="Evaluate an arithmetic expression such as '(12 + 8) * 3 / 4'.",
    )
    def calculate(
        self, expression: Annotated[str, "The arithmetic expression to evaluate"]
    ) -> Annotated[str, "The numeric result, or an error message"]:
        try:
            tree = ast.parse(expression, mode="eval")
            return str(self._eval(tree.body))
        except Exception as exc:  # noqa: BLE001
            return f"Could not evaluate '{expression}': {exc}"
