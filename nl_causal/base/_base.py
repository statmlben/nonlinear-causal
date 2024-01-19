import pandas as pd
import numpy as np

def print_msg_box(msg, indent=1, width=None, title=None):
    """Print message-box with optional title."""
    lines = msg.split('\n')
    space = " " * indent
    if not width:
        width = max(map(len, lines))
    box = f'╔{"═" * (width + indent * 2)}╗\n'  # upper_border
    if title:
        box += f'║{space}{title:<{width}}{space}║\n'  # title
        box += f'║{space}{"-" * len(title):<{width}}{space}║\n'  # underscore
    box += ''.join([f'║{space}{line:<{width}}{space}║\n' for line in lines])
    box += f'╚{"═" * (width + indent * 2)}╝'  # lower_border
    print(box)

def print_inf_res(beta, p_value, beta_CI, name='2SLS', precision=4):
    """Print the Result of Causal Inference."""

    np.set_printoptions(precision=4)

    if name=="2SLS":
        m = "x = z^T theta + omega; \n" "y = beta x + z^T alpha + epsilon. \n"
    elif name=="2SIR":
        m = "ψ(x) = z^T theta + omega; \n" "y = beta ψ(x) + z^T alpha + epsilon. \n"
    else:
        m = "\n \n"
        "Print function only support 2SLS and 2SIR."
    msg = m + \
         "--- \n" \
         "beta: causal effect from x to y. \n" \
         "--- \n" \
         "Est beta (CI): %.3f (CI: %s) \n" \
         "p-value: %.4f, -log10(p): %.4f" %(beta, beta_CI, p_value, -np.log10(p_value))

    print_msg_box(msg, indent=1, title=name)