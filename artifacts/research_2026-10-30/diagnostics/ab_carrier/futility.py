"""Registered futility rule of the carrier A/B at seed 20261030.

Exit 3 (the registered stop) if the treatment has neither a lower C nor a lower I
than the control, exit 0 otherwise.  Any other status (1 for an uncaught Python
exception, 2 for an unopenable script) is a failure of the check, not a decision.
"""
import json
import sys


def counts(path):
    d = json.load(open(path))["replay_distances"]
    return d["without_structure"] + d["at_most_0.01"], d["without_structure"] + d["at_most_0.5"]


c_control, i_control = counts(sys.argv[1])
c_treat, i_treat = counts(sys.argv[2])
print("control C, I:", c_control, i_control, "| treatment C, I:", c_treat, i_treat)
sys.exit(0 if (c_treat < c_control or i_treat < i_control) else 3)
