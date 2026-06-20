"""(deprecated) ``rebase_to_pr`` was removed apr-25.

Empirical observation over 8h of swarm operation: 0 calls to
``rebase_to_pr`` across all specialists. The tool's intent was to let
an agent overwrite its workdir's ``train_gpt.py`` with an external
PR's unpacked source. In practice agents consume the PR library as a
technique-donor reference (`read_pr_library` + `read_pr_source`) and
never wholesale-rebase. The tool has been removed from the SDK
loadout to simplify the agent's mental model.

Operator-side baseline switching is handled by
``--reset-stale-workdirs`` on the supervisor (re-stages every workdir
from the package-root ``train_gpt.py``). That covers the legitimate
"swap baseline" workflow.

The original implementation is preserved in git history; to re-enable
this tool, re-export it from ``tools/__init__.py``, add ``rebase_to_pr``
back to ``agents.base._CUSTOM_TOOL_NAMES`` and ``_bind_tools()``, and
restore the import lines in those two files.
"""
