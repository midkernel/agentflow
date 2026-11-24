from __future__ import annotations

from pathlib import Path

from agentflow.agents.base import AgentAdapter
from agentflow.agents.secrets import wrap_secret_files
from agentflow.env import merge_env_layers
from agentflow.prepared import ExecutionPaths, PreparedExecution
from agentflow.specs import NodeSpec, RepoInstructionsMode, ToolAccess


ANTIGRAVITY_EFFORTS = ("low", "medium", "high", "xhigh", "max")


def antigravity_print_timeout(timeout_seconds: int) -> str:
    """Return the ``--print-timeout`` value for a node timeout.

    The CLI's own deadline fires slightly before AgentFlow's hard kill so that
    ``agy`` still gets to emit its final ``result`` event (``status: ERROR``,
    ``finished_reason: print_timeout``) with whatever partial output exists.
    """
    grace = min(30, max(1, timeout_seconds // 20))
    return f"{max(1, timeout_seconds - grace)}s"


class AntigravityAdapter(AgentAdapter):
    """Run Google's Antigravity CLI (``agy``) once, headless, per node attempt.

    ``agy`` authenticates through its own OAuth login stored under the user's
    home directory; AgentFlow never reads or forwards those credentials.
    """

    def prepare(self, node: NodeSpec, prompt: str, paths: ExecutionPaths) -> PreparedExecution:
        if node.mcps:
            raise ValueError(
                "antigravity adapter does not support `mcps`. Configure MCP servers with "
                "`agy mcp add ...` on the execution host instead."
            )
        self.validate_node_features(node)

        effort = node.model_settings.reasoning_effort
        if effort is not None and effort not in ANTIGRAVITY_EFFORTS:
            raise ValueError(
                f"antigravity `reasoning_effort` must be one of {', '.join(ANTIGRAVITY_EFFORTS)}; got {effort!r}"
            )

        provider = self.provider_config(node.provider, node.agent)
        executable = node.executable or "agy"
        repo_instructions_ignored = node.repo_instructions_mode == RepoInstructionsMode.IGNORE

        command: list[str] = [
            executable,
            "--print",
            prompt,
            "--output-format",
            "stream-json",
            "--print-timeout",
            antigravity_print_timeout(node.timeout_seconds),
        ]
        if node.tools == ToolAccess.READ_ONLY:
            # Plan mode only lets the agent read the workspace; it records its plan
            # in the CLI's own state directory rather than editing project files.
            command.extend(["--mode", "plan"])
        else:
            command.append("--dangerously-skip-permissions")
        if node.model:
            command.extend(["--model", node.model])
        if effort:
            command.extend(["--effort", effort])
        if repo_instructions_ignored:
            # Launch from the runtime dir so workspace instruction files are not
            # discovered from cwd, keep the project reachable, and stop print mode
            # from expanding slash commands or skills.
            command.extend(["--add-dir", paths.target_workdir, "--disable-slash-commands"])
        command.extend(node.extra_args)

        env = merge_env_layers(getattr(provider, "env", None), node.env)
        cwd = paths.target_workdir
        runtime_files: dict[str, str] = {}
        if repo_instructions_ignored:
            cwd = str(Path(paths.target_runtime_dir))
            # Materializing a file guarantees the runtime dir exists before launch.
            runtime_files[self.relative_runtime_file(".antigravity-cwd")] = ""
        prepared = PreparedExecution(
            command=command,
            env=env,
            cwd=cwd,
            trace_kind="antigravity",
            runtime_files=runtime_files,
        )
        return wrap_secret_files(node, prepared, paths)
