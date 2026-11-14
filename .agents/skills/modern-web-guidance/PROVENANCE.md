# Modern Web Guidance provenance

Repository-local snapshot of [Google Chrome Modern Web Guidance](https://github.com/GoogleChrome/modern-web-guidance), installed for frontend development assistance on 2026-11-14.

- Distribution commit: `4dd968349e54ca45a05dd5c818fb035c9054b022`.
- Source directory: [`skills/modern-web-guidance`](https://github.com/GoogleChrome/modern-web-guidance/tree/4dd968349e54ca45a05dd5c818fb035c9054b022/skills/modern-web-guidance).
- Skill version: `2026_11_09-5bcc021b`; 209 guide documents plus `SKILL.md`.
- Optional CLI: `modern-web-guidance@1.0.0`, with `DISABLE_TELEMETRY=1` for every invocation. The package is not an application dependency.
- npm tarball SHA-256: `bd7aa623ad10710a9ad6678fdbcc0b890c903c54fd6abfc126ff42de01f0e577`.

The pinned subtree was downloaded using the Codex skill-installer helper with an explicit repository-local destination. No npm installer, global configuration, lifecycle script or telemetry command ran. The 209 guides are unchanged. `SKILL.md` is locally modified to pin CLI commands, disable telemetry, permit bundled-guide use offline, avoid automatic updates and respect existing project/user permissions. This is the only modified upstream file. `UPSTREAM.json` records all original file hashes and the local skill hash.

Codex and current Copilot discover `.agents/skills/modern-web-guidance`. Relative symlinks from `.claude/skills/modern-web-guidance` and `.github/skills/modern-web-guidance` expose the same files to Claude and Copilot surfaces that use those directories. Checkouts must preserve symlinks. Agent support varies by version; installation does not demonstrate that a particular running agent has loaded the skill. Start a new turn/session after updating the checkout.

The guide snapshot can be searched and read without installing or executing the optional CLI. Updating the snapshot or CLI requires an intentional reviewed version change, preserving attribution and these local adjustments. No application source, package manifests, lockfiles, deployment settings or browser policy is changed by this installation. Upstream performance-uplift figures are not validated by this installation.

## License and attribution

Per the [pinned upstream license statement](https://github.com/GoogleChrome/modern-web-guidance/blob/4dd968349e54ca45a05dd5c818fb035c9054b022/README.md#license), software is licensed under Apache-2.0 and documentation/guide content under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/), unless otherwise noted. The upstream Apache license is retained as `LICENSE`; it does not replace the documentation license.

Credit: Google Chrome Modern Web Guidance contributors. Portions of the documentation derive from [MDN Web Docs](https://developer.mozilla.org/) by Mozilla Contributors and [W3C](https://www.w3.org/), [WHATWG](https://whatwg.org/) and [IETF](https://www.ietf.org/) specifications. Original per-guide references and notices are retained. Midkernel made only the `SKILL.md` adjustments described above; no endorsement is implied.
