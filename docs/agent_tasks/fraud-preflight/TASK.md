# Reproduce the fraud project's local preflight

Task: `fraud-preflight` | Executor: Gemini/Antigravity | Reviewer: Codex
Repository: https://github.com/sachin2967/fraud-risk-scoring-pipeline
Branch: `gemini/fraud-preflight` | Base: `f30a1d6ab93c50244e4a4a5b96d880e4de86bbe9`

## Objective

Read the repository README and evaluation/operations instructions. Reproduce the existing local checks without changing application code, tests, datasets, model artifacts or evaluation claims. Write a concise PREFLIGHT.md beside this task contract describing the serving/UI entry points, observed test outcomes, missing prerequisites, and one bounded next implementation task for Codex to review. This first task validates the Gemini handoff before feature work.

## Allowed changes

- `docs/agent_tasks/fraud-preflight/PREFLIGHT.md`

This contract and `docs/agent_tasks/fraud-preflight/RESULT.md` are also in scope. This is a coordination
contract, not an OS sandbox. Never edit another agent's checkout or branch.
Read repository instructions; retain frozen evaluations and disclosed limitations.
Do not modify tests merely to make a check pass, remove checks, or weaken assertions.
Do not use paid APIs, deploy, retrain models, install dependencies, spawn agents,
access secrets, or widen scope unless this task explicitly authorizes it.

## Acceptance criteria

- Existing source, tests, evaluation files and artifacts remain unchanged.
- PREFLIGHT.md cites actual repository paths and check outcomes, distinguishing observed results from suggestions.
- RESULT.md records the changes, actual checks, unresolved issues, and next review action.
- Publish the task documentation on the gemini/fraud-preflight branch and report its pushed SHA and draft PR or compare URL. Do not merge.

## Required checks

- argv: `["{python}", "-m", "compileall", "-q", "src", "scripts", "tests"]`; timeout: 60s
- argv: `["{python}", "-m", "pytest", "-q"]`; timeout: 300s

Execute argv lists directly, without a shell. They run from the task worktree.
`{python}` means the interpreter used by the handoff helper. Missing dependencies
are a blocker; do not silently install them. Record genuine failures and timeouts.
Do not claim live-model quality or cloud deployment from offline tests.

## Execution and publication

1. Verify branch/base and inspect status. If the branch's remote moved, fetch and
   preserve the newer work before editing. One active writer owns this branch.
2. Implement this task only. Write `docs/agent_tasks/fraud-preflight/RESULT.md` with the change, each check's
   actual outcome, remaining issues and the next review action. Do not mark the
   task accepted: Codex reviews the implementation and evidence.
3. Run the helper (local command shown below) to check scope and execute the planned
   checks. Fix within scope, at most twice. Stop with a useful report if blocked.
4. Inspect the diff for secrets, private data, model binaries and unrelated work.
   Stage only the named task files and allowed changes; commit normally. Fetch
   immediately before pushing this branch. No amend, force push, reset or merge.
   Routine commits/pushes are already owner-authorized. Push tested milestones.
5. Update an existing PR for this task branch, or open one draft PR against the
   base review branch if none exists. Do not create a duplicate. If GitHub tools
   are unavailable, push the branch and report the compare URL plus the blocker;
   do not describe a compare URL as an existing PR. Check CI on the pushed SHA.
6. End with the pushed SHA, PR/compare URL, actual checks, remaining gates, and
   hand ownership to Codex. Claude may review; it must not write concurrently.

Helper argv (JSON; use as a process argument list):

```json
["C:\\Program Files\\Python311\\python.exe", "C:\\Users\\sachi\\Documents\\Project\\ml-portfolio-gemini-setup\\agent_handoff\\handoff.py", "--state", "C:\\Users\\sachi\\AppData\\Local\\ml-portfolio-agent-handoff", "check", "fraud-preflight"]
```

The helper is an optional convenience outside this workspace. If your extension
cannot access it, run the same required checks locally and report that limitation;
do not grant broad filesystem access just to reach the helper.

Base review branch: `codex/evaluation-and-agent-handoff`.
Compare: https://github.com/sachin2967/fraud-risk-scoring-pipeline/compare/codex/evaluation-and-agent-handoff...gemini/fraud-preflight?expand=1

## Stop and report

Stop on missing credentials/dependencies, exhausted model quota, ambiguous scope,
unexpected remote commits, a required paid action, or two unsuccessful fix attempts.
Keep partial work and explain the blocker. A model response or exit code 0 alone
does not establish that implementation, tests, push, CI or review succeeded.
