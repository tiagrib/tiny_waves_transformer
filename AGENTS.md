# MetaKitchen Agent Guide

This is a multi-repository workspace. Each sub-repo has its own `.git` and may have its own agent instruction files.

## Agent Roles

| Role | Entry point | Purpose |
|---|---|---|
| **Orchestrator** | `metak-orchestrator/AGENTS.md` | Plans, delegates, reviews. Never writes app code. |
| **Worker** | `<repo>/AGENTS.md` + `<repo>/CUSTOM.md` | Implements tasks inside a single repo. |

## Structure

```
meta-repo/
├── AGENTS.md                    <- you are here
├── metak-shared/                <- shared context (architecture, API contracts, glossary)
├── metak-orchestrator/          <- coordination workspace (TASKS.md, STATUS.md, EPICS.md)
├── repo-*/                      <- application sub-repos
└── meta.code-workspace          <- VS Code multi-root workspace file
```

## Agent Rules

1. **Read this file and any agent instructions in your working repo before starting work.**
2. **`metak-shared/` is read-only.** Never modify files there without explicit user approval.
3. **One agent, one repo.** Do not work across multiple repos in a single session. Use the orchestrator pattern for cross-repo work.
4. **Orchestrated tasks:** Read your assignment from `metak-orchestrator/TASKS.md`. Update `metak-orchestrator/STATUS.md` when done or blocked.
5. **API contracts live in `metak-shared/api-contracts/`.** Always reference these for schemas — never import directly from another repo's source. Before starting implementation, verify that your work matches the current contract version.
6. **Contract verification:** If you discover a discrepancy between a contract and the running system, stop and report it to the orchestrator — do not silently deviate.
7. **Consult `metak-shared/architecture.md`** for system boundaries and service interactions.

## Project Structure

Each sub-repo should maintain:
- `STRUCT.md` — a map of the repo's file/folder structure with brief descriptions. Keep it updated as you add or move files.
- `TODO.md` — a local backlog of improvements, tech debt, or ideas discovered during work.

## When Stuck

1. Re-read the task acceptance criteria and relevant contract.
2. Check `metak-shared/LEARNED.md` for known solutions.
3. Search the codebase for similar patterns.
4. If still blocked after these steps, update `metak-orchestrator/STATUS.md` with what you tried and mark the task as blocked.

## Coding Standards

See `metak-shared/coding-standards.md` for full standards. Key rules:
- All code must pass linting and tests before committing.
- Commit messages follow Conventional Commits: `type(scope): description`
- Never import directly from another repo's source code.

## Custom Instructions

Read and follow `CUSTOM.md` at the repository root for project-specific rules.
