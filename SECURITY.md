# Security Policy

## Reporting a Vulnerability

If you discover a security issue in this repository, please avoid posting sensitive details in a public GitHub issue immediately.

Instead, please report it privately to the repository maintainer through GitHub security reporting if enabled, or through a private maintainer contact channel.

When reporting, include:

- a clear description of the issue
- the affected file, module, or script
- steps to reproduce when possible
- the potential impact
- any suggested remediation if you have one

## Scope

This repository is a research codebase release, so the most relevant security concerns are:

- accidental exposure of secrets or credentials
- unsafe handling of external commands or shell execution
- insecure defaults in scripts that interact with remote services
- path handling that could cause unsafe filesystem behavior

## Sensitive Material

Please do not include live API keys, personal access tokens, private dataset links, or other secrets in public issues, pull requests, or screenshots.

If you accidentally exposed a credential in a fork, local log, or conversation, rotate or revoke it as soon as possible.

## Supported Versions

Security fixes, when applicable, are expected to land on the default branch of this public release. There is currently no long-term support branch policy for older snapshots.
