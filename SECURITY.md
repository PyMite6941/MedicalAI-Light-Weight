# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| latest  | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do not** open a public issue
2. Email: **greshamd27@gmail.com** with subject "SECURITY: [repo-name]"
3. Include steps to reproduce, impact assessment, and any proof-of-concept
4. Allow 90 days for remediation before public disclosure

## Scope

This policy covers:
- Application-level vulnerabilities (auth, authorization, input validation)
- API security issues
- Dependency vulnerabilities (when exploitable)
- LLM-specific: prompt injection, jailbreak, data extraction vulnerabilities

Out of scope:
- Issues requiring physical access
- Social engineering attacks
- Vulnerabilities in third-party services (report to them directly)

## Response Timeline

- **Acknowledgment**: Within 48 hours
- **Triage**: Within 7 days
- **Fix target**: 30 days for critical, 90 days for high/medium
- **Disclosure**: Coordinated after fix deployed

## Attribution

Credit will be given in release notes unless reporter requests anonymity.
